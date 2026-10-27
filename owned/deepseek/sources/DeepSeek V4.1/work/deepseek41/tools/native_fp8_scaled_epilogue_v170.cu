// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project; native raw-pointer adaptation 2026.
// Adapted from vLLM v0.8.5 csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh
// and csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp.
// https://github.com/vllm-project/vllm/tree/v0.8.5/csrc/quantization/cutlass_w8a8
// CUTLASS headers retain their NVIDIA BSD-3-Clause license independently.
// This prototype is not connected to the model runtime. No Torch dependency.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace {
thread_local std::string last_error;
using Input = cutlass::float_e4m3_t;
using Output = cutlass::bfloat16_t;
using Tile = cutlass::gemm::GemmShape<64, 128, 64>;
using Warp = cutlass::gemm::GemmShape<32, 64, 64>;
using Instruction = cutlass::gemm::GemmShape<16, 8, 32>;
using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    Tile, Warp, float, 4, 1>;
using ScaleA = cutlass::epilogue::threadblock::VisitorColBroadcast<
    ThreadMap, float, cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>, false>;
using ScaleB = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    ThreadMap, float, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>, false>;
using MultiplyScales = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using ScaleProduct = cutlass::epilogue::threadblock::Sm80EVT<
    MultiplyScales, ScaleA, ScaleB>;
using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
using MultiplyAndRound = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, Output, float, cutlass::FloatRoundStyle::round_to_nearest>;
// Explicit tree: BF16_RN(accumulator * (activation_scale * weight_scale)).
// This intentionally differs from upstream's (accumulator * weight_scale) *
// activation_scale. No FastAccum operator or split-K is enabled.
using Scaled = cutlass::epilogue::threadblock::Sm80EVT<
    MultiplyAndRound, Accumulator, ScaleProduct>;
using OutputStride = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
    ThreadMap, Output, cutlass::FloatRoundStyle::round_to_nearest, OutputStride>;
using Epilogue = cutlass::epilogue::threadblock::Sm80EVT<Store, Scaled>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    Input, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
    Input, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
    float, cutlass::layout::RowMajor, 4, float, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    Tile, Warp, Instruction, Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5,
    cutlass::arch::OpMultiplyAdd, 1>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

void check(cutlass::Status result, const char* operation) {
    if (result != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cutlassGetStatusString(result));
}
void validate(int rows, int k, int n, int valid_rows) {
    if (rows < 16 || rows > 4096 || rows % 16 || k < 32 || k > 8192 || k % 32 ||
        n < 16 || n > 32768 || n % 16 || valid_rows < 1 || valid_rows > rows)
        throw std::invalid_argument("v170 requires aligned FP8 rows/K/N and valid BF16 rows");
}
Gemm::Arguments arguments(const uint8_t* x, const uint8_t* w, const float* xs,
                         const float* ws, int rows, int k, int n,
                         uint16_t* out, int valid_rows) {
    validate(rows, k, n, valid_rows);
    typename ScaleA::Arguments a_scale{xs, 0.f, {}};
    typename ScaleB::Arguments b_scale{ws, 0.f, {}};
    typename ScaleProduct::Arguments scale_args{a_scale, b_scale, {}};
    typename Scaled::Arguments scaled_args{{}, scale_args, {}};
    typename Store::Arguments store_args{
        reinterpret_cast<Output*>(out),
        OutputStride{int64_t(n), cute::Int<1>{}, cute::Int<0>{}}};
    typename Epilogue::Arguments epilogue_args{scaled_args, store_args};
    // Using valid_rows as M masks stores without a padded BF16 output buffer.
    // Encoded input storage may still have `rows` padded rows, as in FP8LT.
    return Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        cutlass::gemm::GemmCoord{valid_rows, n, k}, 1,
        epilogue_args, x, w, nullptr, nullptr,
        0, 0, 0, 0, int64_t(k), int64_t(k), int64_t(n), int64_t(n)};
}
}

extern "C" {
const char* ds41_fp8_scaled_v170_error() { return last_error.c_str(); }

// Metadata inspection does not initialize CUDA or query a device.
int ds41_fp8_scaled_v170_info(int64_t* values, size_t count) {
    if (!values || count < 8) { last_error = "v170 info needs eight values"; return 1; }
    values[0] = 170; values[1] = 89;
    values[2] = Tile::kM; values[3] = Tile::kN; values[4] = Tile::kK;
    values[5] = sizeof(typename Kernel::SharedStorage);
    values[6] = 0;  // split-K disabled
    values[7] = 0;  // FastAccum disabled
    return 0;
}

// Returns SIZE_MAX on validation failure. It neither allocates nor launches.
size_t ds41_fp8_scaled_v170_workspace_bytes(int rows, int k, int n, int valid_rows) {
    try {
        auto args = arguments(nullptr, nullptr, nullptr, nullptr, rows, k, n,
                              nullptr, valid_rows);
        return Gemm::get_workspace_size(args);
    } catch (const std::exception& e) { last_error = e.what(); return SIZE_MAX; }
}

// C ABI alongside ds41_fp8lt_run_bf16: raw pointers, caller-owned workspace,
// caller-supplied stream. The prototype never allocates GPU memory.
int ds41_fp8_scaled_v170_run(const uint8_t* x, const uint8_t* w,
                           const float* xs, const float* ws,
                           int rows, int k, int n, uint16_t* out,
                           int valid_rows, void* workspace,
                           size_t workspace_bytes, uintptr_t stream) {
    try {
        if (!x || !w || !xs || !ws || !out)
            throw std::invalid_argument("v170 requires all input/output pointers");
        for (const void* ptr : {static_cast<const void*>(x), static_cast<const void*>(w),
                               static_cast<const void*>(out)})
            if (reinterpret_cast<uintptr_t>(ptr) % 16)
                throw std::invalid_argument("v170 FP8/output pointers must be 16-byte aligned");
        auto args = arguments(x, w, xs, ws, rows, k, n, out, valid_rows);
        const size_t required = Gemm::get_workspace_size(args);
        if (required > workspace_bytes || (required && !workspace))
            throw std::invalid_argument("v170 caller workspace is insufficient");
        Gemm op;
        check(op.can_implement(args), "CUTLASS can_implement");
        check(op(args, workspace, reinterpret_cast<cudaStream_t>(stream)), "CUTLASS GEMM");
        const auto status = cudaGetLastError();
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
}
