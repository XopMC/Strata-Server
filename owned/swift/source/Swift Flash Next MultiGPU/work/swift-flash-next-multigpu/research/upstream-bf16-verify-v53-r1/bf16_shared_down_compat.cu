// Compatibility for raw BF16 shared-expert DOWN weights only.
// Helpers below are copied literally from the selected shared_expert.cu producer.
// Existing selected native shared_expert object and all its kernels stay unchanged.
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int THREADS = 128;
__global__ void native_swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply.
    // Explicit intrinsics reproduce its --use_fast_math operations without
    // changing compilation of the default legacy arithmetic in this file.
    out[i] = __fdividef(gate[i], 1.0f + __expf(-gate[i])) * up[i];
}

__device__ __forceinline__ double warp_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

__global__ void scalar_gate_kernel(const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ w_bf16,
                                   float* __restrict__ out, int n_embd) {
    __shared__ double scratch[8];   // 8 warps: the launch is <<<1, 256>>>
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        acc += (double) f32_from_bf16(x_bf16[i]) * (double) f32_from_bf16(w_bf16[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    acc = warp_sum_d(acc);
    if (lane == 0) scratch[warp] = acc;
    __syncthreads();
    const int nw = ((int) blockDim.x + 31) >> 5;
    if (warp == 0) {
        acc = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0;
        acc = warp_sum_d(acc);
        if (threadIdx.x == 0) out[0] = (float) (1.0 / (1.0 + exp(-acc)));
    }
}

__global__ void native_scalar_sigmoid_kernel(float* gate) {
    // Match the single-token CUDA sigmoid's FP32 fast-math operations without changing legacy kernels'
    // compilation flags. The dot product was already reduced by the pinned native MMVF implementation.
    gate[0] = __fdividef(1.0f, 1.0f + __expf(-gate[0]));
}

__global__ void scale_rows_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[(size_t) t * n + i] *= g[t];
}

__global__ void sigmoid_scale_rows_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gt = __fdividef(1.0f, 1.0f + __expf(-g[t]));
        out[(size_t) t * n + i] *= gt;
    }
}
}  // namespace

void shared_expert_multi_bf16_down_compat(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* down_bf16, const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream) {
    // Row kernels are dynamic; native MMVQ and BF16 MMVF already split9..16 into unchanged<=8 launches.
    if (n_tok < 1 || n_tok > kVerifyMaxT || !nw.q8_1 || !nw.gate_data || !nw.up_data || !down_bf16 || !stream)
        throw std::invalid_argument("shared_expert_multi_bf16_down_compat: needs native gate/up, BF16 down, scratch and a stream");
    const bool native_bf16 = shared_expert_native_bf16_enabled();
    cudaStream_t cs = (cudaStream_t) stream;
    const void* x_q8_1 = nw.x_q8_1;
    if (!x_q8_1) {
        native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
        x_q8_1 = nw.q8_1;
    }
    native_mmvq(nw.gate_type, nw.gate_data, x_q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, x_q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    // BF16 DOWN consumes the original F32 SwiGLU result, without an intermediate quantization.
    const int n = (int) (n_ff * n_tok);
    native_swiglu_kernel<<<(unsigned) ((n + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate, up, gate, n);
    bf16_gemv_fp32_mmvf_multi(gate, n_ff, down_bf16, out, n_embd, n_ff, n_embd, n_tok, stream);
    static const bool batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    if (native_bf16 && batch && n_tok > 1) {   // one gemv for all rows (outputs identical), fused sigmoid+scale
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
        sigmoid_scale_rows_kernel<<<dim3((unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok), THREADS, 0, cs>>>(
            out, g, (int) n_embd);
    } else if (native_bf16 && n_tok == 1) {
        bf16_gemv_fp32_mmvf(x, gate_inp_bf16, g, n_embd, 1, stream);
        sigmoid_scale_rows_kernel<<<dim3((unsigned) ((n_embd + THREADS - 1) / THREADS), 1u), THREADS, 0, cs>>>(
            out, g, (int) n_embd);
    } else {
        for (int t = 0; t < n_tok; ++t) {
            if (native_bf16) {
                bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
                native_scalar_sigmoid_kernel<<<1, 1, 0, cs>>>(g + t);
            } else {
                scalar_gate_kernel<<<1, 256, 0, cs>>>(x_bf16 + (size_t) t * n_embd, gate_inp_bf16, g + t, (int) n_embd);
            }
        }
        scale_rows_kernel<<<dim3((unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok), THREADS, 0, cs>>>(
            out, g, (int) n_embd);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("shared_expert_multi_bf16_down_compat: ") + cudaGetErrorString(e));
}
}  // namespace strata::kernels
