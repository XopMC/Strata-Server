#include "../../../../../../../include/strata_ds/fresh_diagnostic_visibility.hpp"
// Bounded original-FP8 wo_a verifier rows, with one weight load per output/K element.
// Included by wo_a_fp8.cu; do not add a second object for this companion to the same library.
#include "strata/ds41/wo_a_fp8.hpp"
#include "strata/ds41/config.hpp"
#include <cuda_fp8.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::ds41 {
namespace woa_many_v116_detail {
using bf16 = __nv_bfloat16;
constexpr int kRows = kOGroups * kOLora;
constexpr int kIn = kHeads * kHeadDim / kOGroups;
constexpr int kInputStride = kHeads * kHeadDim;
constexpr int kUnroll = 8;

// Same original-code decode and BF16-subnormal rounding as wo_a_fp8.cu:deq.
__device__ __forceinline__ float deq(uint8_t w, uint8_t e) {
    __nv_fp8_e4m3 value;
    value.__x = w;
    const uint32_t bits = e == 0 ? 0x00400000u : (e == 255 ? 0x7FC00000u : uint32_t(e) << 23);
    const float decoded = float(value) * __uint_as_float(bits);
    return e <= 2 ? __bfloat162float(__float2bfloat16_rn(decoded)) : decoded;
}

template<int M>
__global__ void multiply(const bf16* o, const uint8_t* w, const uint8_t* scale, bf16* y) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= kRows) return;
    const int group_offset = (row / kOLora) * kIn;
    const uint8_t* wr = w + int64_t(row) * kIn;
    const uint8_t* sr = scale + (row / 32) * (kIn / 32);
    float acc[M] = {};
    for (int i = lane; i < kIn; i += kUnroll * 32) {
        uint8_t wv[kUnroll], sv[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = wr[i + 32 * u];
            sv[u] = sr[i / 32 + u];
        }
        // Every activation row sees the same i/u sequence as the scalar kernel.
        // Decode once, but do not merge accumulators or change the shuffle tree.
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const float weight = deq(wv[u], sv[u]);
#pragma unroll
            for (int t = 0; t < M; ++t)
                acc[t] += __bfloat162float(o[size_t(t) * kInputStride + group_offset + i + 32 * u]) * weight;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int t = 0; t < M; ++t)
            acc[t] += __shfl_down_sync(0xffffffffu, acc[t], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < M; ++t)
            y[size_t(t) * kRows + row] = __float2bfloat16_rn(acc[t]);
    }
}
}  // namespace woa_many_v116_detail

void wo_a_grouped_fp8_many_v116(const __nv_bfloat16* o, const uint8_t* w, const uint8_t* scale,
                               int m, __nv_bfloat16* y, cudaStream_t stream) {
    if (!o || !w || !scale || !y || m < 1 || m > 6)
        throw std::invalid_argument("wo_a_many_v116 requires original FP8/scales and1..6 rows");
    if (m == 1) {
        wo_a_grouped_fp8(o, w, scale, y, stream);
        return;
    }
    constexpr int blocks = woa_many_v116_detail::kRows / 8;
#define RUN(M) case M: woa_many_v116_detail::multiply<M><<<blocks,256,0,stream>>>(o,w,scale,y); break
    switch (m) { RUN(2); RUN(3); RUN(4); RUN(5); RUN(6); }
#undef RUN
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace strata::ds41

// Actual scalar-vs-many component, using the unchanged scalar kernel in this object.
// Caller owns bounded buffers and stream; no internal allocation or device selection.
extern "C" int ds41_native_woa_many_v116_probe(const uint16_t* input, const uint8_t* weights,
        const uint8_t* scales, int m, uint16_t* output, int serial, void* stream) {
    if (!strata::ds41::publication::metadata_snapshot(1)
            || !input || !weights || !scales || !output || m < 1 || m > 6
            || (serial != 0 && serial != 1)) return 1;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != 0 || !strata::ds41::publication::runtime_snapshot(1)) return 2;
    try {
        using namespace strata::ds41;
        const auto* x = reinterpret_cast<const __nv_bfloat16*>(input);
        auto* y = reinterpret_cast<__nv_bfloat16*>(output);
        const auto selected = reinterpret_cast<cudaStream_t>(stream);
        if (serial) {
            for (int t = 0; t < m; ++t)
                wo_a_grouped_fp8(x + size_t(t) * kHeads * kHeadDim, weights, scales,
                                y + size_t(t) * kOGroups * kOLora, selected);
        } else wo_a_grouped_fp8_many_v116(x, weights, scales, m, y, selected);
        return 0;
    } catch (...) { return 3; }
}
