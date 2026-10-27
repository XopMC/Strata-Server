// Exact source extraction; see EXTRACTION.json. No policy dispatcher.
#include "selector_v7_cuda.hpp"
#include <cuda_runtime.h>
namespace swift_selector_v7_leaf { namespace {
using namespace strata::kernels;
constexpr int R=4,TK_T=1024;
__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}
template <int PER>
__global__ void __launch_bounds__(TK_T) compact_topk_reg_kernel_v7(const float* __restrict__ scores,
                                                              const int32_t* __restrict__ steps, int64_t max_blocks,
                                                              int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        __syncthreads();
        // the digit: the first bin from the top where the count above reaches `width` (bin 0 when none of 255..1
        // does).  Threads 0-255 (warps 0-7) prefix-sum the 256 bins in descending order instead of one thread
        // walking them: the same digit and count above.
        int v = 0, x = 0;
        if (t < 256) {
            v = hist[0][255 - t];
            x = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) s_warp[warp] = x;
        }
        __syncthreads();
        if (t < 256) {
            int inc = x;
            for (int w2 = 0; w2 < warp; ++w2) inc += s_warp[w2];
            const int before = above + inc - v;
            if (before < width && (above + inc >= width || t == 255)) {
                s_digit = 255 - t;
                s_above = before;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}
} // anonymous
cudaError_t final_select(const float*s,const int32_t*t,int q,int64_t stride,int64_t cap,int32_t*out,cudaStream_t cs){
 if(!s||!t||!out||q<1||q>256||stride!=Packed||cap!=2051)return cudaErrorInvalidValue;
 compact_topk_reg_kernel_v7<33><<<q,TK_T,0,cs>>>(s,t,stride,cap,out);
 return cudaGetLastError();
}
void* final_kernel() {return reinterpret_cast<void*>(compact_topk_reg_kernel_v7<33>);}
} // namespace
