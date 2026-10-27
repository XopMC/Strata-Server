// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include "q8_1_il_v53.cuh"
#include "strata/kernels/dp4a.hpp"
namespace swift_wide53_leaf { namespace {
constexpr int WARP=32,WARPS=4,Q8K=32;
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IlQ6K {
    using F = Q6KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[2][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            x.u(kby + r.bq8_offset + 2 * i, iqs % 8, u[i]);
            x.scales(kby + r.bq8_offset + 2 * i, d8[i]);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[2] = {u[0][c], u[1][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q6_q8_dot_impl(r.vl, r.vh, uc, r.scales, r.d, dc);
        }
    }
};

template <typename I, int NC, int R>
__launch_bounds__(4 * WARP)
__global__ void native_mmvq_il_kernel(const typename I::F::Block* __restrict__ w, const int* __restrict__ xq,
                                      const float* __restrict__ xd, float* __restrict__ y, int n_in, int n_out) {
    using F = typename I::F;
    constexpr int SUB = WARP / F::T;   // weight blocks a chunk covers
    const int lane = int(threadIdx.x) & (WARP - 1);
    const auto x = q8_1_cols<NC, I::PM>(xq, xd, n_in / Q8K);
    const int bpr = n_in / F::DIV;
    const int kqs = F::kqs(lane);
    const int nm = (bpr + SUB - 1) / SUB;
    // A grid-stride loop over the row groups, though the grid gives each warp one: without it ptxas gives several
    // instances fewer registers (the fork measured Q6_K's at 3 columns 10-50% slower).
    for (int grp = int(blockIdx.x) * 4 + (int(threadIdx.x) >> 5); grp * R < n_out; grp += int(gridDim.x) * 4) {
        const int row0 = grp * R;
        float acc[4][R][NC];
#pragma unroll
        for (int v = 0; v < 4; ++v)
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[v][i][c] = 0.0f;
        for (int m0 = 0; m0 < nm; m0 += 4) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int kbx = SUB * (m0 + v) + lane / F::T;
                if (kbx < bpr) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        const int row = min(row0 + i, n_out - 1);   // a partial last group recomputes its last row
                        const typename F::W wv = F::load(w + std::size_t(row) * bpr + kbx, kqs);
                        float o[NC];
                        I::template apply<NC>(wv, x, kbx * F::KBY, kqs, o);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[v][i][c] += o[c];
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float s = acc[0][i][c];
                s += acc[1][i][c];
                s += acc[2][i][c];
                s += acc[3][i][c];
                s = warp_sum(s);
                if (lane == 0 && row0 + i < n_out) y[std::size_t(c) * n_out + row0 + i] = s;
            }
    }
}

template <typename I, int R>
void launch_il(const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const auto* w = static_cast<const typename I::F::Block*>(weights);
    const unsigned blocks = unsigned(((n_out + R - 1) / R + 3) / 4);
    if (ncols == 1) {   // 143-dense: one column reads the plain q8_1 blocks (x_il is then the caller's x_q8_1)
        native_mmvq_il_kernel<I, 1, R><<<blocks, 4 * WARP, 0, s>>>(w, static_cast<const int*>(x_il), nullptr, y, n_in, n_out);
        return;
    }
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    const int* xq = I::PM ? parts.pm : parts.bm;
    const float* xd = parts.d;
    switch (ncols) {
#define STRATA_IL_CASE(N) case N: native_mmvq_il_kernel<I, N, R><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); break;
        STRATA_IL_CASE(2) STRATA_IL_CASE(3) STRATA_IL_CASE(4)
#undef STRATA_IL_CASE
        // 147-verify: 5..8 columns (the cp = 8 copy) at 1 or 2 rows a warp; launch_il_rows maps 4 rows to 2 there
#define STRATA_IL_CASE(N) case N: if constexpr (R <= 2) native_mmvq_il_kernel<I, N, R><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); break;
        STRATA_IL_CASE(5) STRATA_IL_CASE(6) STRATA_IL_CASE(7) STRATA_IL_CASE(8)
#undef STRATA_IL_CASE
        default: throw std::invalid_argument("native_mmvq_il requires 1 <= ncols <= 8");
    }
}

template <typename I>
void launch_il_rows(int r, const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    if (ncols > 4 && r == 4) r = 2;   // 5..8 columns: 1 or 2 rows a warp
    switch (r) {
    case 1: launch_il<I, 1>(weights, x_il, y, n_in, n_out, ncols, s); break;
    case 4: launch_il<I, 4>(weights, x_il, y, n_in, n_out, ncols, s); break;
    default: launch_il<I, 2>(weights, x_il, y, n_in, n_out, ncols, s); break;
    }
}

}
// Private experiment only: original wide launcher policy is unchanged.
// In the native wrapper T5..8, requested4 means effective2; no R4 kernel
// is instantiated or silently presented as a distinct numerical layout.
inline void launch_original_wide(const void*w,const void*il,float*y,int n_in,int rows,int T,int requested,cudaStream_t stream){
    launch_il_rows<IlQ6K>(requested,w,il,y,n_in,rows,T,stream);
}
} // namespace swift_wide53_leaf
