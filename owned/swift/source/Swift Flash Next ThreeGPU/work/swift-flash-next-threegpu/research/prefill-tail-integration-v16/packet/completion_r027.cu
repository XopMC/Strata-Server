// Compile with EXACT original R56 flags, including -use_fast_math.
// SOURCE PROPOSAL ONLY. Actual bit/canary/graph/wholechain proof remains pending.
#include "tail_api.hpp"
#include "quantize.cuh"
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace swift_tail_r027 {
namespace {
// Original MMQ/fixup AND R56 TUs use -use_fast_math (--ftz=true).
// Explicit FTZ addition preserves the original fixup partial/add boundary.
__device__ __forceinline__ float fixup_add(float a, float b) {
    float r;
    asm volatile("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float boundary(float x) {
    float r; asm volatile("mov.b32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float completed_scalar(
        const float* raw, const float* partial, const TileSpan* spans, const int32_t* previous,
        int64_t absolute_row, int r, int group, int local_row, int ng, int J, int ntx, int64_t ld) {
    const int it = r / 128, i = r % 128, jt = local_row / J, j = local_row % J;
    const int tile = (it * ng + group) * ntx + jt;
    const TileSpan s = spans[tile];
    const float base = raw[absolute_row * ld + r];
    if (s.count == 0) return base; // Do not change signedzero with an artificial +0.
    float sum = 0.0f;
    for (uint32_t n = 0; n < s.count; ++n) {
        const int b = previous[s.offset + n];
        sum = fixup_add(sum, partial[size_t(b) * J * 128 + j * 128 + i]);
    }
    return boundary(fixup_add(base, sum));
}
__device__ __forceinline__ float hidden(
        const float* raw, const float* partial, const TileSpan* spans, const int32_t* previous,
        int64_t ar, int k, int group, int local_row, int ng, int J, int ntx, int64_t ld,
        float sg, float su, float* receipt, int64_t receipt_ld) {
    const float gate = completed_scalar(raw, partial, spans, previous, ar, k,
                                       group, local_row, ng, J, ntx, ld);
    const float up = completed_scalar(raw, partial, spans, previous, ar, 640 + k,
                                     group, local_row, ng, J, ntx, ld);
    if (receipt) { receipt[ar * receipt_ld + k] = gate; receipt[ar * receipt_ld + 640 + k] = up; }
    // Literal original R56 scale/SwiGLU and rounded FP32 boundary.
    const float g = gate * sg, u = up * su;
    return boundary((g / (1.f + __expf(-g))) * u);
}
__global__ void tail_complete_r56_d4(
        const float* __restrict__ raw, int64_t ld, int64_t nr,
        const int32_t* __restrict__ bounds, int ng, const float* __restrict__ tails, int64_t row0,
        const float* __restrict__ partial, const TileSpan* __restrict__ spans,
        const int32_t* __restrict__ previous, int J, int ntx,
        void* __restrict__ vy, float* __restrict__ row_sd,
        float* __restrict__ receipt, int64_t receipt_ld) {
    constexpr int vals_per_scale = 32;
    const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
    constexpr int64_t ne0 = 1024, ne00 = 640;
    if (i0 >= ne0) return;
    ggml_cuda_pdl_sync();
    const int64_t row = blockIdx.x, ar = row0 + row;
    float sg = 0.f, su = 0.f;
    int e = 0, local_row = 0;
    if ((threadIdx.x & 31) == 0) {
        int hi = ng;
        while (e + 1 < hi) { const int mid = (e + hi) / 2; if (ar >= bounds[mid]) e = mid; else hi = mid; }
        sg = tails[4 * e]; su = tails[4 * e + 1]; local_row = int(ar - bounds[e]);
    }
    sg = __shfl_sync(0xffffffffu, sg, 0); su = __shfl_sync(0xffffffffu, su, 0);
    e = __shfl_sync(0xffffffffu, e, 0); local_row = __shfl_sync(0xffffffffu, local_row, 0);
    if (row_sd && blockIdx.y == 0 && threadIdx.x == 0) row_sd[row] = tails[4 * e + 2];
    const float4 xi = i0 < ne00 ? make_float4(
        hidden(raw, partial, spans, previous, ar, i0, e, local_row, ng, J, ntx, ld, sg, su, receipt, receipt_ld),
        hidden(raw, partial, spans, previous, ar, i0+1, e, local_row, ng, J, ntx, ld, sg, su, receipt, receipt_ld),
        hidden(raw, partial, spans, previous, ar, i0+2, e, local_row, ng, J, ntx, ld, sg, su, receipt, receipt_ld),
        hidden(raw, partial, spans, previous, ar, i0+3, e, local_row, ng, J, ntx, ld, sg, su, receipt, receipt_ld))
        : make_float4(0.f, 0.f, 0.f, 0.f);
    float amax = fabsf(xi.x); amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z)); amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, WARP_SIZE));
    const float d_inv = 127.0f / amax;
    char4 q; q.x=roundf(xi.x*d_inv); q.y=roundf(xi.y*d_inv); q.z=roundf(xi.z*d_inv); q.w=roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;
    auto* y = (block_q8_1_mmq*)vy;
    const int64_t k_block = i0 / QK8_1_MMQ, iqs = i0 % QK8_1_MMQ;
    const int64_t ib = k_block * nr + row;
    ((char4*)y[ib].qs)[iqs/4] = q;
    if (iqs % 32 == 0) y[ib].d4[iqs/32] = d;
}
void need(bool x, const char* s) { if (!x) throw std::runtime_error(s); }
bool overlap(const void* a, size_t na, const void* b, size_t nb) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    if (!na || !nb) return false;
    if (x + na < x || y + nb < y) throw std::runtime_error("R027 interval overflow");
    return x < y + nb && y < x + na;
}
} // namespace

void enqueue_completion(const strata::prefill::mmq::Product& p, const Shape& s, const Binding& b) {
    const auto& g = s.geometry;
    need(b.owner == g.owner && b.owner >= 0 && b.owner < 3 && b.stream && b.epoch &&
         b.identity_ids == p.ids && b.identity_ids && p.type == 40 && p.n == g.groups &&
         p.w_rows == 1280 && p.w_cols == 2560 && p.expert_bytes == 1843200 && !p.tab && !p.y_scale &&
         p.total_rows == g.total_rows && p.max_rows == g.max_rows && p.ld_dst == g.ld_dst &&
         (p.ld_dst == 1280 || p.ld_dst == 2560) && p.dst && p.bounds && b.tails && b.hq &&
         b.nr > 0 && b.row0 >= 0 && b.nr <= p.total_rows && b.row0 <= p.total_rows - b.nr &&
         b.device_spans && b.span_count == size_t(g.tiles) && b.previous_count <= size_t(g.grid) &&
         (!b.previous_count || b.device_previous_blocks) && b.partial_capacity >= s.partial_bytes &&
         (!s.partial_bytes || b.partial), "R027 final owning shape/lease mismatch");
    const size_t qbytes = size_t(1024 / QK8_1_MMQ) * size_t(b.nr) * sizeof(block_q8_1_mmq);
    const size_t rawbytes = size_t(p.total_rows) * size_t(p.ld_dst) * sizeof(float);
    need(b.hq_capacity >= qbytes && (!b.row_sd || b.row_sd_capacity >= size_t(b.nr) * 4),
         "R027 final output capacity");
    auto output_is_disjoint = [&](const void* output, size_t bytes) {
        return !overlap(output,bytes,p.dst,rawbytes) &&
            !overlap(output,bytes,b.partial,s.partial_bytes) &&
            !overlap(output,bytes,b.device_spans,b.span_count*sizeof(TileSpan)) &&
            !overlap(output,bytes,b.device_previous_blocks,b.previous_count*sizeof(int32_t)) &&
            !overlap(output,bytes,b.tails,size_t(p.n)*4*sizeof(float)) &&
            !overlap(output,bytes,p.bounds,size_t(p.n+1)*sizeof(int32_t)) &&
            !overlap(output,bytes,p.w,size_t(p.n)*p.expert_bytes);
    };
    need(output_is_disjoint(b.hq,qbytes), "R027 final output clobbers live input");
    if (b.row_sd) {
        const size_t bytes = size_t(b.nr)*4;
        need(output_is_disjoint(b.row_sd,bytes) && !overlap(b.row_sd,bytes,b.hq,qbytes),
             "R027 row scale output clobbers live input/output");
    }
    if (b.diagnostic_completed_gu) {
        need(b.diagnostic_ld >= 1280 && b.diagnostic_ld <= 2560,
             "R027 diagnostic row stride");
        const size_t bytes = size_t(p.total_rows) * size_t(b.diagnostic_ld) * 4;
        need(b.diagnostic_capacity >= bytes && output_is_disjoint(b.diagnostic_completed_gu,bytes) &&
             !overlap(b.diagnostic_completed_gu,bytes,b.hq,qbytes) &&
             !overlap(b.diagnostic_completed_gu,bytes,b.row_sd,b.row_sd ? size_t(b.nr)*4 : 0),
             "R027 diagnostic allocation must be disjoint");
    }
    tail_complete_r56_d4<<<dim3((unsigned)b.nr,2),128,0,b.stream>>>
        (p.dst,p.ld_dst,b.nr,p.bounds,p.n,b.tails,b.row0,b.partial,b.device_spans,
         b.device_previous_blocks,g.J,g.ntx,b.hq,b.row_sd,b.diagnostic_completed_gu,b.diagnostic_ld);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace swift_tail_r027
