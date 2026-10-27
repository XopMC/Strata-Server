// Source-only v69: private CUDA HC up-projection+mix from Strata PR1370.
// Upstream sergqwer/Strata-pr d8ed55c1f401981f2df7444d1eec577c516a9c17, MIT (LICENSE in source-v69).
// No existing kernels/header/ABI are modified; default OFF via STRATA_HC_UPMIX_V69.
#include "hc_upmix_v69.hpp"
#include "strata/core/emulate.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
namespace strata::prefill {
namespace {
constexpr int N=2560,HC=4,D=N*HC,LR=320;
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }

#if !defined(__HIPCC__)
// S23 on CUDA (sm_80+; opt-in STRATA_HC_UPMIX_V69=1, see hc_upmix in prefill.cpp): the hyper-connection read's up
// projection (lo16 x w_up^T, BF16 in, FP32 accumulate over K = 320) with gr_mix_r as its epilogue, so `gated`
// (T x 10240 FP32, 336 MB for an 8,192-token chunk) is neither written by the GEMM nor read back by the mix - gr_mix_r
// reads R and `gated` at ~96% of the bandwidth already, so only fewer bytes make the read faster.  A block: 128 tokens
// x 32 columns x the 4 streams (128 output columns), K staged in 32-deep slices (cp.async, double-buffered), 8 warps of
// 32 tokens x 16 columns x 4 streams on mma.m16n8k16 (ldmatrix); a lane then holds its (token, column) pairs for all 4
// streams, and the mix runs in registers in gr_mix_r_kernel's order (x_c = R * rs * w; s = fmaf(x_c, sigm(g_c), s);
// s / 4) with the BF16 / FP16 images.  The epilogue's R rows are asked into L2 at the start.  With cuBLAS's up GEMM
// summing K in the same order (CUDA 13.3 on an RTX 5090 from 33 tokens: gr_upmix_parity) `mixed` is the default pair's
// bits.
constexpr int UC_BM = 128, UC_BD = 32;   // tokens, columns a stream
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800   // cp.async, bf16 mma: the kernel is empty before sm_80
constexpr int UC_KC = 32, UC_LD = UC_KC + 8;   // k a slice, a staged row
__device__ __forceinline__ void uc_pf_l2(const float* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ void uc_cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void uc_ldm4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void uc_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif
__global__ void __launch_bounds__(256, 2) gr_upmix_cuda_kernel(const uint16_t* __restrict__ lo16,
                                                              const uint16_t* __restrict__ wu,
                                                              const float* __restrict__ R, const float* __restrict__ rs,
                                                              const float* __restrict__ w, float* __restrict__ mixed,
                                                              uint16_t* __restrict__ mixed16,
                                                              uint16_t* __restrict__ mixed_h,
                                                              int64_t T) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) uint16_t sA[2][UC_BM][UC_LD];
    __shared__ __align__(16) uint16_t sB[2][HC * UC_BD][UC_LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    // blocks column-tile fastest: the blocks in flight share a token tile (its lo16 rows stay in L2, the epilogue's R
    // reads run along rows)
    const int64_t t0 = (int64_t) blockIdx.y * UC_BM;
    const int d0 = blockIdx.x * UC_BD;
    // the epilogue's R rows (128 tokens x 4 streams x 128 bytes) asked into L2 now, so they arrive under the GEMM
    for (int i = tid; i < UC_BM * HC; i += 256) {
        const int64_t t = t0 + i / HC;
        if (t < T) uc_pf_l2(R + t * D + (i % HC) * N + d0);
    }
    auto stage = [&](int kc, int buf) {
        for (int i = tid; i < UC_BM * (UC_KC / 8); i += 256) {
            const int r = i / (UC_KC / 8), q = i % (UC_KC / 8);
            const int64_t t = t0 + r < T ? t0 + r : T - 1;   // the last tile's missing rows read a real row, unused
            uc_cp16(&sA[buf][r][q * 8], lo16 + t * LR + kc * UC_KC + q * 8);
        }
        for (int i = tid; i < HC * UC_BD * (UC_KC / 8); i += 256) {
            const int r = i / (UC_KC / 8), q = i % (UC_KC / 8);
            uc_cp16(&sB[buf][r][q * 8], wu + ((int64_t) (r / UC_BD) * N + d0 + r % UC_BD) * LR + kc * UC_KC + q * 8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[2][HC][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int c = 0; c < HC; ++c)
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[mt][c][s][i] = 0.0f;
    constexpr int NK = LR / UC_KC;
    stage(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int buf = kc & 1;
        if (kc + 1 < NK) {
            stage(kc + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < UC_KC / 16; ++ks) {
            unsigned a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) uc_ldm4(a[mt], &sA[buf][wm * 32 + mt * 16 + (lane & 15)][ks * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                unsigned b[4];   // b[0..1]: columns 0-7 of this warp's 16, b[2..3]: 8-15 (k 0-7, 8-15 each)
                uc_ldm4(b, &sB[buf][c * UC_BD + wn * 16 + (lane & 7) + ((lane >> 4) << 3)][ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    uc_mma(acc[mt][c][0], a[mt], b[0], b[1]);
                    uc_mma(acc[mt][c][1], a[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();   // every read of `buf` before stage(kc + 2) refills it
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t t = t0 + wm * 32 + mt * 16 + (lane >> 2) + h * 8;
            if (t >= T) continue;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int d = d0 + wn * 16 + s * 8 + (lane & 3) * 2;
                float sm[2] = {0.0f, 0.0f};
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float2 r2 = *reinterpret_cast<const float2*>(R + t * D + c * N + d);
                    const float2 w2 = *reinterpret_cast<const float2*>(w + c * N + d);
                    const float rsc = rs[t * HC + c];
                    const float x0 = r2.x * rsc * w2.x, x1 = r2.y * rsc * w2.y;   // gr_mix_r_kernel's order
                    sm[0] = fmaf(x0, sigm(acc[mt][c][s][h * 2 + 0]), sm[0]);
                    sm[1] = fmaf(x1, sigm(acc[mt][c][s][h * 2 + 1]), sm[1]);
                }
                sm[0] /= (float) HC;
                sm[1] /= (float) HC;
                *reinterpret_cast<float2*>(mixed + t * N + d) = make_float2(sm[0], sm[1]);
                if (mixed16)
                    *reinterpret_cast<uint32_t*>(mixed16 + t * N + d) =
                        (uint32_t) bf(sm[0]) | ((uint32_t) bf(sm[1]) << 16);
                if (mixed_h)
                    *reinterpret_cast<uint32_t*>(mixed_h + t * N + d) = (uint32_t) hf(sm[0]) | ((uint32_t) hf(sm[1]) << 16);
            }
        }
#endif
}
#endif

}  // namespace
bool gr_upmix_v69(const uint16_t* lo16, const uint16_t* w_up, const float* R, const float* rs,
                  const float* w_norm, float* mixed, uint16_t* mixed16, uint16_t* mixed_h,
                  int64_t T, void* stream) {
    static const bool enabled=[] {
        const char* e=std::getenv("STRATA_HC_UPMIX_V69");
        return e && e[0]=='1' && e[1]=='\0';
    }();
    if(!enabled || T<33 || !lo16 || !w_up || !R || !rs || !w_norm || !mixed || !mixed16 || !mixed_h)
        return false;
#if defined(__HIPCC__)
    return false;
#else
    // Current logical device only; no cudaSetDevice, global one-card assumption or late function attributes.
    struct Meta { bool known=false, supported=false, logged=false; };
    static thread_local Meta meta[64];
    int dev=-1;
    if(cudaGetDevice(&dev)!=cudaSuccess || dev<0 || dev>=64) return false;
    Meta& m=meta[dev];
    if(!m.known) {
        int major=0,shared_limit=0;
        if(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev)!=cudaSuccess ||
           cudaDeviceGetAttribute(&shared_limit,cudaDevAttrMaxSharedMemoryPerBlock,dev)!=cudaSuccess)
            return false;
        m.supported=major>=8 && strata::cc_major_of(major)>=8 && shared_limit>=40960;
        m.known=true;
    }
    if(!m.supported) return false;
    gr_upmix_cuda_kernel<<<dim3(N/UC_BD,(unsigned)((T+UC_BM-1)/UC_BM)),256,0,(cudaStream_t)stream>>>(
        lo16,w_up,R,rs,w_norm,mixed,mixed16,mixed_h,T);
    const cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) {
        std::fprintf(stderr,"prefill gr_upmix_v69: %s\n",cudaGetErrorString(e));
        std::exit(1); // Never execute fallback after a partially issued kernel.
    }
    const char* log=std::getenv("STRATA_HC_UPMIX_V69_LOG");
    if(!m.logged && log && log[0]=='1' && log[1]=='\0') {
        std::fprintf(stderr,"strata v69: HC-upmix dispatch issued CUDA%d T=%lld H2560 HC4 K320 shared40960B\n",dev,(long long)T);
        m.logged=true; // Dispatch evidence only; neither a completion nor a timing claim.
    }
    return true;
#endif
}
}  // namespace strata::prefill
