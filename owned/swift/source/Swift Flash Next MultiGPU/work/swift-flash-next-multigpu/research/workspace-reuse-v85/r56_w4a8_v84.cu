// SOURCE ONLY. Private current-V77 port of exact R56; current MMQ ABI untouched.
#include "r56_w4a8_v84.hpp"
#include "quantize.cuh"
#include <stdexcept>
#include <cstdio>
namespace strata::prefill::mmq {
namespace {
// Opaque FP32 register boundary prevents cross-kernel-boundary reassociation.
__device__ __forceinline__ float store_boundary_r56(float v) {
    float r; asm volatile("mov.b32 %0, %1;" : "=f"(r) : "f"(v)); return r;
}
__device__ __forceinline__ float h_value_r56(const float* row,int k,float sg,float su) {
    const float g=row[k]*sg,u=row[640+k]*su;
    return store_boundary_r56((g/(1.f+__expf(-g)))*u);
}
__global__ void swiglu_quantize_d4_r56(const float* __restrict__ dm,int64_t ld,int64_t nr,
        const int32_t* __restrict__ bounds,int ng,const float* __restrict__ tails,int64_t row0,
        void* __restrict__ vy,float* __restrict__ diagnostic_h, float* __restrict__ row_sd) {
    constexpr int vals_per_scale=32;
    const int64_t i0=((int64_t)blockDim.x*blockIdx.y+threadIdx.x)*4;
    constexpr int64_t ne0=1024,ne00=640;
    if(i0>=ne0)return;
    ggml_cuda_pdl_sync();
    const int64_t row=blockIdx.x;
    float sg=0.f,su=0.f; int e=0;
    if((threadIdx.x&31)==0){int hi=ng;const int64_t absr=row0+row;
        while(e+1<hi){const int mid=(e+hi)/2;if(absr>=bounds[mid])e=mid;else hi=mid;}
        sg=tails[4*e];su=tails[4*e+1];}
    sg=__shfl_sync(0xffffffffu,sg,0);su=__shfl_sync(0xffffffffu,su,0);
    // One bit-preserving receipt store per row, after the same expert lookup.
    if(row_sd && blockIdx.y==0 && threadIdx.x==0)row_sd[row]=tails[4*e+2];
    const float* src=dm+row*ld;
    const float4 xi=i0<ne00?make_float4(h_value_r56(src,i0,sg,su),h_value_r56(src,i0+1,sg,su),
        h_value_r56(src,i0+2,sg,su),h_value_r56(src,i0+3,sg,su)):make_float4(0.f,0.f,0.f,0.f);
    if(diagnostic_h && i0<ne00)((float4*)(diagnostic_h+row*640))[i0/4]=xi;
    float amax=fabsf(xi.x);amax=fmaxf(amax,fabsf(xi.y));amax=fmaxf(amax,fabsf(xi.z));amax=fmaxf(amax,fabsf(xi.w));
#pragma unroll
    for(int offset=vals_per_scale/8;offset>0;offset>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,offset,WARP_SIZE));
    const float d_inv=127.0f/amax;
    char4 q;q.x=roundf(xi.x*d_inv);q.y=roundf(xi.y*d_inv);q.z=roundf(xi.z*d_inv);q.w=roundf(xi.w*d_inv);
    const float d=1.0f/d_inv;
    auto* y=(block_q8_1_mmq*)vy;
    const int64_t k_block=i0/QK8_1_MMQ,iqs=i0%QK8_1_MMQ;
    const int64_t ib=k_block*nr+row;
    ((char4*)y[ib].qs)[iqs/4]=q;
    if(iqs%32==0)y[ib].d4[iqs/32]=d;
}
}
void r56_v84_swiglu_d4(const float* dm,int64_t ld,int64_t nr,const int32_t* bounds,
        int ng,const float* tails,int64_t row0,void* hq,void* stream,float* diagnostic_h,float* row_sd) {
    if(nr==0)return;
    static thread_local bool announced=false;if(!announced){int d=-1;cudaGetDevice(&d);std::fprintf(stderr,"R56 V85 actual CUDA%d: fused scaledSwiGLU/D4, ld%lld, row_sd%d, no intermediate H\n",d,(long long)ld,row_sd?1:0);announced=true;}
    if((ld!=1280&&ld!=2560)||nr<0||ng<=0||ng>16||!dm||!bounds||!tails||!hq)throw std::runtime_error("R56 V70 requires private GU ld1280/2560 geometry");
    // Exact frozen quantizer:128 threads,1024 padded cols ->2 column tiles.
    swiglu_quantize_d4_r56<<<dim3((unsigned)nr,2),CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,(cudaStream_t)stream>>>(dm,ld,nr,bounds,ng,tails,row0,hq,diagnostic_h,row_sd);
    const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
}
