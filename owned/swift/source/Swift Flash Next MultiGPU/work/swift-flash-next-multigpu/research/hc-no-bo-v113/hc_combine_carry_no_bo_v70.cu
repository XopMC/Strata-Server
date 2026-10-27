// SOURCE ONLY; not built/run. Arithmetic transcribed from pinned prefill/kernels.cu:
// moe_combine_kernel + gr_write_norm_rs_kernel. Retain original compiler FP flags.
#include "hc_combine_carry_no_bo_v70.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace swift_hc_carry_no_bo_v70 {
namespace {
constexpr int N=2560,HC=4,D=N*HC,K=10,Threads=256,Per=N/Threads;
struct Span {uintptr_t p;uint64_t n;};
bool span(const void* p,uint64_t n,Span& out) {
    const auto z=reinterpret_cast<uintptr_t>(p);
    if(!p||!n||n>UINTPTR_MAX-z)return false;
    out={z,n};return true;
}
bool overlap(Span a,Span b){return a.p<b.p+b.n&&b.p<a.p+a.n;}
bool contains(Span a,Span b){return b.p>=a.p&&b.p+b.n<=a.p+a.n;}
__device__ __forceinline__ float sigmoid(float x){return 1.0f/(1.0f+__expf(-x));}
__device__ __forceinline__ uint16_t bf(float f){uint32_t u=__float_as_uint(f);u+=0x7fffu+((u>>16)&1u);return uint16_t(u>>16);}
__device__ float warp_sum(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffffu,v,o);
    return v;
}
// Same256-thread two-level tree as original block_sum, including its barriers.
__device__ float block_sum(float v,float* sh){
    const int lane=threadIdx.x&31,w=threadIdx.x>>5;
    v=warp_sum(v);__syncthreads();if(lane==0)sh[w]=v;__syncthreads();
    const int nw=(blockDim.x+31)>>5;
    float t=threadIdx.x<nw?sh[threadIdx.x]:0.0f;
    if(w==0)t=warp_sum(t);
    if(threadIdx.x==0)sh[0]=t;__syncthreads();return sh[0];
}
template<bool StoreBo>
__global__ void __launch_bounds__(Threads) combine_write_norm(Args a){
    __shared__ float sh[32];
    const int64_t t=blockIdx.x;
    float bo[Per];
    // Each output retains the original k order and FP32 bo rounding boundary.
    #pragma unroll
    for(int j=0;j<Per;++j){
        const int d=threadIdx.x+j*Threads;
        float s=0.0f;
        #pragma unroll
        for(int k=0;k<K;++k){
            const int64_t r=a.slot[t*K+k];
            const float v=a.row_sd?a.dm[r*N+d]*a.row_sd[r]:a.dm[r*N+d];
            s=fmaf(a.weight[t*K+k],v,s);
        }
        const float b=s+a.shared[t*N+d]*sigmoid(a.shared_gate[t]);
        bo[j]=b;if constexpr(StoreBo)a.bo[t*N+d]=b; // Only this dead store is optional.
    }
    // Serial HC rows reuse bo without rereading the ten-expert Dm four times.
    // Every row still maps d=threadIdx.x+256*j, exactly as original F2.
    for(int c=0;c<HC;++c){
        const int64_t row=t*HC+c;
        const float sc=2.0f*sigmoid(a.inject[t*HC+c]/float(HC));
        float v[Per],ss=0.0f;
        #pragma unroll
        for(int j=0;j<Per;++j){
            const int d=threadIdx.x+j*Threads;
            const float x=fmaf(bo[j],sc,a.residual[row*N+d]);
            a.residual[row*N+d]=x;v[j]=x;ss+=x*x;
        }
        const float rs=rsqrtf(block_sum(ss,sh)/float(N)+a.eps);
        if(threadIdx.x==0)a.rs[row]=rs;
        #pragma unroll
        for(int j=0;j<Per;++j){
            const int d=threadIdx.x+j*Threads;
            const float x=v[j]*rs*a.next_norm[c*N+d];
            a.xn16[t*D+c*N+d]=bf(x);
        }
        // block_sum's next entry already joins the CTA. This explicit end-of-row
        // join also prevents sh reuse while another warp still reads its scale.
        __syncthreads();
    }
}
}
cudaError_t validate(const Args& a){
    if(a.tokens<1||a.tokens>49152||!std::isfinite(a.eps)||a.eps<0)return cudaErrorInvalidValue;
    const uint64_t T=uint64_t(a.tokens);
    const void* p[]={a.dm,a.slot,a.weight,a.shared,a.shared_gate,a.inject,a.next_norm};
    const uint64_t n[]={T*K*N*4,T*K*4,T*K*4,T*N*4,T*4,T*HC*4,D*4};
    Span inputs[8]{},outputs[4]{},region{},xq{},hc{};
    for(int i=0;i<7;++i)if(!span(p[i],n[i],inputs[i])||(inputs[i].p&3u))return cudaErrorInvalidValue;
    int ni=7;if(a.row_sd){if(!span(a.row_sd,T*K*4,inputs[ni])||(inputs[ni].p&3u))return cudaErrorInvalidValue;++ni;}
    const void* op[]={a.bo,a.residual,a.rs,a.xn16};
    const uint64_t on[]={T*N*4,T*D*4,T*HC*4,T*D*2};
    for(int i=0;i<4;++i){
        if(!span(op[i],on[i],outputs[i])||(outputs[i].p&(i==3?1u:3u)))return cudaErrorInvalidValue;
        for(int j=0;j<ni;++j)if(overlap(outputs[i],inputs[j]))return cudaErrorInvalidValue;
        for(int j=0;j<i;++j)if(overlap(outputs[i],outputs[j]))return cudaErrorInvalidValue;
    }
    if(reinterpret_cast<const void*>(a.xn16)!=a.retired_xq||
       !span(a.region,a.region_bytes,region)||!span(a.retired_xq,a.xq_payload_bytes,xq)||
       !span(a.region,a.hc_view_bytes,hc)||!contains(region,hc)||overlap(hc,outputs[3])||
       !contains(region,xq)||!contains(xq,outputs[3]))return cudaErrorInvalidValue;
    return cudaSuccess;
}
cudaError_t launch(const Args& a,cudaStream_t stream){
    const cudaError_t e=validate(a);if(e!=cudaSuccess)return e;
    if(a.store_bo)combine_write_norm<true><<<unsigned(a.tokens),Threads,0,stream>>>(a);
    else combine_write_norm<false><<<unsigned(a.tokens),Threads,0,stream>>>(a);
    return cudaGetLastError();
}
}
