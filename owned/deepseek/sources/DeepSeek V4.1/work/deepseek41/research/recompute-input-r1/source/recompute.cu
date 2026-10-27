#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace {
using bf16=__nv_bfloat16;
constexpr int D=5120,H=4;
// Same 1024-thread order and separate BF16 rounding as original ops::rmsnorm.
__global__ void norm_save(const bf16*x,const bf16*w,bf16*y,float*inverse,int rows){
    __shared__ float sh[32];
    const int r=blockIdx.x; x+=size_t(r)*D;y+=size_t(r)*D;
    float ss=0.f;
    for(int i=threadIdx.x;i<D;i+=1024){const float v=__bfloat162float(x[i]);ss+=v*v;}
    for(int off=16;off>0;off>>=1)ss+=__shfl_down_sync(0xffffffffu,ss,off);
    if((threadIdx.x&31)==0)sh[threadIdx.x>>5]=ss;
    __syncthreads();
    if(threadIdx.x==0){float sum=0.f;for(int i=0;i<32;i++)sum+=sh[i];sh[0]=sum;}
    __syncthreads();ss=sh[0];__syncthreads();
    const float inv=rsqrtf(__fadd_rn(__fdiv_rn(ss,float(D)),1e-20f));
    if(threadIdx.x==0)inverse[r]=inv;
    for(int i=threadIdx.x;i<D;i+=1024)y[i]=__float2bfloat16_rn(__bfloat162float(w[i])*(__bfloat162float(x[i])*inv));
}
// A warp reconstructs exactly one existing activation-scale group. No new quant.
__global__ void gather_recompute(const bf16*hc,const float*pre,const float*inv,const bf16*w,const int32_t*ids,int rows,int capacity,bf16*out){
    const size_t at=size_t(blockIdx.x)*256+threadIdx.x;if(at>=size_t(rows)*D)return;
    const int row=ids[at/D];assert(row>=0&&row<capacity);const int d=at%D;
    float sum=0.f;
#pragma unroll
    for(int j=0;j<H;j++)sum+=pre[size_t(row)*H+j]*__bfloat162float(hc[(size_t(row)*H+j)*D+d]);
    const float collapsed=__bfloat162float(__float2bfloat16_rn(sum));
    const float normalized=__bfloat162float(__float2bfloat16_rn(__bfloat162float(w[d])*(collapsed*inv[row])));
    float maximum=fabsf(normalized);
    for(int shift=16;shift>0;shift>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,shift));
    int exponent;const float mantissa=frexpf(fmaxf(maximum,1e-4f)*(1.f/448.f),&exponent);
    const int e=mantissa==.5f?exponent-1:exponent;const float scale=ldexpf(1.f,e);
    const __nv_fp8_e4m3 q(fminf(fmaxf(normalized/scale,-448.f),448.f));
    out[at]=__float2bfloat16_rn(float(q)*scale);
}
void check(){const auto rc=cudaGetLastError();if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
}
extern "C" void ds41_recompute_norm_r1(const uint16_t*x,const uint16_t*w,uint16_t*y,float*inv,int rows){
    if(!x||!w||!y||!inv||rows<1||rows>4096)throw std::invalid_argument("recompute RMS rows");
    norm_save<<<rows,1024>>>(reinterpret_cast<const bf16*>(x),reinterpret_cast<const bf16*>(w),reinterpret_cast<bf16*>(y),inv,rows);check();
}
extern "C" void ds41_recompute_gather_r1(const uint16_t*hc,const float*pre,const float*inv,const uint16_t*w,const int32_t*ids,int rows,int capacity,uint16_t*out){
    if(!hc||!pre||!inv||!w||!ids||!out||rows<1||rows>16384||capacity<1)throw std::invalid_argument("recompute gather rows");
    gather_recompute<<<(size_t(rows)*D+255)/256,256>>>(reinterpret_cast<const bf16*>(hc),pre,inv,reinterpret_cast<const bf16*>(w),ids,rows,capacity,reinterpret_cast<bf16*>(out));check();
}
