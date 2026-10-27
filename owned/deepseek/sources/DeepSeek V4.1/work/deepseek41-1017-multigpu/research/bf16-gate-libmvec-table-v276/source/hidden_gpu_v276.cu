#include "hidden_gpu_v276.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
__device__ __forceinline__ uint16_t rb(float x){const uint32_t u=__float_as_uint(x);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
__device__ __forceinline__ float bf(uint16_t x){return __uint_as_float(uint32_t(x)<<16);}
__device__ __forceinline__ float absolute(float x){return __uint_as_float(__float_as_uint(x)&0x7fffffff);}
__device__ __forceinline__ float max32(float v){for(int step=16;step;step>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,step));return v;}
__device__ __forceinline__ float fp8value(int code){if(code<8)return __fmul_rn(float(code),0.001953125f);return __uint_as_float(uint32_t((code>>3)+120)<<23|uint32_t(code&7)<<20);}
__global__ void hidden(const uint32_t*expbits,const uint8_t*domain,const float*gate,const float*up,const float*alpha,
 uint16_t*hb,float*hf,int8_t*a8,float*scales,int32_t*sums,int*error){
 const int lane=threadIdx.x,block=blockIdx.x,index=block*32+lane,route=block/72;
 const float rawg=gate[index],rawu=up[index],a=alpha[route];
 const bool bad=!isfinite(rawg)||!isfinite(rawu)||!isfinite(a)||a<0;
 if(__any_sync(0xffffffff,bad)){if(!lane)atomicCAS(error,0,4);return;}
 float g=bf(rb(rawg)),u=bf(rb(rawu));g=fminf(g,10.f);u=fmaxf(-10.f,fminf(u,10.f));
 const uint16_t code=rb(g);const bool invalid=!isfinite(g)||!isfinite(u)||domain[code]!=1;
 if(__any_sync(0xffffffff,invalid)){if(!lane)atomicCAS(error,0,4);return;}
 const float ex=__uint_as_float(expbits[code]);
 const float h=bf(rb(__fmul_rn(__fmul_rn(__fdiv_rn(g,__fadd_rn(1.f,ex)),u),a)));
 if(__any_sync(0xffffffff,!isfinite(h))){if(!lane)atomicCAS(error,0,4);return;}
 // Literal Task18 post: hiddenBF16 -> per32 FP8/E8M0 -> original A8.
 const float maximum=fmaxf(1e-4f,max32(absolute(h)));
 const float raw=__fmul_rn(maximum,1.f/448.f);const uint32_t ur=__float_as_uint(raw);
 const int exponent=int((ur>>23)&255)-127+((ur&0x7fffff)!=0);
 const float fscale=__uint_as_float(uint32_t(exponent+127)<<23);
 const float normalized=__fdiv_rn(h,fscale),mag=fminf(absolute(normalized),448.f);const uint32_t m=__float_as_uint(mag);
 int q=mag<0.015625f?__float2int_rn(__fmul_rn(mag,512.f)):int((m+0x7ffff+((m>>20)&1))>>20)-960;q=min(q,126);
 const float signed_value=__uint_as_float(__float_as_uint(fp8value(q))|(__float_as_uint(normalized)&0x80000000u));
 const float v=__fmul_rn(signed_value,fscale),mx=max32(absolute(v)),scale=__fdiv_rn(mx,127.f);
 if(__any_sync(0xffffffff,!isfinite(v)||!isfinite(scale)||(mx>0&&scale==0))){if(!lane)atomicCAS(error,0,4);return;}
 const float divisor=scale>0?scale:1.f;int quant=__float2int_rn(__fdiv_rn(v,divisor));quant=max(-127,min(127,quant));
 __shared__ int code4[32];code4[lane]=quant;__syncthreads();
 hb[index]=rb(h);hf[index]=v;a8[index]=int8_t(quant);if(!lane)scales[block]=scale;
 if(!(lane&3))sums[block*8+lane/4]=code4[lane]+code4[lane+1]+code4[lane+2]+code4[lane+3];
}
}
void ds276_launch_hidden(const uint32_t*expbits,const uint8_t*domain,const float*g,const float*u,const float*a,int routes,
 uint16_t*hb,float*hf,int8_t*codes,float*scales,int32_t*sums,int*error,cudaStream_t stream){
 if(!expbits||!domain||!g||!u||!a||!hb||!hf||!codes||!scales||!sums||!error||routes<1||routes>36)throw std::invalid_argument("276 hidden shape/pointers");
 // Pure integer address/range checks are capture-safe; never query CUDA here.
 const size_t n=size_t(routes)*2304;
 const void*inputs[5]={expbits,domain,g,u,a};const size_t inbytes[5]={65536*4,65536,n*4,n*4,size_t(routes)*4};
 const void*outputs[6]={hb,hf,codes,scales,sums,error};const size_t outbytes[6]={n*2,n*4,n,n/32*4,n/4*4,4};
 auto overlaps=[](const void*x,size_t nx,const void*y,size_t ny){const uintptr_t a=reinterpret_cast<uintptr_t>(x),b=reinterpret_cast<uintptr_t>(y);return nx>UINTPTR_MAX-a||ny>UINTPTR_MAX-b||(a<b+ny&&b<a+nx);};
 for(int i=0;i<6;i++){for(int j=0;j<5;j++)if(overlaps(outputs[i],outbytes[i],inputs[j],inbytes[j]))throw std::invalid_argument("276 readonly/output alias");for(int j=0;j<i;j++)if(overlaps(outputs[i],outbytes[i],outputs[j],outbytes[j]))throw std::invalid_argument("276 output alias");}
 if((reinterpret_cast<uintptr_t>(expbits)|reinterpret_cast<uintptr_t>(g)|reinterpret_cast<uintptr_t>(u)|reinterpret_cast<uintptr_t>(a)|reinterpret_cast<uintptr_t>(hf)|reinterpret_cast<uintptr_t>(scales)|reinterpret_cast<uintptr_t>(sums)|reinterpret_cast<uintptr_t>(error))%4||reinterpret_cast<uintptr_t>(hb)%2)throw std::invalid_argument("276 typed pointer alignment");
 hidden<<<routes*72,32,0,stream>>>(expbits,domain,g,u,a,hb,hf,codes,scales,sums,error);
}
