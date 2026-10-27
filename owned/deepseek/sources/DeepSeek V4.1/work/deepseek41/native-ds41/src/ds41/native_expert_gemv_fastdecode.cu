#include "strata/ds41/native_expert_gemv.hpp"
#include "strata/ds41/native_act_storage.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
namespace strata::ds41 {
namespace {
__device__ float fp4(int c){uint32_t m=c&7;uint32_t raw=m<2?(m?0x3f000000u:0u):((m/2+126)<<23)|((m&1)<<22);return __uint_as_float(raw|(uint32_t(c&8)<<28));}
__device__ float e8(uint8_t s){return __int_as_float(s?int(s)<<23:0x00400000);}
__device__ float weight_value(NativeDeviceProjection p,int row,int col,int columns){
 if(!p.bits){const auto*w=p.weights+size_t(row)*(columns/2);int code=(col&1)?w[col/2]>>4:w[col/2]&15;return fp4(code)*e8(p.scales[size_t(row)*(columns/32)+col/32]);}
 const int code_bytes=columns*p.bits/8;const auto*w=p.weights+size_t(row)*(code_bytes+columns/32*2);int bit=col*p.bits,byte=bit/8;int code=((int(w[byte])|(int(w[byte+1])<<8))>>(bit%8))&((1<<p.bits)-1);if(code==((1<<p.bits)-1))return NAN;
 int scale=int(w[code_bytes+col/32*2])|(int(w[code_bytes+col/32*2+1])<<8);float value=float(code-((1<<(p.bits-1))-1))*__int_as_float(scale<<16);return __bfloat162float(__float2bfloat16_rn(value));
}
__global__ void gemv(const uint16_t*x,NativeDeviceProjection first,NativeDeviceProjection second,int rows,int cols,uint16_t*out){
 const int lane=threadIdx.x&31,index=blockIdx.x*4+(threadIdx.x>>5);if(index>=rows*(second.weights?2:1))return;const int row=index%rows;const auto p=index<rows?first:second;float sum=0;
 for(int j=lane;j<cols;j+=32)sum=__fmaf_rn(__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[j]),weight_value(p,row,j,cols),sum);
 for(int offset=16;offset;offset>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,offset));if(!lane)reinterpret_cast<__nv_bfloat16*>(out)[index]=__float2bfloat16_rn(sum);
}
__global__ void activation(const uint16_t*pair,const float*alpha,float limit,uint16_t*h){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2304)return;const auto*p=reinterpret_cast<const __nv_bfloat16*>(pair);float g=__bfloat162float(p[i]),u=__bfloat162float(p[2304+i]);if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}float v=(g/(1.f+expf(-g)))*u*alpha[0];reinterpret_cast<__nv_bfloat16*>(h)[i]=__float2bfloat16_rn(v);}
__global__ void one_index(int32_t*index){*index=0;}
__global__ void many_indices(int32_t*indices,int count){int i=threadIdx.x;if(i<count)indices[i]=i;}
__global__ void many_gemv(const uint16_t*x,const NativeDeviceExpert*experts,int count,int rows,int cols,bool gate_up,uint16_t*out){
 int expert=blockIdx.y,lane=threadIdx.x&31,index=blockIdx.x*4+(threadIdx.x>>5);
 if(expert>=count||index>=rows*(gate_up?2:1))return;
 const auto p=experts[expert].projection[gate_up?(index<rows?0:1):2];int row=index%rows;
 const auto*input=reinterpret_cast<const __nv_bfloat16*>(x)+(gate_up?0:size_t(expert)*cols);float sum=0.f;
 for(int col=lane;col<cols;col+=32)sum=__fmaf_rn(__bfloat162float(input[col]),weight_value(p,row,col,cols),sum);
 for(int offset=16;offset;offset>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,offset));
 if(!lane)reinterpret_cast<__nv_bfloat16*>(out)[size_t(expert)*rows*(gate_up?2:1)+index]=__float2bfloat16_rn(sum);
}
__global__ void many_activation(const uint16_t*pair,const float*alpha,int count,float limit,uint16_t*h){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count*2304)return;
 int expert=index/2304,col=index%2304;const auto*p=reinterpret_cast<const __nv_bfloat16*>(pair)+expert*4608;
 float g=__bfloat162float(p[col]),u=__bfloat162float(p[2304+col]);
 if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
 float value=(g/(1.f+expf(-g)))*u*alpha[expert];reinterpret_cast<__nv_bfloat16*>(h)[index]=__float2bfloat16_rn(value);
}
}
void native_expert_cached_many(const uint16_t*x,const NativeDeviceExpert*experts,const float*alpha,int count,float limit,uint16_t*out,void*workspace,size_t bytes,cudaStream_t stream){
 if(!x||!experts||!alpha||!out||!workspace||count<1||count>6||bytes<256*1024)throw std::invalid_argument("invalid native cached expert batch");
 auto*base=static_cast<uint8_t*>(workspace);auto*codes=base;auto*scales=base+5120;auto*indices=reinterpret_cast<int32_t*>(base+5376);auto*q=reinterpret_cast<uint16_t*>(base+5632);
 auto*device_experts=reinterpret_cast<NativeDeviceExpert*>(base+16384);auto*pair=reinterpret_cast<uint16_t*>(base+17408);auto*h=reinterpret_cast<uint16_t*>(base+73728);
 auto*hc=base+102400;auto*hs=base+116224;auto*qh=reinterpret_cast<uint16_t*>(base+117760);auto*a=reinterpret_cast<float*>(base+146432);
 static_assert(sizeof(NativeDeviceExpert)*6<=1024);
 auto check=[](cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));};
 check(cudaMemcpyAsync(device_experts,experts,count*sizeof(NativeDeviceExpert),cudaMemcpyHostToDevice,stream));
 check(cudaMemcpyAsync(a,alpha,count*4,cudaMemcpyHostToDevice,stream));
 many_indices<<<1,32,0,stream>>>(indices,count);native_act_storage_pack(x,1,5120,codes,scales,stream);native_act_storage_gather(codes,scales,indices,1,1,5120,q,stream);
 many_gemv<<<dim3((4608+3)/4,count),128,0,stream>>>(q,device_experts,count,2304,5120,true,pair);
 many_activation<<<(count*2304+255)/256,256,0,stream>>>(pair,a,count,limit,h);
 native_act_storage_pack(h,count,2304,hc,hs,stream);native_act_storage_gather(hc,hs,indices,count,count,2304,qh,stream);
 many_gemv<<<dim3((5120+3)/4,count),128,0,stream>>>(qh,device_experts,count,5120,2304,false,out);check(cudaGetLastError());
}
void native_expert_cached_gemv(const uint16_t*x,const float*alpha,float limit,const NativeDeviceProjection&gate,const NativeDeviceProjection&up,const NativeDeviceProjection&down,uint16_t*out,void*workspace,size_t bytes,cudaStream_t stream){

 if(!x||!alpha||!out||!workspace||bytes<65536||!gate.weights||(!gate.bits&&!gate.scales)||!up.weights||(!up.bits&&!up.scales)||!down.weights||(!down.bits&&!down.scales))throw std::invalid_argument("invalid native cached GEMV buffers");
 auto*base=static_cast<uint8_t*>(workspace);auto*codes=base;auto*scales=base+5120;auto*index=reinterpret_cast<int32_t*>(base+5376);auto*q=reinterpret_cast<uint16_t*>(base+5632);auto*pair=reinterpret_cast<uint16_t*>(base+15872);auto*h=pair+4608;auto*hc=reinterpret_cast<uint8_t*>(h+2304);auto*hs=hc+2304;auto*qh=reinterpret_cast<uint16_t*>(base+32768);
 one_index<<<1,1,0,stream>>>(index);native_act_storage_pack(x,1,5120,codes,scales,stream);native_act_storage_gather(codes,scales,index,1,1,5120,q,stream);
 gemv<<<(4608+3)/4,128,0,stream>>>(q,gate,up,2304,5120,pair);
 activation<<<9,256,0,stream>>>(pair,alpha,limit,h);native_act_storage_pack(h,1,2304,hc,hs,stream);native_act_storage_gather(hc,hs,index,1,1,2304,qh,stream);
 gemv<<<(5120+3)/4,128,0,stream>>>(qh,down,NativeDeviceProjection{},5120,2304,out);
 auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
}
