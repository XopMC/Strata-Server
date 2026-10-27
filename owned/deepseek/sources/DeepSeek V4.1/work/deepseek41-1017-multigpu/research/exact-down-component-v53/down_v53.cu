#include "down_v53.hpp"
#include <cstdlib>
#include <cstring>
namespace ds1017_exact_down_v53 {
namespace {
constexpr int TileBytes=288, Chunk=32;
__device__ __forceinline__ uint32_t packed4(const uint8_t*block,int row,int segment){
 const uint16_t nib=*reinterpret_cast<const uint16_t*>(block+segment*32+row*2);
 const uint32_t select=uint32_t(nib)&0x7777u;
 uint32_t positive,negative,out;
 asm("prmt.b32 %0,%1,%2,%3;":"=r"(positive):"r"(0x0f0e0d0cu),"r"(0x18141210u),"r"(select));
 asm("prmt.b32 %0,%1,%2,%3;":"=r"(negative):"r"(0x090a0b0cu),"r"(0x00040608u),"r"(select));
 const uint32_t signs=0x3210u|((uint32_t(nib)&0x8888u)>>1);
 asm("prmt.b32 %0,%1,%2,%3;":"=r"(out):"r"(positive),"r"(negative),"r"(signs));
 return out;
}
__device__ __forceinline__ float factor(const uint8_t*block,int row){
 unsigned code=*reinterpret_cast<const uint16_t*>(block+256+row*2);
 return __fmul_rn(__int_as_float(code?code<<23:0x00400000u),0.5f);
}

__device__ __forceinline__ int bias(const int32_t*s,int t,int g){int v=0;
 #pragma unroll
 for(int j=0;j<8;j++)v+=s[t*(K/4)+g*8+j];return 12*v;
}


// Integer dot products remain independent for each K32. Shared memory only
// replaces the global integer scratch; DOWN accumulates ascending g=0..71.
// The final chunk has eight groups; producers and consumers both guard that tail.
__global__ void shared_tile(const uint8_t*const*weights,const Request*request,float*out){
 const Entry d=request->entry[blockIdx.y];
 const uint8_t*w=weights[d.weight];
 const int count=d.count,rowtile=blockIdx.x;
 const int lane=threadIdx.x&31,group=lane>>2,sub=lane&3,warp=threadIdx.x>>5;
 __shared__ int32_t dots[Chunk*16*MaxCount];
 const int ot=threadIdx.x/16,orow=threadIdx.x%16;
 float acc=0.f;
 for(int first=0;first<Groups;first+=Chunk){
  for(int local=warp;local<Chunk;local+=4){
   const int g=first+local;if(g>=Groups)continue; // warp-uniform: g is identical in all32 lanes
   const uint8_t*b=w+(size_t(rowtile)*Groups+g)*TileBytes;
   uint32_t a0=packed4(b,group,sub),a1=packed4(b,group+8,sub),a2=packed4(b,group,sub+4),a3=packed4(b,group+8,sub+4);
   uint32_t b0=0,b1=0;
   if(group<count){int q=d.view[group];b0=*reinterpret_cast<const uint32_t*>(request->codes+q*K+g*32+sub*4);b1=*reinterpret_cast<const uint32_t*>(request->codes+q*K+g*32+sub*4+16);}
   int d0,d1,d2,d3,z=0;
   asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
    :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(z));
   int dot[4]={d0,d1,d2,d3};
   #pragma unroll
   for(int j=0;j<4;j++){
    const int r=group+8*(j/2),t=2*sub+(j&1);
    if(t<count)dots[(t*Chunk+local)*16+r]=dot[j]-bias(request->sums,d.view[t],g);
   }
  }
  __syncthreads();
  if(ot<count){
   for(int local=0;local<Chunk&&first+local<Groups;local++){
    const int g=first+local;
    const uint8_t*b=w+(size_t(rowtile)*Groups+g)*TileBytes;
    acc=__fmaf_rn(__int2float_rn(dots[(ot*Chunk+local)*16+orow]),__fmul_rn(factor(b,orow),request->scales[d.view[ot]*Groups+g]),acc);
   }
  }
  __syncthreads();
 }
 if(ot<count)out[d.output_words+ot*Rows+rowtile*16+orow]=acc;
}

}
cudaError_t launch(const uint8_t*const*weights,const Request*request,int experts,uint32_t output_words,float*out,cudaStream_t stream){
 if(!weights||!request||!out||experts<1||experts>MaxExperts||!output_words||output_words>MaxOutputWords||output_words%32)return cudaErrorInvalidValue;
 shared_tile<<<dim3(Rows/16,experts),128,0,stream>>>(weights,request,out);
 return cudaPeekAtLastError();
}
}
