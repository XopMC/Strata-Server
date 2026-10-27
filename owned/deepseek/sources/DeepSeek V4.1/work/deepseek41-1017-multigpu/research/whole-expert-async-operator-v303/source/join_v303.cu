#include "plan_v303.hpp"
#include <cuda_runtime.h>
namespace ds303 {
namespace {
__global__ void gather(const Plan*p,const float*raw,float*gate,float*up){
 unsigned at=blockIdx.x*blockDim.x+threadIdx.x;
 if(at>=p->routes*Hidden)return;unsigned r=at/Hidden,c=at%Hidden;
 gate[at]=raw[p->gate[r]+c];up[at]=raw[p->up[r]+c];
}
__global__ void round_scatter(const Plan*p,const float*raw,uint16_t*out,int*error){
 unsigned at=blockIdx.x*blockDim.x+threadIdx.x;
 if(at>=p->routes*Dim||*error)return;unsigned r=at/Dim,c=at%Dim;
 float v=raw[p->down[r]+c];if(!isfinite(v)){atomicExch(error,30301);return;}
 uint32_t bits=__float_as_uint(v);uint16_t b=uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
 out[p->original_slot[r]*Dim+c]=b;
}
}
void gather303(const Plan*p,const float*r,float*g,float*u,cudaStream_t s){gather<<<(MaxRoutes*Hidden+255)/256,256,0,s>>>(p,r,g,u);}
void scatter_bf16303(const Plan*p,const float*r,uint16_t*b,int*e,cudaStream_t s){round_scatter<<<(MaxRoutes*Dim+255)/256,256,0,s>>>(p,r,b,e);}
}
