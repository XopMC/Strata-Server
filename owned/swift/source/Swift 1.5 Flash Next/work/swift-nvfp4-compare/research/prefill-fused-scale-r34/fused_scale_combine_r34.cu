// NV4-only candidate: exact scale-then-FMA, overlay metadata in DEAD Hq prefix.
// Requires whole-T*K Hq allocation, per-group output offsets and all-MMQ R20 guard.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace strata::prefill::mmq {
namespace {
constexpr int N=2560,K=10;
void check_r34(cudaError_t e,const char*what){if(e!=cudaSuccess){std::fprintf(stderr,"R34 %s: %s\n",what,cudaGetErrorString(e));std::abort();}}
__global__ void save_scales_r34_kernel(uint8_t* hq,int64_t nr,const int32_t* bounds,int ng,const float* tails,int64_t row0){
 const int64_t r=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(r>=nr)return;
 const int64_t absr=row0+r;int e=0,hi=ng;
 while(e+1<hi){const int mid=(e+hi)/2;if(absr>=bounds[mid])e=mid;else hi=mid;}
 // AFTER group down-MMQ, previous/current Hq data are dead. Future group quantization
 // begins at next_global_row*1152, strictly above metadata [0,next_global_row*4).
 reinterpret_cast<float*>(hq)[absr]=tails[4*e+2];
}
__global__ void combine_r34_kernel(const float* __restrict__ Dm,const uint8_t* __restrict__ hq,const int32_t* __restrict__ slot,const float* __restrict__ w,const float* __restrict__ shared,const float* __restrict__ sg,float* __restrict__ bo,int64_t T){
 const int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=T*N)return;
 const int64_t t=i/N,d=i%N;float s=0.f;
#pragma unroll
 for(int k=0;k<K;++k){
  const int64_t row=slot[t*K+k];const float scale=reinterpret_cast<const float*>(hq)[row];
  const float scaled=__fmul_rn(Dm[row*N+d],scale);
  s=fmaf(w[t*K+k],scaled,s);
 }
 const float sigmoid=__fdiv_rn(1.0f,1.0f+__expf(-sg[t])); // Match frozen precise shared-gate divide despite MMQ -use_fast_math.
 bo[i]=s+shared[i]*sigmoid;
}
}
void swift_nv4_save_down_r34(void* hq,int64_t nr,const int32_t* bounds,int ng,const float* tails,int64_t row0,void* stream){
 if(nr<=0)return;save_scales_r34_kernel<<<(unsigned)((nr+255)/256),256,0,(cudaStream_t)stream>>>((uint8_t*)hq,nr,bounds,ng,tails,row0);check_r34(cudaGetLastError(),"save scales");
}
void swift_nv4_combine_r34(const float* Dm,const void* hq,const int32_t* slot,const float* w,const float* shared,const float* sg,float* bo,int64_t T,void* stream){
 if(T<=0)return;combine_r34_kernel<<<(unsigned)((T*N+255)/256),256,0,(cudaStream_t)stream>>>(Dm,(const uint8_t*)hq,slot,w,shared,sg,bo,T);check_r34(cudaGetLastError(),"scale+combine");
}
}
