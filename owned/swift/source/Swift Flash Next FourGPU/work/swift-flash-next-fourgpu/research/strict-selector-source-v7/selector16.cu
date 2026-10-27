// Integer register-radix machinery adapted from Strata's MIT-licensed
// qsa_select.cu. Only local candidate selection is new; final weighted selection
// is performed by the original frozen device kernel in the host caller.
#include "selector16.hpp"
#include <cuda_runtime.h>
namespace swift_selector_v7_leaf {
namespace {
constexpr int T=512,WARPS=T/32;
__device__ __forceinline__ uint32_t order_key(float s){
 const float v=s+0.0f;if(!(v==v))return 0u;const uint32_t b=__float_as_uint(v);
 return (b&0x80000000u)?~b:(b|0x80000000u);
}
__device__ __forceinline__ int scan(int v,int* shared,int& total){
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;int x=v;
 #pragma unroll
 for(int o=1;o<32;o<<=1){int y=__shfl_up_sync(0xffffffffu,x,o);if(lane>=o)x+=y;}
 if(lane==31)shared[warp]=x;__syncthreads();
 if(warp==0){int w=lane<WARPS?shared[lane]:0,z=w;
  #pragma unroll
  for(int o=1;o<32;o<<=1){int y=__shfl_up_sync(0xffffffffu,z,o);if(lane>=o)z+=y;}
  if(lane<WARPS)shared[lane]=z-w;if(lane==31)shared[WARPS]=z;
 }
 __syncthreads();const int out=shared[warp]+x-v;total=shared[WARPS];__syncthreads();return out;
}
template<bool Aggregate,int PER>
__global__ void __launch_bounds__(T) local_candidates(const float* scores,const int32_t* steps,int64_t max_blocks,Workspace w){
 __shared__ int hist[WARPS][256],s_warp[WARPS+1],s_digit,s_above,s_done;
 const int q=blockIdx.x,part=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
 const int64_t stride=(max_blocks+Parts-1)/Parts,lo=part*stride;
 const int64_t blocks=(int64_t(steps[q*4+1])+3)/4;
 const int64_t nb=lo>=blocks?0:min(stride,blocks-lo);
 const int nout=int(min(int64_t(Keep),nb));
 int32_t* out=w.local_ids+(q*Parts+part)*Keep;
 if(t==0)w.counts[q*Parts+part]=nout;
 if(nb<=Keep){for(int j=t;j<nb;j+=T)out[j]=int32_t(lo+j);return;}
 const int64_t per=(nb+T-1)/T,b0=int64_t(t)*per,b1=min(b0+per,nb);
 const float* sc=scores+int64_t(q)*max_blocks+lo;
 uint32_t keys[PER];
 #pragma unroll
 for(int j=0;j<PER;++j)keys[j]=(b0+j<b1)?order_key(sc[b0+j]):0u;
 uint32_t prefix=0;int above=0;
 for(int shift=24;shift>=0;shift-=8){
  for(int bin=lane;bin<256;bin+=32)hist[warp][bin]=0;__syncwarp();
  const uint32_t mask=shift==24?0u:0xffffffffu<<(shift+8);
  #pragma unroll
  for(int j=0;j<PER;++j){
   const bool valid=b0+j<b1&&(keys[j]&mask)==(prefix&mask);
   const int bin=valid?int((keys[j]>>shift)&255):-1;
   if constexpr(Aggregate){const unsigned same=__match_any_sync(0xffffffffu,bin);if(bin>=0&&lane==__ffs(same)-1)atomicAdd(&hist[warp][bin],__popc(same));}
   else if(bin>=0)atomicAdd(&hist[warp][bin],1);
  }
  __syncthreads();
  if(t<256){int sum=0;for(int ww=0;ww<WARPS;++ww)sum+=hist[ww][t];hist[0][t]=sum;}
  __syncthreads();
  if(t==0){int cum=above,d=255;for(;d>0;--d){if(cum+hist[0][d]>=Keep)break;cum+=hist[0][d];}s_digit=d;s_above=cum;s_done=(cum+hist[0][d]==Keep);}
  __syncthreads();prefix|=uint32_t(s_digit)<<shift;above=s_above;__syncthreads();if(s_done)break;
 }
 int gt=0,eq=0;
 #pragma unroll
 for(int j=0;j<PER;++j)if(b0+j<b1){gt+=keys[j]>prefix;eq+=keys[j]==prefix;}
 int total;const int before=scan(eq,s_warp,total);int left=max(0,min(eq,Keep-above-before));
 int pos=scan(gt+left,s_warp,total);
 #pragma unroll
 for(int j=0;j<PER;++j)if(b0+j<b1){if(keys[j]>prefix)out[pos++]=int32_t(lo+b0+j);else if(keys[j]==prefix&&left>0){out[pos++]=int32_t(lo+b0+j);--left;}}
}
__global__ void compact(const float* scores,const int32_t* steps,int64_t max_blocks,Workspace w){
 __shared__ int offsets[Parts+1];const int q=blockIdx.x,t=threadIdx.x;
 if(t==0){offsets[0]=0;for(int p=0;p<Parts;++p)offsets[p+1]=offsets[p]+w.counts[q*Parts+p];
  const int count=offsets[Parts],nk=steps[q*4+1],tail=nk%4;
  int last=-1;for(int p=Parts-1;p>=0;--p)if(w.counts[q*Parts+p]){last=w.local_ids[(q*Parts+p)*Keep+w.counts[q*Parts+p]-1];break;}
  const bool partial=tail&&last==nk/4;const int packed_nk=4*count-(partial?4-tail:0);
  int32_t* st=w.packed_steps+q*4;st[0]=packed_nk-1;st[1]=packed_nk;st[2]=packed_nk/4;st[3]=min(steps[q*4+3],packed_nk);
  // Frozen final selector may read this zero-weight dummy tail key.
  w.packed_scores[q*Packed+count]=0.f;w.mapping[q*Packed+count]=-1;
 }
 __syncthreads();
 for(int i=t;i<offsets[Parts];i+=blockDim.x){int p=0;while(p+1<Parts&&i>=offsets[p+1])++p;
  const int id=w.local_ids[(q*Parts+p)*Keep+i-offsets[p]];w.mapping[q*Packed+i]=id;w.packed_scores[q*Packed+i]=scores[int64_t(q)*max_blocks+id];
 }
}
__global__ void unpack_ids(const int32_t* steps,int64_t cap,Workspace w,int32_t* out){
 const int q=blockIdx.x,t=threadIdx.x,width=steps[q*4+3],pnk=w.packed_steps[q*4+1];
 for(int j=t;j<width;j+=blockDim.x){const int id=w.packed_ids[int64_t(q)*cap+j];out[int64_t(q)*cap+j]=(id>=0&&id<pnk&&id/4<Packed)?4*w.mapping[q*Packed+id/4]+id%4:-123456789;}
}
}
cudaError_t chunks(const float* s,const int32_t* st,int nq,int64_t max,Workspace w,int aggregate,cudaStream_t cs){
 if(nq<1||nq>256||max<1||max>262146)return cudaErrorInvalidValue;
 if(max<=131074) {
  if(aggregate)local_candidates<true,17><<<dim3(nq,Parts),T,0,cs>>>(s,st,max,w);
  else local_candidates<false,17><<<dim3(nq,Parts),T,0,cs>>>(s,st,max,w);
 } else {
  if(aggregate)local_candidates<true,33><<<dim3(nq,Parts),T,0,cs>>>(s,st,max,w);
  else local_candidates<false,33><<<dim3(nq,Parts),T,0,cs>>>(s,st,max,w);
 }
 return cudaGetLastError();
}
cudaError_t pack(const float* s,const int32_t* st,int nq,int64_t max,Workspace w,cudaStream_t cs){compact<<<nq,256,0,cs>>>(s,st,max,w);return cudaGetLastError();}
cudaError_t translate(const int32_t* st,int nq,int64_t cap,Workspace w,int32_t* out,cudaStream_t cs){unpack_ids<<<nq,256,0,cs>>>(st,cap,w,out);return cudaGetLastError();}
}

// V7 cold proof metadata only; never called by select.
namespace swift_selector_v7_leaf {
void leaf_kernels(void**out){out[0]=reinterpret_cast<void*>(local_candidates<false,17>);out[1]=reinterpret_cast<void*>(local_candidates<false,33>);out[2]=reinterpret_cast<void*>(compact);out[3]=reinterpret_cast<void*>(unpack_ids);}
}
