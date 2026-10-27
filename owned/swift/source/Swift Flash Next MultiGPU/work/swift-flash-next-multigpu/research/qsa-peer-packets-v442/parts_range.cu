// Integer register-radix machinery adapted from Strata's MIT-licensed
// qsa_select.cu. Only local candidate selection is new; final weighted selection
// is performed by the original frozen device kernel in the host caller.
#include "selector16.hpp"
#include <cuda_runtime.h>
namespace qsa_peer_v441 {
using swift_selector_v7_leaf::Workspace;using swift_selector_v7_leaf::Parts;using swift_selector_v7_leaf::Keep;
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
__global__ void __launch_bounds__(T) local_candidates(const float* scores,const int32_t* steps,int64_t max_blocks,Workspace w,int first_part){
 __shared__ int hist[WARPS][256],s_warp[WARPS+1],s_digit,s_above,s_done;
 const int q=blockIdx.x,part=first_part+blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
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
}
cudaError_t chunks(const float*s,const int32_t*st,int nq,int64_t max,Workspace w,int first_part,int parts,cudaStream_t cs){
 if(nq<1||nq>8||max<1||max>262146||first_part<0||parts<1||first_part+parts>16)return cudaErrorInvalidValue;
 if(max<=131074)local_candidates<false,17><<<dim3(nq,parts),T,0,cs>>>(s,st,max,w,first_part);
 else local_candidates<false,33><<<dim3(nq,parts),T,0,cs>>>(s,st,max,w,first_part);
 return cudaGetLastError();
}
constexpr int PacketWords=8+2*8*Keep;
__global__ void copy_inputs(const float*sq,const float*sd,const int32_t*ss,const float*sr,float*dq,float*dd,int32_t*ds,float*dr,int q){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<q*512;i+=blockDim.x*gridDim.x)dq[i]=sq[i];
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<128;i+=blockDim.x*gridDim.x){dd[i]=sd[i];dr[i]=sr[i];}
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<q*4;i+=blockDim.x*gridDim.x)ds[i]=ss[i];
}
__global__ void gather_bits(const float*s,int64_t stride,Workspace w,uint32_t*out){int q=blockIdx.x;auto*packet=out+q*PacketWords;
 if(threadIdx.x<8)packet[threadIdx.x]=uint32_t(w.counts[q*Parts+threadIdx.x]);
 for(int j=threadIdx.x;j<8*Keep;j+=blockDim.x){int p=j/Keep,k=j%Keep;int count=w.counts[q*Parts+p];int id=w.local_ids[(q*Parts+p)*Keep+k];packet[8+j]=uint32_t(id);packet[8+8*Keep+j]=k<count?__float_as_uint(s[int64_t(q)*stride+id]):0xdeadbeefu;}}
__global__ void publish_bits(float*s,int64_t stride,Workspace w,const uint32_t*in){int q=blockIdx.x;const auto*packet=in+q*PacketWords;
 if(threadIdx.x<8)w.counts[q*Parts+threadIdx.x]=int32_t(packet[threadIdx.x]);
 for(int j=threadIdx.x;j<8*Keep;j+=blockDim.x){int p=j/Keep,k=j%Keep,id=int32_t(packet[8+j]);w.local_ids[(q*Parts+p)*Keep+k]=id;if(k<int(packet[p]))reinterpret_cast<uint32_t*>(s)[int64_t(q)*stride+id]=packet[8+8*Keep+j];}}
cudaError_t inputs(const float*sq,const float*sd,const int32_t*ss,const float*sr,float*dq,float*dd,int32_t*ds,float*dr,int q,cudaStream_t cs){copy_inputs<<<4,256,0,cs>>>(sq,sd,ss,sr,dq,dd,ds,dr,q);return cudaGetLastError();}
cudaError_t gather(const float*s,int64_t stride,Workspace w,uint32_t*out,int q,cudaStream_t cs){gather_bits<<<q,256,0,cs>>>(s,stride,w,out);return cudaGetLastError();}
cudaError_t publish(float*s,int64_t stride,Workspace w,const uint32_t*in,int q,cudaStream_t cs){publish_bits<<<q,256,0,cs>>>(s,stride,w,in);return cudaGetLastError();}
}
