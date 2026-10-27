// Query-register specialization. Original FP32 operations/order; no tensorcores or rounding changes.
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
namespace qsa_peer_v441 {
namespace {
template<int NT> __global__ void __launch_bounds__(256) score_register112(const float* __restrict__ pooled,const float* __restrict__ dead,const float* __restrict__ query,const int32_t* __restrict__ steps,int nq,int64_t max_blocks,float* __restrict__ out,int64_t lo,int64_t hi){
 const int lane=threadIdx.x&31;float4 q[NT][4];int64_t nkv[NT],bid[NT];int64_t top=0;
 #pragma unroll
 for(int t=0;t<NT;++t){
  nkv[t]=steps[t*4+1];bid[t]=steps[t*4+2];top=bid[t]>top?bid[t]:top;
  #pragma unroll
  for(int h=0;h<4;++h)q[t][h]=*reinterpret_cast<const float4*>(query+t*512+h*128+lane*4);
 }
 const float4 kd=*reinterpret_cast<const float4*>(dead+lane*4);
 for(int64_t b=lo+int64_t(blockIdx.x)*8+(threadIdx.x>>5);b<=top&&b<max_blocks&&b<hi;b+=int64_t(gridDim.x)*8){
  const float4 kp=*reinterpret_cast<const float4*>(pooled+b*128+lane*4);
  #pragma unroll
  for(int t=0;t<NT;++t){
   if(b>bid[t])continue;const float4 k=b==bid[t]?kd:kp;float score=0;
   #pragma unroll
   for(int h=0;h<4;++h){const float4 v=q[t][h];float d=k.x*v.x+k.y*v.y+k.z*v.z+k.w*v.w;
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1)d+=__shfl_xor_sync(0xffffffffu,d,offset);
    score+=d>0?d:0;
   }
   if(lane==0){if(b==bid[t]&&nkv[t]%4!=0)score+=1e9f;out[t*max_blocks+b]=score;}
  }
 }
}
}
cudaError_t score(int nq,const float*pooled,const float*dead,const float*query,const int32_t*steps,int64_t max,float*out,int64_t lo,int64_t hi,cudaStream_t cs){
 if(nq<1||nq>8||max<1||max>262146||lo<0||hi<lo||hi>max)return cudaErrorInvalidValue;
 const int grid=256;if(hi==lo)return cudaSuccess;
 #define CASE(N) case N:score_register112<N><<<grid,256,0,cs>>>(pooled,dead,query,steps,nq,max,out,lo,hi);break;
 switch(nq){CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)}
 #undef CASE
 return cudaGetLastError();
}
}
