// Base implementation + sealed V86 helpers are included unchanged once.
#include "../cpu-resident-grouped-chain-v6-r1/grouped_v6.cu"
#include "compact_v8.hpp"
namespace cpu_compact_v8 { namespace {
namespace G=cpu_grouped_v6;namespace L=swift_cpu_semantics_v86;namespace R=cpu_rows_v5;
__global__ void compact(G::Images i,WorkList*list,int bank){
 list->count=0;list->sequence=0;
 if(!G::valid(i,bank))return;
 unsigned n=0;for(int row=0;row<i.plan->T*10;++row)if(G::active(i,bank,row))list->rows[n++]=row;
 list->count=n;list->sequence=i.plan->sequence;
}
__device__ bool list_valid(G::Images i,const WorkList*list,int bank){
 if(!G::valid(i,bank))return false;
 if(list->count>80||list->sequence!=i.plan->sequence){atomicOr(&i.status->error,0x400u);return false;}return true;
}
__global__ void gu_persistent(G::Images i,const WorkList*list,int bank){
 __shared__ unsigned n;__shared__ const unsigned char*w;__shared__ int job;
 if(threadIdx.x==0)n=list_valid(i,list,bank)?list->count:0;__syncthreads();
 for(unsigned work=blockIdx.x;work<n*(2*G::F/16);work+=gridDim.x){
  if(threadIdx.x==0){job=list->rows[work/(2*G::F/16)];w=nullptr;if(job>=0&&job<80&&G::valid(i,bank)&&G::active(i,bank,job))w=G::weights(i,bank,job);else atomicOr(&i.status->error,0x400u);}
  __syncthreads();
  if(w){
   const int lane=threadIdx.x&15,r=int(work%(2*G::F/16))*16+int(threadIdx.x)/16,row=r%G::F;
   const auto*wr=w+(r>=G::F?L::UpOffset:0)+size_t(row)*L::GuRow;
   const float d=L::row_dot(wr,i.input_q+size_t(i.plan->jobs[job].token)*L::InputQ,G::H/64,lane);
   if(lane==0)(r>=G::F?i.up_raw:i.gate_raw)[size_t(job)*G::F+row]=d;
   if(r==0&&lane==0)atomicAdd(&i.status->jobs,1u);
  }
  __syncthreads(); // no CTA thread overwrites shared job/weight before peers finish
 }
}
__global__ void down_persistent(G::Images i,const WorkList*list,int bank){
 __shared__ unsigned n;__shared__ const unsigned char*w;__shared__ int job;
 if(threadIdx.x==0)n=list_valid(i,list,bank)?list->count:0;__syncthreads();
 for(unsigned work=blockIdx.x;work<n*(G::H/16);work+=gridDim.x){
  if(threadIdx.x==0){job=list->rows[work/(G::H/16)];w=nullptr;if(job>=0&&job<80&&G::valid(i,bank)&&G::active(i,bank,job))w=G::weights(i,bank,job);else atomicOr(&i.status->error,0x400u);}
  __syncthreads();
  if(w){
   const int lane=threadIdx.x&15,r=int(work%(G::H/16))*16+int(threadIdx.x)/16;
   const float d=L::row_dot(w+L::DownOffset+size_t(r)*L::DownRow,i.hidden_q+size_t(job)*L::HiddenQ,G::F/64,lane);
   if(lane==0){const float sd=reinterpret_cast<const float*>(w+L::TailOffset)[2];const float z=__fmul_rn(d,sd);const size_t at=size_t(job)*G::H+r;i.down_raw[at]=d;i.output[at]=z;if(!L::finite(d)||!L::finite(sd)||!L::finite(z))atomicOr(&i.status->error,8u);}
   if(r==0&&lane==0)atomicAdd(&i.status->output_rows,1u);
  }
  __syncthreads();
 }
}
}
cudaError_t chain(const G::Images&i,WorkList*list,int bank,int grid,cudaStream_t s){
 if(!s||bank<0||bank>=4||grid<1||grid>2048||!L::aligned(list,8)||!L::aligned(i.plan,8)||!L::aligned(i.plan_error,4)||!L::aligned(i.physical,8)||!L::aligned(i.input,4)||!i.input_q||!L::aligned(i.gate_raw,4)||!L::aligned(i.up_raw,4)||!L::aligned(i.gate_scaled,4)||!L::aligned(i.up_scaled,4)||!L::aligned(i.hidden,4)||!i.hidden_q||!L::aligned(i.down_raw,4)||!L::aligned(i.output,4)||!L::aligned(i.status,8))return cudaErrorInvalidValue;
 L::Span spans[]={{i.plan,sizeof(R::Plan)},{i.plan_error,4},{i.physical,sizeof(four_banks_v3::PhysicalDeviceView)},{i.input,size_t(G::MaxT)*G::H*4},{i.input_q,size_t(G::MaxT)*L::InputQ},{i.gate_raw,size_t(G::Capacity)*G::F*4},{i.up_raw,size_t(G::Capacity)*G::F*4},{i.gate_scaled,size_t(G::Capacity)*G::F*4},{i.up_scaled,size_t(G::Capacity)*G::F*4},{i.hidden,size_t(G::Capacity)*G::F*4},{i.hidden_q,size_t(G::Capacity)*L::HiddenQ},{i.down_raw,size_t(G::Capacity)*G::H*4},{i.output,size_t(G::Capacity)*G::H*4},{i.status,sizeof(G::Status)},{list,sizeof(WorkList)}};
 if(!L::disjoint(spans,15))return cudaErrorInvalidValue;int current=-1;cudaError_t e=cudaGetDevice(&current);if(e!=cudaSuccess)return e;if(current!=bank)return cudaErrorInvalidDevice;
 e=cudaMemsetAsync(i.status,0,sizeof(G::Status),s);if(e!=cudaSuccess)return e;
 compact<<<1,1,0,s>>>(i,list,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 G::input_q<<<dim3(1,G::MaxT),128,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 gu_persistent<<<grid,256,0,s>>>(i,list,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 G::hidden<<<dim3((G::F+255)/256,G::Capacity),256,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 G::hidden_q<<<(G::Capacity*(G::F/32)+127)/128,128,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 down_persistent<<<grid,256,0,s>>>(i,list,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 G::publish<<<1,1,0,s>>>(i,bank);return cudaGetLastError();
}
}
