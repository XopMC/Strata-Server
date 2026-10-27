// Reuse the sealed V86 helpers literally in this TU; compile this file instead
// of separately linking cpu_semantics_v86.cu. Original licenses remain included.
#include "../cpu-semantics-cuda-leaf-v86/cpu_semantics_v86.cu"
#include "grouped_v6.hpp"
namespace cpu_grouped_v6 { namespace {
namespace L=swift_cpu_semantics_v86;namespace R=cpu_rows_v5;namespace B=four_banks_v3;
constexpr unsigned BadPlan=0x100u,BadCounts=0x200u;
__device__ bool active(const Images&i,int bank,int row){
 return i.plan->jobs[row].kind==R::Cpu&&i.plan->jobs[row].physical.device==bank;
}
__device__ bool valid(const Images&i,int bank){
 if(*i.plan_error){atomicOr(&i.status->error,BadPlan);return false;}
 const auto&p=*i.plan;const auto&v=*i.physical;
 if(p.T<1||p.T>MaxT||!p.sequence||!p.logical_epoch||p.physical_epoch!=v.physical_epoch||!v.physical_epoch||!v.base[bank]){atomicOr(&i.status->error,BadPlan);return false;}
 return atomicAdd(&i.status->error,0u)==0;
}
__device__ const unsigned char*weights(const Images&i,int bank,int row){
 const auto&j=i.plan->jobs[row];const auto&r=j.physical;const auto&v=*i.physical;
 const uintptr_t base=reinterpret_cast<uintptr_t>(v.base[bank]);const uint64_t offset=r.byte_offset,limit=v.bank_bytes[bank];
 if(j.row!=row||j.token!=row/10||j.rank!=row%10||row>=i.plan->T*10||j.key<0||r.key!=uint32_t(j.key)||r.physical_slot<0||r.blob_bytes!=B::BlobBytes||r.physical_epoch!=i.plan->physical_epoch||offset!=uint64_t(r.physical_slot)*B::SlotStride||!base||(base&255u)||offset>limit||uint64_t(B::BlobBytes)>limit-offset||offset>UINTPTR_MAX-base||uint64_t(B::BlobBytes)>UINTPTR_MAX-(base+offset)){
  atomicOr(&i.status->error,BadPlan);return nullptr;
 }
 return v.base[bank]+offset;
}
__global__ void input_q(Images i,int bank){
 if(!valid(i,bank))return;const int token=blockIdx.y,block=blockIdx.x*blockDim.x+threadIdx.x;
 if(token>=i.plan->T||block>=H/32)return;
 bool used=false;for(int rank=0;rank<10;++rank)used|=active(i,bank,token*10+rank);
 if(!used)return;
 L::quant_block(i.input+size_t(token)*H+block*32,i.input_q+size_t(token)*L::InputQ+block*34,&i.status->error);
 if(block==0)atomicAdd(&i.status->input_tokens,1u);
}
__global__ void gu(Images i,int bank){
 const int job=blockIdx.y;
 __shared__ const unsigned char*w;
 if(threadIdx.x==0){w=nullptr;if(valid(i,bank)&&active(i,bank,job))w=weights(i,bank,job);}
 __syncthreads();if(!w)return; // uniform CTA admission before full-mask shuffles
 const int lane=threadIdx.x&15,r=int(blockIdx.x)*(blockDim.x/16)+int(threadIdx.x)/16;
 if(r>=2*F)return;const int row=r%F;
 const unsigned char*wr=w+(r>=F?L::UpOffset:0)+size_t(row)*L::GuRow;
 const float d=L::row_dot(wr,i.input_q+size_t(i.plan->jobs[job].token)*L::InputQ,H/64,lane);
 if(lane==0)(r>=F?i.up_raw:i.gate_raw)[size_t(job)*F+row]=d;
 if(r==0&&lane==0)atomicAdd(&i.status->jobs,1u);
}
__global__ void hidden(Images i,int bank){
 const int job=blockIdx.y;
 __shared__ const unsigned char*w;
 if(threadIdx.x==0){w=nullptr;if(valid(i,bank)&&active(i,bank,job))w=weights(i,bank,job);}
 __syncthreads();if(!w)return; // uniform CTA admission before full-mask shuffles
 const int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=F)return;const size_t at=size_t(job)*F+r;
 const float*sc=reinterpret_cast<const float*>(w+L::TailOffset);
 const float a=__fmul_rn(i.gate_raw[at],sc[0]),b=__fmul_rn(i.up_raw[at],sc[1]);i.gate_scaled[at]=a;i.up_scaled[at]=b;
 const float neg=__uint_as_float(__float_as_uint(a)^0x80000000u);
 const float d=__fadd_rn(1.f,L::exp_cpu(neg));const float value=__fmul_rn(__fdiv_rn(a,d),b);i.hidden[at]=value;
 if(!L::finite(a)||!L::finite(b)||!L::finite(value)||!L::finite(sc[0])||!L::finite(sc[1]))atomicOr(&i.status->error,4u);
}
__global__ void hidden_q(Images i,int bank){
 if(!valid(i,bank))return;const int block=blockIdx.x*blockDim.x+threadIdx.x;
 if(block>=Capacity*(F/32))return;const int job=block/(F/32),b=block%(F/32);if(!active(i,bank,job))return;
 L::quant_block(i.hidden+size_t(job)*F+b*32,i.hidden_q+size_t(job)*L::HiddenQ+b*34,&i.status->error);
}
__global__ void down(Images i,int bank){
 const int job=blockIdx.y;
 __shared__ const unsigned char*w;
 if(threadIdx.x==0){w=nullptr;if(valid(i,bank)&&active(i,bank,job))w=weights(i,bank,job);}
 __syncthreads();if(!w)return; // uniform CTA admission before full-mask shuffles
 const int lane=threadIdx.x&15,r=int(blockIdx.x)*(blockDim.x/16)+int(threadIdx.x)/16;if(r>=H)return;
 const float d=L::row_dot(w+L::DownOffset+size_t(r)*L::DownRow,i.hidden_q+size_t(job)*L::HiddenQ,F/64,lane);
 if(lane==0){const float sd=reinterpret_cast<const float*>(w+L::TailOffset)[2];const float z=__fmul_rn(d,sd);const size_t at=size_t(job)*H+r;i.down_raw[at]=d;i.output[at]=z;if(!L::finite(d)||!L::finite(sd)||!L::finite(z))atomicOr(&i.status->error,8u);}
 if(r==0&&lane==0)atomicAdd(&i.status->output_rows,1u);
}
__global__ void publish(Images i,int bank){
 if(!valid(i,bank))return;
 unsigned jobs=0,tokens=0;
 for(int t=0;t<i.plan->T;++t){bool used=false;for(int k=0;k<10;++k)if(active(i,bank,t*10+k)){++jobs;used=true;}tokens+=unsigned(used);}
 if(i.status->jobs!=jobs||i.status->output_rows!=jobs||i.status->input_tokens!=tokens){i.status->error|=BadCounts;return;}
 i.status->stamp={i.plan->physical_epoch,i.plan->logical_epoch,i.plan->sequence};
 // Root still joins the producer completion event before owner scatter.
 // This is a bank-result stamp, never the original host completion flag.
 __threadfence_system();
}
}
cudaError_t chain(const Images&i,int bank,cudaStream_t s){
 if(!s||bank<0||bank>=4||!L::aligned(i.plan,8)||!L::aligned(i.plan_error,4)||!L::aligned(i.physical,8)||!L::aligned(i.input,4)||!i.input_q||!L::aligned(i.gate_raw,4)||!L::aligned(i.up_raw,4)||!L::aligned(i.gate_scaled,4)||!L::aligned(i.up_scaled,4)||!L::aligned(i.hidden,4)||!i.hidden_q||!L::aligned(i.down_raw,4)||!L::aligned(i.output,4)||!L::aligned(i.status,8))return cudaErrorInvalidValue;
 L::Span spans[]={{i.plan,sizeof(R::Plan)},{i.plan_error,4},{i.physical,sizeof(B::PhysicalDeviceView)},{i.input,size_t(MaxT)*H*4},{i.input_q,size_t(MaxT)*L::InputQ},{i.gate_raw,size_t(Capacity)*F*4},{i.up_raw,size_t(Capacity)*F*4},{i.gate_scaled,size_t(Capacity)*F*4},{i.up_scaled,size_t(Capacity)*F*4},{i.hidden,size_t(Capacity)*F*4},{i.hidden_q,size_t(Capacity)*L::HiddenQ},{i.down_raw,size_t(Capacity)*H*4},{i.output,size_t(Capacity)*H*4},{i.status,sizeof(Status)}};
 if(!L::disjoint(spans,14))return cudaErrorInvalidValue;
 int current=-1;cudaError_t admission=cudaGetDevice(&current);if(admission!=cudaSuccess)return admission;if(current!=bank)return cudaErrorInvalidDevice;
 cudaError_t e=cudaMemsetAsync(i.status,0,sizeof(Status),s);if(e!=cudaSuccess)return e;
 input_q<<<dim3(1,MaxT),128,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 gu<<<dim3(2*F/16,Capacity),256,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 hidden<<<dim3((F+255)/256,Capacity),256,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 hidden_q<<<(Capacity*(F/32)+127)/128,128,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 down<<<dim3(H/16,Capacity),256,0,s>>>(i,bank);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 publish<<<1,1,0,s>>>(i,bank);return cudaGetLastError();
}
}
