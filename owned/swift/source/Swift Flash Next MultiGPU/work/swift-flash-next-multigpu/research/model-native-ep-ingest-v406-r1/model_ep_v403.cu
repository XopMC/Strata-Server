#include "model_ep_v403.hpp"
namespace model_ep_v403 {
static_assert(offsetof(Frame,abort)==4&&offsetof(Frame,current_window)==8&&offsetof(Frame,plan_window)==16&&offsetof(Frame,T)==24&&offsetof(Frame,layer)==28&&offsetof(Frame,error)==32,"V406 mapped header lane ABI");
namespace {
__device__ uint32_t cv(const uint32_t*p){uint32_t x;asm volatile("ld.global.cv.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;}
__device__ uint64_t cv64(const uint64_t*p){uint64_t x;asm volatile("ld.global.cv.u64 %0,[%1];":"=l"(x):"l"(p):"memory");return x;}
__global__ void ingest_kernel(Frame*f,DevicePlan*out,int bank,int T,int layer,int ring){
 __shared__ int valid;
 if(threadIdx.x==0){
  // Match the proven original Strata mapped-doorbell polling policy. Explicit
  // volatile loads plus a pause; system fence after observing publication.
  const volatile uint32_t*ready=&f->ready;const volatile uint32_t*abort=&f->abort;
  while(*ready<uint32_t(ring)&&!*abort)__nanosleep(64);
  __threadfence_system();
 }
 __syncthreads();
 // Fixed registered bounds permit speculative metadata copying only. No
 // pointer is dereferenced until all admission checks complete on this stream.
 const auto*src=reinterpret_cast<const uint32_t*>(&f->bank[bank]);
 auto*dst=reinterpret_cast<uint32_t*>(out);
 for(int i=threadIdx.x;i<int(sizeof(DevicePlan)/4);i+=blockDim.x)dst[i]=cv(src+i);
 __syncthreads();
 if(threadIdx.x<32){
  const int lane=int(threadIdx.x);
  uint32_t word=0;
  constexpr int field[8]={1,8,6,7,4,5,2,3};
  if(lane<8)word=cv(reinterpret_cast<const uint32_t*>(f)+field[lane]);
  const uint32_t current_lo=__shfl_sync(0xffffffffu,word,6);
  const uint32_t current_hi=__shfl_sync(0xffffffffu,word,7);
  bool good=true;
  if(lane==0||lane==1)good=word==0;
  else if(lane==2)good=int32_t(word)==T;
  else if(lane==3)good=int32_t(word)==layer;
  else if(lane==4)good=word==current_lo;
  else if(lane==5)good=word==current_hi;
  const unsigned mask=__ballot_sync(0xffffffffu,good);
  if(lane==0){
   const int G=out->counts[0],R=out->counts[1];
   valid=mask==0xffffffffu && G>=0&&G<=T*10&&R>=G&&R<=T*10&&
    out->counts[2]==0&&out->counts[3]==0&&out->start[0]==0&&out->start[G]==R;
   if(!valid)reinterpret_cast<volatile uint32_t*>(f->status)[bank]=1;
  }
 }
 __syncthreads();
 if(!valid)for(int i=threadIdx.x;i<int(sizeof(DevicePlan)/4);i+=blockDim.x)dst[i]=0;
}
__global__ void copy_kernel(const DevicePlan*p,const uint4*src,uint4*dst,int packs){
 if(p->counts[0]==0)return;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<packs;i+=blockDim.x*gridDim.x)dst[i]=src[i];
}
__global__ void finish_kernel(Frame*f,int bank,int ring){
 __threadfence_system();reinterpret_cast<volatile uint32_t*>(f->completed)[bank]=uint32_t(ring);
}
}
cudaError_t ingest(Frame*f,DevicePlan*p,int bank,int T,int layer,int ring,cudaStream_t s){
 ingest_kernel<<<1,128,0,s>>>(f,p,bank,T,layer,ring);return cudaGetLastError();
}
cudaError_t copy_q8_if_needed(const DevicePlan*p,const uint8_t*src,uint8_t*dst,int T,cudaStream_t s){
 copy_kernel<<<4,128,0,s>>>(p,reinterpret_cast<const uint4*>(src),reinterpret_cast<uint4*>(dst),T*Qrow/16);return cudaGetLastError();
}
cudaError_t mark_completed(Frame*f,int bank,int ring,cudaStream_t s){finish_kernel<<<1,1,0,s>>>(f,bank,ring);return cudaGetLastError();}
}
