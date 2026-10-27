#include "row_adapter_v5.cuh"
#include <cstddef>
namespace cpu_rows_v5 { namespace {
__device__ bool ref_ok(const Inputs&in,const four_banks_v3::PhysicalRef&r,int key,uint32_t&bad){
 if(r.key!=uint32_t(key)||r.device<0||r.device>=Devices||r.physical_slot<0||r.blob_bytes!=four_banks_v3::BlobBytes){bad=MissingCoverage;return false;}
 if(r.physical_epoch!=in.expected_physical_epoch){bad=EpochMismatch;return false;}
 const uint64_t slot=uint64_t(r.physical_slot),limit=in.physical.bank_bytes[r.device];
 const uintptr_t base=reinterpret_cast<uintptr_t>(in.physical.base[r.device]);
 if(!base||(base&255u)||slot>UINT64_MAX/four_banks_v3::SlotStride||r.byte_offset!=slot*four_banks_v3::SlotStride||
    r.byte_offset>limit||uint64_t(r.blob_bytes)>limit-r.byte_offset||r.byte_offset>UINTPTR_MAX-base||
    uint64_t(r.blob_bytes)>UINTPTR_MAX-(base+r.byte_offset)){bad=BadBounds;return false;}
 return true;
}
__global__ void planner(const Inputs*input,Plan*out,uint32_t*error){
 const int row=threadIdx.x;
 if(row>=Capacity)return;
 const Inputs in=*input;
 Job job{};job.row=row;
 const bool valid_input=in.ids&&in.logical.virtual_slots&&in.physical.refs&&in.T>=1&&in.T<=MaxT&&in.layer>=0&&in.layer<four_banks_v3::Layers&&in.sequence!=0;
 if(row==0){out->T=in.T;out->layer=in.layer;out->sequence=in.sequence;out->logical_epoch=in.logical.logical_epoch;out->physical_epoch=in.physical.physical_epoch;}
 if(!valid_input){job.kind=Invalid;out->jobs[row]=job;atomicOr(error,uint32_t(BadInput));return;}
 if(in.expected_physical_epoch!=in.physical.physical_epoch||in.expected_logical_epoch!=in.logical.logical_epoch||!in.expected_physical_epoch||!in.expected_logical_epoch){job.kind=Invalid;out->jobs[row]=job;atomicOr(error,uint32_t(EpochMismatch));return;}
 if(row>=in.T*K){out->jobs[row]=job;return;}
 job.token=row/K;job.rank=row%K;
 const int expert=in.ids[row];
 if(expert<0||expert>=four_banks_v3::ExpertsPerLayer){job.kind=Invalid;out->jobs[row]=job;atomicOr(error,uint32_t(BadRoute));return;}
 const int key=in.layer*four_banks_v3::ExpertsPerLayer+expert;job.key=key;
 if(in.logical.virtual_slots[key]>=0){job.kind=Native;out->jobs[row]=job;return;}
 job.physical=in.physical.refs[key];uint32_t bad=0;
 if(!ref_ok(in,job.physical,key,bad)){job.kind=Invalid;out->jobs[row]=job;atomicOr(error,bad);return;}
 job.kind=Cpu;out->jobs[row]=job;
}
__global__ void scatter(const Plan*plan,const Results*result,uint32_t*parts,uint32_t*error){
 const int row=blockIdx.x;
 if(atomicAdd(error,0u))return;
 const Job job=plan->jobs[row];
 if(job.kind==Inactive||job.kind==Native)return;
 if(job.kind!=Cpu||job.row!=row||job.token!=row/K||job.rank!=row%K||plan->T<1||plan->T>MaxT||row>=plan->T*K||job.physical.device<0||job.physical.device>=Devices){if(threadIdx.x==0)atomicOr(error,uint32_t(BadInput));return;}
 const int dev=job.physical.device;
 const uint32_t*src=result->rows[dev];const ResultStamp*sp=result->stamps[dev];
 if(!src||!sp){if(threadIdx.x==0)atomicOr(error,uint32_t(MissingCoverage));return;}
 const ResultStamp stamp=*sp;
 if(stamp.sequence!=plan->sequence||stamp.logical_epoch!=plan->logical_epoch||stamp.physical_epoch!=plan->physical_epoch){if(threadIdx.x==0)atomicOr(error,uint32_t(BadResultStamp));return;}
 // Unsigned-word loads/stores: preserve all payload bits, including signed zero
 // and NaN payloads in adapter tests. No floating arithmetic or CPU subtotal.
 for(int c=threadIdx.x;c<H;c+=blockDim.x)parts[size_t(row)*H+c]=src[size_t(row)*H+c];
}
struct Span {const void*p;size_t n;};
bool separate(const Span*v,int n){for(int i=0;i<n;++i){uintptr_t a=reinterpret_cast<uintptr_t>(v[i].p);if(!a||v[i].n>UINTPTR_MAX-a)return false;for(int j=0;j<i;++j){uintptr_t b=reinterpret_cast<uintptr_t>(v[j].p);if(a<b+v[j].n&&b<a+v[i].n)return false;}}return true;}
bool a4(const void*p){return p&&(reinterpret_cast<uintptr_t>(p)&3u)==0;}
bool a8(const void*p){return p&&(reinterpret_cast<uintptr_t>(p)&7u)==0;}
}
cudaError_t plan_async(const Inputs*in,Plan*p,uint32_t*e,cudaStream_t s){
 if(!s||!a8(in)||!a8(p)||!a4(e)||reinterpret_cast<const void*>(in)==reinterpret_cast<const void*>(p))return cudaErrorInvalidValue;
 Span spans[]={{in,sizeof(Inputs)},{p,sizeof(Plan)},{e,4}};if(!separate(spans,3))return cudaErrorInvalidValue;
 cudaError_t r=cudaMemsetAsync(e,0,sizeof(*e),s);if(r!=cudaSuccess)return r;
 planner<<<1,128,0,s>>>(in,p,e);return cudaGetLastError();
}
cudaError_t scatter_async(const Plan*p,const Results*r,uint32_t*out,uint32_t*e,cudaStream_t s){
 if(!s||!a8(p)||!a8(r)||!a4(out)||!a4(e))return cudaErrorInvalidValue;
 Span spans[]={{p,sizeof(Plan)},{r,sizeof(Results)},{out,size_t(Capacity)*H*4},{e,4}};if(!separate(spans,4))return cudaErrorInvalidValue;
 scatter<<<Capacity,256,0,s>>>(p,r,out,e);return cudaGetLastError();
}
}
