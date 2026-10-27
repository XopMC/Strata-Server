#include "dispatch_kernels_v10.cuh"
namespace four_dispatch_v10 { namespace {
namespace R=cpu_rows_v5;namespace B=four_banks_v3;namespace P=swift_ep_v26;
constexpr unsigned Bad=0x10000u;
__device__ bool physical_ok(const B::PhysicalRef&r,const B::PhysicalDeviceView&v,int key){
 if(r.key!=uint32_t(key)||r.device<0||r.device>=4||r.physical_slot<0||r.blob_bytes!=B::BlobBytes||r.physical_epoch!=v.physical_epoch)return false;
 const uint64_t off=r.byte_offset,n=v.bank_bytes[r.device];const uintptr_t base=reinterpret_cast<uintptr_t>(v.base[r.device]);
 return base&&!(base&255u)&&off==uint64_t(r.physical_slot)*B::SlotStride&&off<=n&&B::BlobBytes<=n-off&&off<=UINTPTR_MAX-base&&B::BlobBytes<=UINTPTR_MAX-(base+off);
}
__global__ void resolve_kernel(R::Plan*p,const R::Inputs*in,uint32_t*error,NativeMask*mask,uint32_t*stage,int layer,int T){
 mask->count=0;mask->error=0;mask->sequence=0;for(int i=0;i<80;++i)mask->rows[i]=0;
 if(*error||p->layer!=layer||p->T!=T||T<1||T>8||layer<0||layer>=48){*error|=Bad;mask->error=*error;atomicOr(stage,*error);return;}
 for(int row=0;row<T*10;++row){auto&j=p->jobs[row];if(j.kind!=R::Cpu&&j.kind!=R::Native){*error|=Bad;continue;}
  if(j.kind==R::Native){j.physical=in->physical.refs[j.key];if(!physical_ok(j.physical,in->physical,j.key))*error|=Bad;else mask->rows[mask->count++]=row;}
 }
 if(*error){mask->count=0;mask->error=*error;atomicOr(stage,*error);return;}mask->sequence=p->sequence;
}
__global__ void copy_kernel(OwnerView v,BankBuffers dst,BankBuffers src){
 const unsigned i=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 if(dst.cpu.input!=v.original_x)for(unsigned j=i;j<unsigned(v.T)*H;j+=stride)reinterpret_cast<uint32_t*>(const_cast<float*>(dst.cpu.input))[j]=reinterpret_cast<const uint32_t*>(v.original_x)[j];
 if(dst.native_q8!=v.original_native_q8)for(unsigned j=i;j<unsigned(v.T)*(Qrow/4);j+=stride)reinterpret_cast<uint32_t*>(dst.native_q8)[j]=static_cast<const uint32_t*>(v.original_native_q8)[j];
 if(dst.cpu.plan!=src.cpu.plan)for(unsigned j=i;j<sizeof(R::Plan)/4;j+=stride)reinterpret_cast<uint32_t*>(const_cast<R::Plan*>(dst.cpu.plan))[j]=reinterpret_cast<const uint32_t*>(src.cpu.plan)[j];
 if(i==0&&dst.cpu.plan_error!=src.cpu.plan_error)*const_cast<uint32_t*>(dst.cpu.plan_error)=*src.cpu.plan_error;
}
__global__ void native_plan_kernel(BankBuffers b,int bank){
 auto*out=b.native_plan;const int lane=int(threadIdx.x);
 __shared__ int live,rows,cutoff,total_groups,total_entries;
 __shared__ int32_t key[80],device[80];
 __shared__ uint32_t kind[80];
 __shared__ int first[80],count[80],bad[80];
 for(unsigned i=unsigned(lane);i<sizeof(P::Plan)/4;i+=blockDim.x)reinterpret_cast<uint32_t*>(out)[i]=0;
 __syncthreads();
 if(lane==0){
  out->epoch=1;live=0;rows=0;
  if(*b.cpu.plan_error)out->error=Bad;
  else {const auto&p=*b.cpu.plan;const auto&v=*b.cpu.physical;
   if(p.physical_epoch!=v.physical_epoch||p.T<1||p.T>8)out->error=Bad;
   else {live=1;rows=p.T*10;}
  }
 }
 __syncthreads();if(!live)return;
 const auto&p=*b.cpu.plan;const auto&v=*b.cpu.physical;
 if(lane<80){
  if(lane<rows){key[lane]=p.jobs[lane].key;kind[lane]=p.jobs[lane].kind;device[lane]=p.jobs[lane].physical.device;}
  else {key[lane]=0;kind[lane]=R::Inactive;device[lane]=0;}
 }
 __syncthreads();
 if(lane<80){
  bool take=lane<rows&&kind[lane]==R::Native&&device[lane]==bank;
  if(take)for(int k=0;k<lane;++k)if(kind[k]==R::Native&&key[k]==key[lane])take=false;
  first[lane]=take?1:0;int n=0;
  if(take)for(int k=lane;k<rows;++k)if(kind[k]==R::Native&&key[k]==key[lane])++n;
  count[lane]=n;bad[lane]=take&&!physical_ok(p.jobs[lane].physical,v,p.jobs[lane].key);
 }
 __syncthreads();
 if(lane==0){
  cutoff=rows;for(int r=0;r<rows;++r)if(bad[r]){cutoff=r;break;}
  int g=0,n=0;for(int r=0;r<cutoff;++r){g+=first[r];n+=count[r];}total_groups=g;total_entries=n;
 }
 __syncthreads();
 if(lane<rows){
  if(first[lane]&&lane<cutoff){
   int g=0,n=0;for(int r=0;r<lane;++r){g+=first[r];n+=count[r];}
   out->ptr[g]=reinterpret_cast<unsigned long long>(v.base[bank])+p.jobs[lane].physical.byte_offset;
   out->start[g]=n;
  }
  if(kind[lane]==R::Native){
   int leader=lane;for(int r=0;r<lane;++r)if(kind[r]==R::Native&&key[r]==key[lane]){leader=r;break;}
   if(first[leader]&&leader<cutoff){
    int n=0;for(int r=0;r<leader;++r)n+=count[r];
    for(int r=leader;r<lane;++r)if(kind[r]==R::Native&&key[r]==key[lane])++n;
    out->dst[n]=n;out->tok[n]=lane/10;out->original[n]=lane;
   }
  }
 }
 __syncthreads();
 if(lane==0){
  // Original failure keeps ONLY earlier complete-group arrays, and no terminal
  // start write. Its public group/entry counts are zero even for that prefix.
  if(cutoff<rows){out->error=Bad;out->groups=out->entries=0;}
  else {out->groups=total_groups;out->entries=total_entries;out->start[total_groups]=total_entries;}
 }
}
__global__ void bank_commit_kernel(BankBuffers b,int bank){
 auto&s=*b.status;s={};const auto&p=*b.cpu.plan;s.error=*b.cpu.plan_error|b.native_plan->error|b.cpu.status->error;
 if(b.cpu.status->stamp.sequence!=p.sequence||b.cpu.status->stamp.logical_epoch!=p.logical_epoch||b.cpu.status->stamp.physical_epoch!=p.physical_epoch)s.error|=Bad;
 if(p.T<1||p.T>8){s.error|=Bad;return;}
 unsigned wanted=0;for(int r=0;r<p.T*10;++r)wanted+=unsigned(p.jobs[r].kind==R::Native&&p.jobs[r].physical.device==bank);
 s.native_rows=unsigned(b.native_plan->entries);if(s.native_rows!=wanted||b.native_plan->epoch!=1)s.error|=Bad;
 if(!s.error)s.stamp={p.physical_epoch,p.logical_epoch,p.sequence};__threadfence_system();
}
struct Sources {BankBuffers bank[4];};
__device__ bool stamp_ok(const BankStatus&s,const R::Plan&p){return !s.error&&s.stamp.sequence==p.sequence&&s.stamp.logical_epoch==p.logical_epoch&&s.stamp.physical_epoch==p.physical_epoch;}
__global__ void gather_kernel(OwnerView v,Sources src,bool cpu){
 auto&home=src.bank[v.device];const auto&p=*home.cpu.plan;const int row=blockIdx.x;
 if(atomicAdd(const_cast<uint32_t*>(home.cpu.plan_error),0u)){if(threadIdx.x==0)atomicOr(v.stage_error,*home.cpu.plan_error);return;}
 const auto&j=p.jobs[row];if(j.kind!=(cpu?R::Cpu:R::Native))return;
 if(j.physical.device<0||j.physical.device>=4||row>=p.T*10||!stamp_ok(*src.bank[j.physical.device].status,p)){if(threadIdx.x==0){atomicOr(const_cast<uint32_t*>(home.cpu.plan_error),Bad);atomicOr(v.stage_error,Bad);}return;}
 const uint32_t*from=reinterpret_cast<const uint32_t*>(cpu?src.bank[j.physical.device].cpu.output:src.bank[j.physical.device].native_rows);
 uint32_t*to=reinterpret_cast<uint32_t*>(cpu?v.parts:v.native_hits);
 for(int c=threadIdx.x;c<H;c+=blockDim.x)to[size_t(row)*H+c]=from[size_t(row)*H+c];
}
__global__ void ledger_kernel(OwnerView v,Sources src,bool final){
 unsigned e=*src.bank[v.device].cpu.plan_error|v.mask->error;
 for(int b=0;b<4;++b)e|=src.bank[b].status->error;
 if(e)atomicOr(v.stage_error,e);
 if(final){const auto&p=*src.bank[v.device].cpu.plan;v.receipt->sequence=p.sequence;v.receipt->error=*v.stage_error;v.receipt->T=uint32_t(p.T);v.receipt->layer=uint32_t(p.layer);__threadfence_system();v.receipt->done=0xa10u;__threadfence_system();}
}
}
namespace detail {
cudaError_t resolve_native(R::Plan*p,const R::Inputs*i,uint32_t*e,NativeMask*m,uint32_t*s,int l,int T,cudaStream_t stream){resolve_kernel<<<1,1,0,stream>>>(p,i,e,m,s,l,T);return cudaGetLastError();}
cudaError_t copy_inputs(OwnerView v,const BankBuffers&b,const BankBuffers&home,cudaStream_t stream){copy_kernel<<<4,256,0,stream>>>(v,b,home);return cudaGetLastError();}
cudaError_t native_plan(const BankBuffers&b,int bank,cudaStream_t s){native_plan_kernel<<<1,128,0,s>>>(b,bank);return cudaGetLastError();}
cudaError_t bank_commit(const BankBuffers&b,int bank,cudaStream_t s){bank_commit_kernel<<<1,1,0,s>>>(b,bank);return cudaGetLastError();}
Sources sources(const std::array<BankBuffers,4>&b){Sources s{};for(int k=0;k<4;++k)s.bank[k]=b[k];return s;}
cudaError_t gather_native(OwnerView v,const std::array<BankBuffers,4>&b,cudaStream_t s){auto src=sources(b);gather_kernel<<<80,256,0,s>>>(v,src,false);auto e=cudaGetLastError();if(e!=cudaSuccess)return e;ledger_kernel<<<1,1,0,s>>>(v,src,false);return cudaGetLastError();}
cudaError_t gather_cpu(OwnerView v,const std::array<BankBuffers,4>&b,cudaStream_t s){auto src=sources(b);gather_kernel<<<80,256,0,s>>>(v,src,true);auto e=cudaGetLastError();if(e!=cudaSuccess)return e;ledger_kernel<<<1,1,0,s>>>(v,src,true);return cudaGetLastError();}
}
cudaError_t reset_stage_error(uint32_t*p,cudaStream_t s){if(!p||!s)return cudaErrorInvalidValue;return cudaMemsetAsync(p,0,4,s);}
}
