#include "row_adapter_v5.hpp"
#include <cstddef>
#include <cstring>
#include <limits>
namespace cpu_rows_v5 {
bool reference_plan(const Inputs&i,Plan&p,uint32_t&e){
 e=0;p=Plan{};p.T=i.T;p.layer=i.layer;p.sequence=i.sequence;p.logical_epoch=i.logical.logical_epoch;p.physical_epoch=i.physical.physical_epoch;
 for(int r=0;r<Capacity;++r)p.jobs[r].row=r;
 if(!i.ids||!i.logical.virtual_slots||!i.physical.refs||i.T<1||i.T>MaxT||i.layer<0||i.layer>=48||!i.sequence){e=BadInput;return false;}
 if(i.expected_physical_epoch!=i.physical.physical_epoch||i.expected_logical_epoch!=i.logical.logical_epoch||!i.expected_physical_epoch||!i.expected_logical_epoch){e=EpochMismatch;return false;}
 // Oracle first produces the logical route, then resolves only the miss subset.
 for(int t=0;t<i.T;++t)for(int k=0;k<K;++k){
  int r=t*K+k,expert=i.ids[r];Job&j=p.jobs[r];j.token=t;j.rank=k;
  if(expert<0||expert>=512){j.kind=Invalid;e|=BadRoute;continue;}
  j.key=i.layer*512+expert;j.kind=i.logical.virtual_slots[j.key]<0?Cpu:Native;
 }
 for(int r=0;r<i.T*K;++r){Job&j=p.jobs[r];if(j.kind!=Cpu)continue;
  const auto&ref=i.physical.refs[j.key];j.physical=ref;
  if(ref.key!=uint32_t(j.key)||ref.device<0||ref.device>=4||ref.physical_slot<0||ref.blob_bytes!=2764816u){j.kind=Invalid;e|=MissingCoverage;continue;}
  if(ref.physical_epoch!=i.expected_physical_epoch){j.kind=Invalid;e|=EpochMismatch;continue;}
  const uint64_t expected=uint64_t(ref.physical_slot)*2765056ull;
  const auto b=reinterpret_cast<uintptr_t>(i.physical.base[ref.device]);const uint64_t n=i.physical.bank_bytes[ref.device];
  if(!b||(b%256)||ref.byte_offset!=expected||expected>n||n-expected<2764816ull||expected>std::numeric_limits<uintptr_t>::max()-b||2764816ull>std::numeric_limits<uintptr_t>::max()-(b+expected)){j.kind=Invalid;e|=BadBounds;}
 }
 return e==0;
}
bool reference_scatter(const Plan&p,const Results&r,uint32_t*out,uint32_t&e){
 if(e)return false;
 // Validate all source planes first. This stronger oracle deliberately makes
 // no promise about destination contents on a failed GPU request.
 if(p.T<1||p.T>8||!out){e|=BadInput;return false;}
 for(int j=0;j<p.T*K;++j)if(p.jobs[j].kind==Cpu){
  int d=p.jobs[j].physical.device;
  if(d<0||d>=4||!r.rows[d]||!r.stamps[d]){e|=MissingCoverage;return false;}
  const auto&s=*r.stamps[d];if(s.sequence!=p.sequence||s.logical_epoch!=p.logical_epoch||s.physical_epoch!=p.physical_epoch){e|=BadResultStamp;return false;}
 }
 for(int t=0;t<p.T;++t)for(int k=0;k<K;++k){int j=t*K+k;if(p.jobs[j].kind==Cpu)std::memcpy(out+size_t(j)*H,r.rows[p.jobs[j].physical.device]+size_t(j)*H,H*sizeof(uint32_t));}
 return true;
}
}
