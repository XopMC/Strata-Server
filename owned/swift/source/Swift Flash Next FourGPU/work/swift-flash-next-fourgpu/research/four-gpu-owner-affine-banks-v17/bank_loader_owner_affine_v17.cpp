#include "bank_loader_v9.hpp"
#include "owner_affine_layout_v17.hpp"
#include "peer_matrix.hpp"
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>
namespace four_loader_v9 {namespace {
constexpr uint64_t Guard=256,MetadataBytes=uint64_t(B::Keys)*sizeof(B::PhysicalRef);
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
void ck(cudaError_t e,const char*s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,"V9 byte budget overflow");return a+b;}
struct Restore {
 int device=-1;
 Restore(){ck(cudaGetDevice(&device),"V9 save device");}
 void finish(){ck(cudaSetDevice(device),"V9 positive device restore");device=-1;}
 ~Restore(){if(device>=0&&cudaSetDevice(device)!=cudaSuccess)std::fprintf(stderr,"V9 FATAL failed device restore; terminate request\n");}
};
struct UploadPlan {
 std::array<std::vector<int>,4>keys;
 bool production=false;
 uint64_t bytes(int d)const{return uint64_t(keys[size_t(d)].size())*B::SlotStride;}
};
struct Bounce {unsigned char*host=nullptr;cudaEvent_t done=nullptr;bool pending=false;};
struct Device {
 unsigned char*allocation=nullptr;unsigned char*base=nullptr;B::PhysicalRef*refs=nullptr;
 uint64_t bank_bytes=0;cudaStream_t stream=nullptr;std::array<Bounce,2>bounce{};unsigned used=0;
};
struct Resources {
 std::array<Device,4>d;Source source;Receipt receipt;bool uploads_drained=false,closed=false;
 std::array<std::array<unsigned char,2*Guard>,4>guard_readback{};
 std::vector<B::PhysicalRef>refs_image,refs_readback; // retained across any failed async setup/drain
 // Only owned streams/allocations. Failure to establish positive drain keeps
 // this entire object/source lease alive through the custom deleter below.
 bool cleanup()noexcept{
  if(closed)return true;
  int old=-1;bool ok=cudaGetDevice(&old)==cudaSuccess;
  for(int k=0;k<4;++k){auto&x=d[size_t(k)];if(!x.stream&&!x.allocation&&!x.refs)continue;
   if(cudaSetDevice(k)!=cudaSuccess){ok=false;continue;}
   if(x.stream&&cudaStreamSynchronize(x.stream)!=cudaSuccess)ok=false;
  }
  if(!ok){if(old>=0)(void)cudaSetDevice(old);return false;}
  for(int k=0;k<4;++k){auto&x=d[size_t(k)];if(cudaSetDevice(k)!=cudaSuccess){ok=false;continue;}
   for(auto&b:x.bounce){if(b.done){if(cudaEventDestroy(b.done)==cudaSuccess)b.done=nullptr;else ok=false;}
    if(b.host){if(cudaFreeHost(b.host)==cudaSuccess)b.host=nullptr;else ok=false;}}
   if(x.refs){if(cudaFree(x.refs)==cudaSuccess)x.refs=nullptr;else ok=false;}
   if(x.allocation){if(cudaFree(x.allocation)==cudaSuccess){x.allocation=nullptr;x.base=nullptr;}else ok=false;}
   if(x.stream){if(cudaStreamDestroy(x.stream)==cudaSuccess)x.stream=nullptr;else ok=false;}
  }
  if(old>=0&&cudaSetDevice(old)!=cudaSuccess)ok=false;
  if(ok){source=Source{};closed=true;}return ok;
 }
};
std::shared_ptr<Resources>resources(){return std::shared_ptr<Resources>(new Resources,[](Resources*p){
 if(p->cleanup())delete p;
 else std::fprintf(stderr,"V9 FATAL owned-resource cleanup lacked positive credit; retaining allocations/source lease until process exit; no retry\n");
});}
uint64_t remaining(const FutureNeed&n){
 need(n.declared,"V9 every device needs an explicit remaining-allocation declaration");
 uint64_t v=0;for(auto b:{n.dense,n.prefill_peak,n.head,n.session,n.controller_other,n.safety_reserve})v=add(v,b);return v;
}
void validate(const Source&s,const Options&o){
 need(s.immutable_arena_owner&&s.get&&s.source_epoch&&s.one_immutable_arena,"V9 needs one retained immutable source arena");
 need(o.physical_epoch&&o.staging_slots_per_device>=1&&o.staging_slots_per_device<=2,"V9 epoch/staging admission");
 need(o.pinned_staging_limit>=uint64_t(4)*o.staging_slots_per_device*B::BlobBytes&&o.pinned_staging_limit<=64ull<<20,"V9 bounded pinned staging budget");
 for(const auto&n:o.future)(void)remaining(n);
}
std::shared_ptr<Resources>upload(const UploadPlan&p,const Source&s,const Options&o
#ifdef STRATA_BANK_LOADER_V9_TESTING
 ,unsigned fail_after_enqueues=0
#endif
){
 validate(s,o);admit_current_four();Restore restore;
 // Metadata preflight of source spans only. No payload traversal/hash/copy.
 std::array<std::vector<SourceSpan>,4>spans;
 uint64_t keys=0;
 for(int d=0;d<4;++d){need(!p.keys[size_t(d)].empty(),"V9 empty physical bank");
  spans[size_t(d)].reserve(p.keys[size_t(d)].size());
  for(int key:p.keys[size_t(d)]){
   need(key>=0&&key<B::Keys,"V9 source key bound");auto z=s.get(key);
   need(z.data&&z.bytes==B::BlobBytes&&z.bytes<=UINTPTR_MAX-uintptr_t(z.data),"V9 missing/invalid native40 source span");
   need(z.kind==HostKind::Pinned||z.kind==HostKind::Pageable,"V9 invalid source memory kind");
   if(z.kind==HostKind::Pinned){
    cudaPointerAttributes first{},last{};
    ck(cudaPointerGetAttributes(&first,z.data),"V9 pinned source registration");
    ck(cudaPointerGetAttributes(&last,z.data+z.bytes-1),"V9 pinned source end registration");
    need(first.type==cudaMemoryTypeHost&&last.type==cudaMemoryTypeHost,"V9 direct source is not registered host memory");
   }
   spans[size_t(d)].push_back(z);++keys;
  }
 }
 auto r=resources();r->source=s;r->receipt.source_epoch=s.source_epoch;r->receipt.physical_epoch=o.physical_epoch;r->receipt.keys=uint32_t(keys);
 // ALL four checks precede FIRST cudaMalloc/cudaHostAlloc/stream allocation.
 for(int d=0;d<4;++d){ck(cudaSetDevice(d),"V9 budget device");
  cudaStreamCaptureStatus capture{};ck(cudaStreamIsCapturing(nullptr,&capture),"V9 startup capture status");need(capture==cudaStreamCaptureStatusNone,"V9 startup loader in capture");
  size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"V9 budget free bytes");
  auto&b=r->receipt.budget[size_t(d)];b.free_before=free;b.total=total;b.future=o.future[size_t(d)];
  b.owned_request=add(add(p.bytes(d),2*Guard),p.production?MetadataBytes:0);b.future_remaining=remaining(o.future[size_t(d)]);
  need(add(b.owned_request,b.future_remaining)<=b.free_before,"V9 explicit bank+remaining peak exceeds current free memory");
  std::fprintf(stderr,"V9_BUDGET device=%d free=%llu owned=%llu remaining=%llu total=%llu not_model_fit_proof=1\n",d,(unsigned long long)b.free_before,(unsigned long long)b.owned_request,(unsigned long long)b.future_remaining,(unsigned long long)b.total);
  const auto&n=b.future;std::fprintf(stderr,"V9_RESERVED device=%d dense=%llu PP=%llu head=%llu session=%llu other=%llu safety=%llu\n",d,(unsigned long long)n.dense,(unsigned long long)n.prefill_peak,(unsigned long long)n.head,(unsigned long long)n.session,(unsigned long long)n.controller_other,(unsigned long long)n.safety_reserve);
 }
 for(int d=0;d<4;++d){ck(cudaSetDevice(d),"V9 allocation device");auto&x=r->d[size_t(d)];x.bank_bytes=p.bytes(d);
  ck(cudaStreamCreateWithFlags(&x.stream,cudaStreamNonBlocking),"V9 owned upload stream");
  ck(cudaMalloc(reinterpret_cast<void**>(&x.allocation),size_t(add(x.bank_bytes,2*Guard))),"V9 exact bank allocation");x.base=x.allocation+Guard;
  need(uintptr_t(x.base)%256==0,"V9 bank alignment");
  if(p.production)ck(cudaMalloc(reinterpret_cast<void**>(&x.refs),MetadataBytes),"V9 refs allocation");
  // Deterministic outer guards and slot padding; all payload bytes are replaced
  // by unchanged source H2D. Startup only; no numerical use of this pattern.
  ck(cudaMemsetAsync(x.allocation,0xa5,size_t(add(x.bank_bytes,2*Guard)),x.stream),"V9 bank initialization");
 }
 // Round-robin source enqueue drives four independent owned upload streams.
 size_t largest=0;for(const auto&v:spans)largest=std::max(largest,v.size());
#ifdef STRATA_BANK_LOADER_V9_TESTING
 unsigned queued=0;
#endif
 for(size_t slot=0;slot<largest;++slot)for(int d=0;d<4;++d){
  if(slot>=spans[size_t(d)].size())continue;ck(cudaSetDevice(d),"V9 upload owner");auto&x=r->d[size_t(d)];const auto&z=spans[size_t(d)][slot];const unsigned char*from=z.data;
  if(z.kind==HostKind::Pageable){auto&b=x.bounce[x.used++%o.staging_slots_per_device];
   if(!b.host){ck(cudaHostAlloc(reinterpret_cast<void**>(&b.host),B::BlobBytes,cudaHostAllocPortable),"V9 bounded pinned staging allocation");
    r->receipt.peak_pinned_staging=add(r->receipt.peak_pinned_staging,B::BlobBytes);need(r->receipt.peak_pinned_staging<=o.pinned_staging_limit,"V9 pinned staging cap");
    ck(cudaEventCreateWithFlags(&b.done,cudaEventDisableTiming),"V9 staging credit event");}
   if(b.pending)ck(cudaEventSynchronize(b.done),"V9 staging reuse completion");
   std::memcpy(b.host,z.data,B::BlobBytes);from=b.host;
   ck(cudaMemcpyAsync(x.base+uint64_t(slot)*B::SlotStride,from,B::BlobBytes,cudaMemcpyHostToDevice,x.stream),"V9 bounced H2D");
   ck(cudaEventRecord(b.done,x.stream),"V9 staging completion record");b.pending=true;
   r->receipt.bounced_bytes[size_t(d)]+=B::BlobBytes;
  }else{
   // Source adapter's pinned() flag is an explicit original registered-span
   // contract. No auto-registration/pageable DMA fallback if it is false.
   ck(cudaMemcpyAsync(x.base+uint64_t(slot)*B::SlotStride,from,B::BlobBytes,cudaMemcpyHostToDevice,x.stream),"V9 pinned source H2D");
   r->receipt.pinned_direct_bytes[size_t(d)]+=B::BlobBytes;
  }
  r->receipt.payload_bytes[size_t(d)]+=B::BlobBytes;
#ifdef STRATA_BANK_LOADER_V9_TESTING
  if(fail_after_enqueues&&++queued==fail_after_enqueues)throw std::runtime_error("V9 injected after queued upload");
#endif
 }
 for(int d=0;d<4;++d){ck(cudaSetDevice(d),"V9 upload completion owner");ck(cudaStreamSynchronize(r->d[size_t(d)].stream),"V9 full bank upload completion");}
 r->uploads_drained=true;r->receipt.four_uploads_complete=true;
 for(int d=0;d<4;++d){ck(cudaSetDevice(d),"V9 guard owner");auto&x=r->d[size_t(d)];auto&guard=r->guard_readback[size_t(d)];
  ck(cudaMemcpyAsync(guard.data(),x.allocation,Guard,cudaMemcpyDeviceToHost,x.stream),"V9 leading guard D2H");
  ck(cudaMemcpyAsync(guard.data()+Guard,x.base+x.bank_bytes,Guard,cudaMemcpyDeviceToHost,x.stream),"V9 trailing guard D2H");
  ck(cudaStreamSynchronize(x.stream),"V9 outer guard completion");for(auto b:guard)need(b==0xa5,"V9 outer bank guard mismatch");
  for(auto&b:x.bounce){if(b.done){ck(cudaEventDestroy(b.done),"V9 release staging event");b.done=nullptr;}if(b.host){ck(cudaFreeHost(b.host),"V9 release staging buffer");b.host=nullptr;}b.pending=false;}
 }
 r->receipt.outer_guards_identical=true;
 // All source H2D completed; source lease and bounded staging can now retire.
 r->source=Source{};restore.finish();return r;
}
}
void admit_current_four(){std::string error;if(!strata_peer_default::require_all(4,error))throw std::runtime_error("V9 mandatory native P2P: "+error);}
struct BankOwner::Session {
 std::shared_ptr<Resources>r;std::unique_ptr<B::PhysicalBanks>banks;
 std::array<B::PhysicalDeviceView,4>views{};std::mutex mutex;bool closing=false;
};
BankOwner::BankOwner(std::shared_ptr<Session>s):s_(std::move(s)){}
BankOwner::~BankOwner()=default;
std::unique_ptr<BankOwner>BankOwner::load(const Source&source,const Options&options){
 auto layout=four_owner_affine_v17::make_owner_affine_layout_v17();UploadPlan p;p.production=true;
 for(int d=0;d<4;++d)p.keys[size_t(d)].assign(layout.slot_keys[size_t(d)].begin(),layout.slot_keys[size_t(d)].end());
 auto r=upload(p,source,options);auto s=std::make_shared<Session>();s->r=r;
 std::array<B::BankBinding,4>binding;
 for(int d=0;d<4;++d)binding[size_t(d)]={r->d[size_t(d)].base,r->d[size_t(d)].bank_bytes,r,true};
 s->banks=std::make_unique<B::PhysicalBanks>(std::move(layout),std::move(binding),options.physical_epoch);
 // Retain BOTH async metadata sources and readback destinations in Resources
 // until positive completion; error unwinding cannot destroy pageable vectors
 // while an owned stream could still access them.
 auto lease=s->banks->lease();auto host=lease.host_refs();
 r->refs_image.assign(host.begin(),host.end());r->refs_readback.resize(B::Keys);
 Restore restore;
 for(int d=0;d<4;++d){auto&x=r->d[size_t(d)];ck(cudaSetDevice(d),"V9 metadata owner");
  ck(cudaMemcpyAsync(x.refs,r->refs_image.data(),MetadataBytes,cudaMemcpyHostToDevice,x.stream),"V9 refs H2D");
  ck(cudaStreamSynchronize(x.stream),"V9 refs H2D completion");
  ck(cudaMemcpyAsync(r->refs_readback.data(),x.refs,MetadataBytes,cudaMemcpyDeviceToHost,x.stream),"V9 independent refs D2H");
  ck(cudaStreamSynchronize(x.stream),"V9 refs D2H completion");
  need(std::memcmp(r->refs_image.data(),r->refs_readback.data(),MetadataBytes)==0,"V9 refs readback mismatch");
  s->views[size_t(d)]=lease.device_view(x.refs,options.physical_epoch);
 }
 restore.finish();r->receipt.metadata_readback=true;return std::unique_ptr<BankOwner>(new BankOwner(std::move(s)));
}
B::PhysicalLease BankOwner::lease()const{std::lock_guard<std::mutex>lock(s_->mutex);need(!s_->closing&&s_->banks,"V9 closed bank owner");return s_->banks->lease();}
B::PhysicalDeviceView BankOwner::view(int d)const{std::lock_guard<std::mutex>lock(s_->mutex);need(!s_->closing&&s_->banks&&d>=0&&d<4,"V9 closed/bad metadata view");return s_->views[size_t(d)];}
std::shared_ptr<const void>BankOwner::retain()const{std::lock_guard<std::mutex>lock(s_->mutex);need(!s_->closing,"V9 owner closing");return s_;}
const Receipt&BankOwner::receipt()const{return s_->r->receipt;}
void BankOwner::close(bool drained){
 std::lock_guard<std::mutex>lock(s_->mutex);need(drained&&s_.use_count()==1,"V9 close requires all external consumers drained/released");
 s_->closing=true;if(s_->banks){s_->banks->retire(true);s_->banks.reset();}
 need(s_->r->cleanup(),"V9 owned-resource cleanup failed; terminate without retry");
}
#ifdef STRATA_BANK_LOADER_V9_TESTING
namespace testing {
void exercise_mid_upload_failure(const Source&s,const Options&o){
 UploadPlan p;for(int d=0;d<4;++d)p.keys[size_t(d)]={d,d+4};
 const auto before=s.immutable_arena_owner.use_count();bool injected=false;
 try{(void)upload(p,s,o,3);}catch(const std::runtime_error&e){injected=std::string(e.what())=="V9 injected after queued upload";}
 need(injected,"V9 mid-upload fixture did not reach injection");
 need(s.immutable_arena_owner.use_count()==before,"V9 failure cleanup retained source: terminate fixture");
}
void exercise_small(const Source&s,const Options&o,const std::function<void(int,int,const unsigned char*,uint64_t,cudaStream_t)>&check){
 need(bool(check),"V9 missing independent fixture checker");UploadPlan p;
 for(int d=0;d<4;++d)p.keys[size_t(d)]={d,d+4};
 auto r=upload(p,s,o);Restore restore;
 for(int d=0;d<4;++d){ck(cudaSetDevice(d),"V9 fixture owner");for(size_t slot=0;slot<2;++slot)check(d,p.keys[size_t(d)][slot],r->d[size_t(d)].base+slot*B::SlotStride,B::BlobBytes,r->d[size_t(d)].stream);
  ck(cudaStreamSynchronize(r->d[size_t(d)].stream),"V9 fixture final consumer");}
 need(r->cleanup(),"V9 fixture cleanup");restore.finish();
}
}
#endif
}
