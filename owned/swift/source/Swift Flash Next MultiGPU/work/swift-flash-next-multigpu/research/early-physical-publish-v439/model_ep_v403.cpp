#include "source_moves_v435.hpp"
#include "early_credit_v439.hpp"
#include "capacity_map_v437.hpp"
#include "model_ep_v403.hpp"
#include "strata/core/verify.hpp"
#include "placement_v403.hpp"
#include "../model-plan-bridge-v400/plan_partition.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

namespace model_ep_v403 {
namespace {
struct Device {int old=-1;explicit Device(int d){cudaGetDevice(&old);if(cudaSetDevice(d)!=cudaSuccess)throw std::runtime_error("V403 device selection");}~Device(){if(old>=0)cudaSetDevice(old);}};
bool ck(cudaError_t rc,std::string&e,const char*at){if(rc==cudaSuccess)return true;e=std::string("V403 ")+at+": "+cudaGetErrorString(rc);return false;}
template<class T>T host_value(const T&x)noexcept{return *reinterpret_cast<const volatile T*>(&x);}
void fence(){std::atomic_thread_fence(std::memory_order_seq_cst);_mm_sfence();}
}
struct Bridge::Impl {
 struct Binding {
  Impl*impl=nullptr;int home=-1;
  strata::core::GpuPlanSink*sink=nullptr;
  void(*publish)(void*)=nullptr;
  void(*fetch)(void*,const uint8_t*const*,int,size_t)=nullptr;
  void*ctx=nullptr;
  strata::core::ExpertDispatch*d=nullptr;
  const int32_t*ids=nullptr;int T=0,layer=-1,cpu=0;bool borrowed=false,published=false;
 } bind[2];
 struct Owner {
  Frame*host=nullptr;Frame*alias[2]{};DevicePlan*plan[2]{};
  cudaStream_t stream=nullptr;uint8_t*q8=nullptr;void*scratch=nullptr;
  cudaGraph_t graph[9][24]{};
  cudaGraph_t parent[9][24]{};
  cudaGraphNode_t child[9][24]{};
  unsigned long long capture_id[9][24]{};
  const uint8_t*source=nullptr;void*local_scratch=nullptr;float*hits=nullptr;
  bool prepared[9]{},bound[9][24]{},local_bound[9][24]{};
 } owner[2];
 std::array<Home,2>homes;
 std::array<std::vector<model_ep_v400::Physical>,2>map;
 size_t moved_pairs=0;
 early_credit_v439::Credit early;
 uint64_t early_bytes=0;
 std::chrono::steady_clock::time_point early_start;
 std::array<void*,2>keys{};
 model_ep_v400::Placement placement;
 cudaStream_t swap_stream[2]{};cudaEvent_t swap_event[2]{};uint8_t*temp=nullptr;
 cudaEvent_t adapt_event[2][2]{};
 bool adapt_pending[2][2]{};
 uint64_t adapt_copies=0,adapt_foreign=0,adapt_bytes=0,deferred_refills=0;
 bool configured=false,rehomed=false,request=false,finished=false,quiesced=false;
 bool awaiting_prompt=false,identity_ready=false;uint64_t request_number=0;
 uint64_t window=0,publications=0,foreign_groups=0,foreign_rows=0;
 const char*failure=nullptr;
 void fail(const char*why)noexcept{if(!failure)failure=why;for(auto&o:owner)if(o.host)o.host->error=1;}
 int home(const void*k)const noexcept{return k==keys[0]?0:k==keys[1]?1:-1;}
 void release_all()noexcept{for(auto&o:owner)if(o.host){o.host->abort=1;fence();*reinterpret_cast<volatile uint32_t*>(&o.host->ready)=UINT32_MAX;}fence();}
 static void publish_hook(void*ctx)noexcept {
  auto&b=*static_cast<Binding*>(ctx);auto&p=*b.impl;auto&f=*p.owner[b.home].host;
  model_ep_v400::Result out;
  bool good=b.borrowed&&!b.published&&b.d&&b.ids&&!p.failure&&p.rehomed;
  if(good){
   std::array<int32_t,512>resident;std::copy_n(b.d->host_res+size_t(b.layer)*512,512,resident.begin());
   const auto&s=*b.sink;int G=s.counts[0],R=s.counts[1];
   good=G>=0&&G<=b.T*10&&R>=G&&R<=b.T*10;
   if(good){
    model_ep_v400::Original in{b.home,b.layer,b.T,b.cpu,
      {b.ids,size_t(b.T*10)},resident,{s.counts,4},{s.start,size_t(G+1)},
      {s.dst,size_t(R)},{s.tok,size_t(R)},
      {reinterpret_cast<const uint64_t*>(s.ptr),size_t(G)}};
    good=p.placement.partition(in,out)==model_ep_v400::Error::Ok;
   }
  }
  const int ring=b.layer%24+1;
  if(good&&ring>1)good=host_value(f.completed[0])==uint32_t(ring-1)&&host_value(f.completed[1])==uint32_t(ring-1);
  f.bank[0]={};f.bank[1]={};
  if(good){for(int bank=0;bank<2;++bank){auto&dst=f.bank[bank];const auto&src=out.plan[bank];
    dst.counts[0]=src.groups;dst.counts[1]=src.entries;
    std::copy_n(src.start.data(),src.groups+1,dst.start);std::copy_n(src.dst.data(),src.entries,dst.dst);
    std::copy_n(src.tok.data(),src.entries,dst.tok);std::copy_n(src.ptr.data(),src.groups,dst.ptr);
   }
   ++p.publications;p.foreign_groups+=out.plan[1-b.home].groups;p.foreign_rows+=out.plan[1-b.home].entries;
  }else {p.fail("original publication/physical partition/retirement refused");f.abort=1;}
  f.T=b.T;f.layer=b.layer;f.plan_window=p.window;b.published=true;
  fence();*reinterpret_cast<volatile uint32_t*>(&f.ready)=good?uint32_t(ring):UINT32_MAX;fence();
  // BOTH branch bodies are immutable before the original union publication.
  // Even a refusal releases valid empty branches and original waiters; final run rejects.
  b.publish(b.ctx);
 }
 static void fetch_hook(void*ctx,const uint8_t*const*src,int n,size_t bytes)noexcept{
  auto&b=*static_cast<Binding*>(ctx);if(n!=0)b.impl->fail("unexpected PCIe fetch");b.fetch(b.ctx,src,n,bytes);
 }
 ~Impl(){
  // ExitGuard is declared after both original verifier owners, so it drains
  // both and restores their callbacks while they are still alive. Their parent
  // graphs can outlive these buffers only as quiescent, never-replayed handles.
  for(int h=0;h<2;++h){
   Device on(1-h);for(auto&set:owner[h].graph)for(auto&g:set)if(g)cudaGraphDestroy(g);
   if(owner[h].stream)cudaStreamDestroy(owner[h].stream);
   if(owner[h].q8)cudaFree(owner[h].q8);if(owner[h].scratch)cudaFree(owner[h].scratch);
  }
  for(int bank=0;bank<2;++bank){Device on(bank);for(auto&o:owner)if(o.plan[bank])cudaFree(o.plan[bank]);
   if(swap_stream[bank])cudaStreamDestroy(swap_stream[bank]);if(swap_event[bank])cudaEventDestroy(swap_event[bank]);
   for(int h=0;h<2;++h)if(adapt_event[h][bank])cudaEventDestroy(adapt_event[h][bank]);
   if(bank==0&&temp)cudaFree(temp);
  }
  for(auto&o:owner)if(o.host)cudaFreeHost(o.host);
 }
};
namespace {Bridge::Impl*active=nullptr;}
Bridge::Bridge()=default;
Bridge::~Bridge(){if(active==p_.get())active=nullptr;}
ExitGuard::~ExitGuard(){std::string e;try{if(bridge_&&!bridge_->quiesce(e)){std::fprintf(stderr,"V403 uncertain parent drain: %s\n",e.c_str());std::abort();}}catch(...){std::fprintf(stderr,"V403 exceptional parent drain; refusing buffer reuse\n");std::abort();}}
bool Bridge::quiesce(std::string&e){
 if(!p_||!p_->configured||p_->quiesced)return true;auto&p=*p_;p.release_all();
 // Both Verifiers are live by ExitGuard construction order. No device-wide
 // sync: release original waits, then positively drain their captured parents.
 for(int h=0;h<2;++h)if(!static_cast<strata::core::Verifier*>(p.keys[h])->release_gpu_waits(5000)){e="original parent did not drain";return false;}
 // Private copies can still be retiring after the last original logical event
 // was recorded. Drain these streams before any cache or private event is freed.
 for(int bank=0;bank<2;++bank){Device on(bank);if(!ck(cudaStreamSynchronize(p.swap_stream[bank]),e,"adaptive private stream drain"))return false;}
 for(auto&b:p.bind){b.sink->ctx=b.ctx;b.sink->publish=b.publish;b.sink->fetch=b.fetch;}
 p.quiesced=true;p.configured=false;active=nullptr;return true;
}

bool requested(){const char*p=std::getenv("STRATA_MODEL_EP_V403");return p&&p[0]=='1'&&p[1]=='\0';}
bool Bridge::enabled()const noexcept{return bool(p_);}
bool owns(const void*k)noexcept{return active&&active->configured&&active->home(k)>=0;}
bool captured(const void*k,int T)noexcept{if(!owns(k)||T<1||T>8)return false;const auto&o=active->owner[active->home(k)];return std::all_of(std::begin(o.bound[T]),std::end(o.bound[T]),[](bool v){return v;});}
bool Bridge::configure(std::array<Home,2>h,std::array<void*,2>keys,
 std::array<strata::core::GpuPlanSink*,2>sinks,std::string&e){
 if(p_||active||!keys[0]||!keys[1]||keys[0]==keys[1]){e="V403 duplicate or invalid model owner";return false;}
 p_=std::make_unique<Impl>();auto&p=*p_;p.homes=std::move(h);p.keys=keys;
 std::array<model_ep_v400::Home,2>view;
 for(int b=0;b<2;++b){auto&v=p.homes[b];if(!v.base||v.offsets.empty()||v.offsets.size()>24*512||v.payload!=2764816){e="V403 private actual256 cache geometry refused";return false;}
  // Full map is constructed below from complete identity plus whole in-bounds pairs.
  view[b]={uint64_t(reinterpret_cast<uintptr_t>(v.base)),v.bytes,v.payload,v.offsets};
  auto*s=sinks[b];if(!s||!s->publish||!s->fetch||!s->ctx){e="V403 original sink callbacks absent";return false;}
  p.bind[b].impl=&p;p.bind[b].home=b;p.bind[b].sink=s;p.bind[b].ctx=s->ctx;p.bind[b].publish=s->publish;p.bind[b].fetch=s->fetch;
 }
 static_assert(std::size(swap0)==3322&&std::size(swap1)==3322);
 if(!capacity_map_v437::build({p.homes[0].offsets.size(),p.homes[1].offsets.size()},std::span(swap0),std::span(swap1),p.map,p.moved_pairs,e))return false;
 if(!p.placement.initialize(view,{p.map[0],p.map[1]},1)){e="V403 full physical map admission refused";return false;}
 for(int b=0;b<2;++b){Device on(b);int can=0;if(!ck(cudaDeviceCanAccessPeer(&can,b,1-b),e,"peer capability")||!can){e="V403 native P2P required";return false;}
  if(!ck(cudaStreamCreateWithFlags(&p.swap_stream[b],cudaStreamNonBlocking),e,"swap stream")||!ck(cudaEventCreateWithFlags(&p.swap_event[b],cudaEventDisableTiming),e,"swap event"))return false;
  for(int h=0;h<2;++h)if(!ck(cudaEventCreateWithFlags(&p.adapt_event[h][b],cudaEventDisableTiming),e,"adaptive physical completion event"))return false;
  for(auto&o:p.owner)if(!ck(cudaMalloc(reinterpret_cast<void**>(&o.plan[b]),sizeof(DevicePlan)),e,"device plan"))return false;
 }
 for(int hidx=0;hidx<2;++hidx){auto&o=p.owner[hidx];Device on(hidx);
  if(!ck(cudaHostAlloc(reinterpret_cast<void**>(&o.host),sizeof(Frame),cudaHostAllocMapped|cudaHostAllocPortable),e,"mapped control"))return false;
  new(o.host)Frame{};
  for(int b=0;b<2;++b){Device alias_on(b);if(!ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&o.alias[b]),o.host,0),e,"control alias"))return false;}
  Device peer_on(1-hidx);
  if(!ck(cudaStreamCreateWithFlags(&o.stream,cudaStreamNonBlocking),e,"foreign capture stream")||
    !ck(cudaMalloc(reinterpret_cast<void**>(&o.q8),8*Qrow),e,"foreign Q8")||
    !ck(cudaMalloc(&o.scratch,strata::kernels::native_expert_scratch_bytes(80,640)),e,"foreign native scratch"))return false;
 }
 {Device on(0);if(!ck(cudaMalloc(reinterpret_cast<void**>(&p.temp),32*p.homes[0].payload),e,"bounded rehome temporary"))return false;}
 for(auto&b:p.bind){b.sink->ctx=&b;b.sink->publish=&Impl::publish_hook;b.sink->fetch=&Impl::fetch_hook;}
 p.configured=true;active=&p;
 std::fprintf(stderr,"V437_CAPACITY_MAP slots=%zu,%zu moved_pairs=%zu complete_bijection=1 unmatched_pairs_identity=1 original_sync_adapt=1 SERIAL_REPEAT_V435=1 native40_original=1 PP_identity_then_rehome=1\n",p.homes[0].offsets.size(),p.homes[1].offsets.size(),p.moved_pairs);return true;
}
bool Bridge::begin_request(std::string&e){
 if(!p_)return true;auto&p=*p_;
 if(!p.configured||p.failure||p.quiesced||(p.request&&!p.finished)||p.awaiting_prompt||p.request_number==UINT64_MAX){e="V435 previous request incomplete/failed; no replay or retry";return false;}
 p.request=true;p.finished=false;p.awaiting_prompt=true;++p.request_number;return true;
}
bool Bridge::prompt_identity(strata::core::ExpertSource&source,const std::vector<int32_t>&resident,std::string&e){
 if(!p_)return true;auto&p=*p_;
 if(!p.configured||p.failure||!p.request||p.finished||!p.awaiting_prompt||p.publications!=48*p.window){e="V435 invalid pre-prompt lifecycle";return false;}
 for(const auto&b:p.bind)if(b.borrowed){e="V435 CPU dispatcher still owns prior frame";return false;}
 for(int h=0;h<2;++h)for(int bank=0;bank<2;++bank)if(p.adapt_pending[h][bank]){e="V435 unjoined original adaptive producer";return false;}
 // The caller waited the complete original verifier/commit and applied every
 // original pending adaptive event/map BEFORE this seam. PP has no live issuer.
 for(int bank=0;bank<2;++bank){Device on(bank);if(!ck(cudaStreamSynchronize(p.swap_stream[bank]),e,"prior physical producer retirement"))return false;}
 std::array<std::vector<source_moves_v435::Copy>,2>moves;
 if(!source_moves_v435::build(resident,{p.homes[0].offsets.size(),p.homes[1].offsets.size()},p.map,moves,e,true))return false;
 if(moves[0].size()!=p.moved_pairs||moves[1].size()!=p.moved_pairs){e="V435 identity source map incomplete";return false;}
 const bool restore=p.rehomed;uint64_t copied=0;const auto t0=std::chrono::steady_clock::now();
 struct Drain {Impl&p;bool success=false;~Drain(){if(!success){p.fail("V435 incomplete identity publication");for(int b=0;b<2;++b){Device on(b);if(p.swap_stream[b])cudaStreamSynchronize(p.swap_stream[b]);}}}}drain{p};
 if(restore){
  std::array<std::vector<const uint8_t*>,2>raw;
  for(int b=0;b<2;++b)for(const auto&item:moves[b]){const auto*src=source.blob_stable(item.key/512,item.key%512);if(!src){e="V435 immutable current adaptive source key missing";return false;}raw[b].push_back(src);}
  for(int b=0;b<2;++b){Device on(b);for(size_t i=0;i<moves[b].size();++i){auto*dst=p.homes[b].base+p.homes[b].offsets[moves[b][i].slot];if(!ck(cudaMemcpyAsync(dst,raw[b][i],p.homes[b].payload,cudaMemcpyHostToDevice,p.swap_stream[b]),e,"current source to original PP identity"))return false;source.note_async_read(raw[b][i],p.swap_stream[b]);copied+=p.homes[b].payload;}}
  for(int b=0;b<2;++b){Device on(b);if(!ck(cudaStreamSynchronize(p.swap_stream[b]),e,"positive identity publication"))return false;}
 }
 // Captured allocations/pointers and the original current logical residency
 // survive; only request-local credits reset after every old reader retired.
 if(!p.early.reset(e))return false;p.early_bytes=0;
 p.rehomed=false;p.identity_ready=true;p.awaiting_prompt=false;p.window=0;p.publications=0;p.foreign_groups=0;p.foreign_rows=0;p.adapt_copies=0;p.adapt_foreign=0;p.adapt_bytes=0;p.deferred_refills=0;drain.success=true;
 const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
 std::fprintf(stderr,"V435_PP_IDENTITY request=%llu restored=%d payload_bytes=%llu current_logical_keys=1 prior_verifier_commit_adapt_drained=1 next_PP_not_started=1 elapsed_ms=%.3f charged_prompt_TTFT=1\n",(unsigned long long)p.request_number,int(restore),(unsigned long long)copied,ms);return true;
}
bool Bridge::early_root_publish(strata::core::ExpertSource&source,const std::vector<int32_t>&resident,std::string&e){
 if(!p_)return true;auto&p=*p_;
 if(!p.configured||!p.request||p.finished||p.rehomed||!p.identity_ready||p.awaiting_prompt||p.window||p.failure){e="V439 invalid root retirement publication lifecycle";return false;}
 std::array<std::vector<source_moves_v435::Copy>,2>moves;
 if(!source_moves_v435::build(resident,{p.homes[0].offsets.size(),p.homes[1].offsets.size()},p.map,moves,e)||moves[0].size()!=p.moved_pairs||moves[1].size()!=p.moved_pairs){e="V439 incomplete original loan-map physical sources";return false;}
 std::vector<const uint8_t*>raw;raw.reserve(moves[0].size());
 for(const auto&item:moves[0]){const auto*ptr=source.blob_stable(item.key/512,item.key%512);if(!ptr){e="V439 immutable root source key missing";return false;}raw.push_back(ptr);}
 if(!p.early.begin(p.request_number,resident,e))return false;
 struct Drain {Impl&p;bool success=false;~Drain(){if(!success){p.early.failed();p.fail("V439 incomplete early root publication");Device on(0);cudaStreamSynchronize(p.swap_stream[0]);}}}drain{p};
 p.early_start=std::chrono::steady_clock::now();
 {Device on(0);for(size_t i=0;i<moves[0].size();++i){const auto&item=moves[0][i];auto*dst=p.homes[0].base+p.homes[0].offsets[item.slot];
  if(!ck(cudaMemcpyAsync(dst,raw[i],p.homes[0].payload,cudaMemcpyHostToDevice,p.swap_stream[0]),e,"early immutable Arena to retired root bank"))return false;
  source.note_async_read(raw[i],p.swap_stream[0]);p.early_bytes+=p.homes[0].payload;
 }}
 if(!p.early.submitted(e))return false;drain.success=true;
 std::fprintf(stderr,"V439_EARLY_ROOT request=%llu payload_bytes=%llu original_owner_positive_retirement=1 successor_may_live=1 full_logical_map_snapshotted=1 global_publication_pending=1\n",(unsigned long long)p.request_number,(unsigned long long)p.early_bytes);return true;
}
bool Bridge::rehome(strata::core::ExpertSource&source,const std::vector<int32_t>&resident,std::string&e){
 if(!p_)return true;auto&p=*p_;if(!p.configured||!p.request||p.rehomed||!p.identity_ready||p.awaiting_prompt||p.window||p.failure){e="V428 invalid post-PP physical publication lifecycle";return false;}
 if(!p.early.final(p.request_number,resident,e))return false;
 struct Drain {Impl&p;bool success=false;~Drain(){if(!success){p.fail("V428 incomplete physical source publication");for(int b=0;b<2;++b){Device on(b);if(p.swap_stream[b])cudaStreamSynchronize(p.swap_stream[b]);}}}}drain{p};
 const auto start=std::chrono::steady_clock::now();const size_t bytes=p.homes[0].payload;
 std::array<std::vector<source_moves_v435::Copy>,2>moves;
 if(!source_moves_v435::build(resident,{p.homes[0].offsets.size(),p.homes[1].offsets.size()},p.map,moves,e))return false;
 if(moves[0].size()!=p.moved_pairs||moves[1].size()!=p.moved_pairs){e="V428 selected complete permutation differs";return false;}
 // Resolve ALL immutable Arena sources before any GPU write. The selected
 // caller excludes staging/complement sources; its single mapped RAM bank and
 // source owner outlive both asynchronous producers and their positive drain.
 std::array<std::vector<const uint8_t*>,2>raw;
 for(int b=0;b<2;++b)for(const auto&item:moves[b]){
  const auto*ptr=source.blob_stable(item.key/512,item.key%512);
  if(!ptr){e="V428 original immutable source key missing";return false;}raw[b].push_back(ptr);
 }
 uint64_t copied=p.early_bytes;
 // Both original PP chains and original identity refills have completed at
 // this seam. No CPU/native eligibility or cache slot metadata is changed.
 // Queue each bank independently; no D2D rehome, no temp save, no legacy retry.
 for(int b=1;b<2;++b){Device on(b);for(size_t i=0;i<moves[b].size();++i){
  const auto&item=moves[b][i];auto*dst=p.homes[b].base+p.homes[b].offsets[item.slot];
  if(!ck(cudaMemcpyAsync(dst,raw[b][i],bytes,cudaMemcpyHostToDevice,p.swap_stream[b]),e,"direct immutable Arena to physical owner"))return false;
  source.note_async_read(raw[b][i],p.swap_stream[b]);copied+=bytes;
 }}
 for(int b=0;b<2;++b){Device on(b);if(!ck(cudaStreamSynchronize(p.swap_stream[b]),e,"positive physical source publication drain"))return false;}
 std::fprintf(stderr,"V436_DEFERRED_IDENTITY_REFILL request=%llu slots=%llu avoided_bytes=%llu final_physical_source_publication_completed=1 no_GEN_before_publication=1\n",(unsigned long long)p.request_number,(unsigned long long)p.deferred_refills,(unsigned long long)(p.deferred_refills*bytes));
 if(!p.early.complete(e))return false;
 std::fprintf(stderr,"V439_PHYSICAL_COMPLETE request=%llu payload_bytes=%llu both_streams_positive_drain=1 total_elapsed_ms=%.3f final_tail_ms=%.3f no_GEN_before_both=1\n",(unsigned long long)p.request_number,(unsigned long long)copied,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-p.early_start).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
 p.rehomed=true;p.identity_ready=false;drain.success=true;const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 std::fprintf(stderr,"V428_SOURCE_REHOME payload_bytes=%llu host_to_device=1 post_PP_positive_drain=1 immutable_Arena=1 original_logical_map=1 old_temporary_unused=1 no_D2D_rehome=1 elapsed_ms=%.3f charged_prompt_TTFT=1 SERIAL_REPEAT_V435=1\n",(unsigned long long)copied,ms);return true;
}
bool Bridge::defer_identity_refill(int bank,int slot,bool&defer,std::string&e){
 defer=false;if(!p_)return true;auto&p=*p_;
 if(!p.configured||p.failure||!p.request||p.finished||p.awaiting_prompt||!p.identity_ready||p.rehomed||p.window||bank<0||bank>1||slot<0||size_t(slot)>=p.map[bank].size()){
  e="V436 identity refill outside the positively closed PP transaction";return false;
 }
 // A bijective placement moves precisely the non-fixed physical destinations.
 // The final mandatory V428 rehome fills EVERY such destination from CURRENT
 // logical source keys after both PP readers retire, including these loan slots.
 const auto dst=p.map[bank][slot];defer=dst.bank!=bank||dst.slot!=slot;
 if(defer)++p.deferred_refills;return true;
}
bool Bridge::copy_adapt(int h,int slot,const uint8_t*source,size_t bytes,std::string&e){
 if(!p_||!p_->configured||!p_->rehomed||p_->finished||h<0||h>1||slot<0||
    size_t(slot)>=p_->map[h].size()||!source||bytes!=p_->homes[h].payload){e="V403R4 invalid physical adaptive fill";return false;}
 auto&p=*p_;const auto dst=p.map[h][slot];const int bank=dst.bank;
 if(bank<0||bank>1||dst.slot<0||size_t(dst.slot)>=p.homes[bank].offsets.size()){e="V403R4 invalid immutable adaptive destination";return false;}
 Device on(bank);
 auto*pointer=p.homes[bank].base+p.homes[bank].offsets[dst.slot];
 if(!ck(cudaMemcpyAsync(pointer,source,bytes,cudaMemcpyHostToDevice,p.swap_stream[bank]),e,"exact adaptive H2D physical bank"))return false;
 p.adapt_pending[h][bank]=true;++p.adapt_copies;p.adapt_foreign+=bank!=h;p.adapt_bytes+=bytes;return true;
}
bool Bridge::join_adapt(int h,cudaStream_t original,std::string&e){
 if(!p_||!p_->configured||!p_->rehomed||h<0||h>1||!original){e="V403R4 invalid logical adaptive completion";return false;}
 auto&p=*p_;
 // Keep ORIGINAL logical-home event/pending/host_res policy. Its stream waits
 // for every physical producer BEFORE the original completion event is recorded.
 // No per-expert events or device-wide barrier; source remains the original RAM arena.
 for(int bank=0;bank<2;++bank)if(p.adapt_pending[h][bank]){
  {Device on(bank);if(!ck(cudaEventRecord(p.adapt_event[h][bank],p.swap_stream[bank]),e,"physical adaptive batch event"))return false;}
  {Device on(h);if(!ck(cudaStreamWaitEvent(original,p.adapt_event[h][bank]),e,"original logical adaptive event dependency"))return false;}
  p.adapt_pending[h][bank]=false;
 }
 return true;
}
bool Bridge::begin_window(int T,std::string&e){
 if(!p_)return true;auto&p=*p_;if(!p.rehomed||p.failure||T<1||T>8||p.window==UINT64_MAX){e="V403 invalid window lifecycle";return false;}
 ++p.window;for(auto&o:p.owner){o.host->ready=0;o.host->abort=0;o.host->current_window=p.window;o.host->plan_window=0;o.host->T=0;o.host->layer=-1;o.host->error=0;for(int b=0;b<2;++b)o.host->completed[b]=o.host->status[b]=0;}
 fence();return true;
}
void Bridge::before_dispatch(strata::core::ExpertDispatch&d,const int32_t*ids,int64_t T,int64_t K,int64_t layer)noexcept{
 if(!p_)return;auto&p=*p_;int h=layer>=24?1:0;auto&b=p.bind[h];
 if(!p.rehomed||!p.window||layer<0||layer>=48||T<1||T>8||K!=10||!ids||!d.host_res||d.plan!=b.sink||b.borrowed){p.fail("unsupported original dispatch context");return;}
 b.d=&d;b.ids=ids;b.T=int(T);b.layer=int(layer);b.cpu=0;b.borrowed=true;b.published=false;
 std::array<bool,512>seen{};
 for(int i=0;i<T*10;++i){int ex=ids[i];if(ex<0||ex>=512){p.fail("routed expert ID invalid");return;}if(!seen[ex]){seen[ex]=true;if(d.host_res[size_t(layer)*512+ex]<0)++b.cpu;}}
}
void Bridge::after_dispatch(strata::core::ExpertDispatch&d)noexcept{
 if(!p_)return;auto&p=*p_;for(auto&b:p.bind)if(b.d==&d&&b.borrowed){if(!b.published||b.sink->cpu_jobs!=b.cpu)p.fail("post-dispatch original CPU jobs mismatch");b.borrowed=false;b.ids=nullptr;b.d=nullptr;return;}p.fail("unbound dispatch completion");
}
bool Bridge::finish_request(std::string&e){if(!p_)return true;auto&p=*p_;if(!p.request||p.finished||!p.rehomed||p.awaiting_prompt||p.identity_ready||p.failure||p.publications!=48*p.window){e=p.failure?p.failure:"V403 incomplete original publication count";return false;}for(int h=0;h<2;++h)for(int b=0;b<2;++b)if(p.adapt_pending[h][b]){e="unjoined adaptive physical producer";return false;}
 std::fprintf(stderr,"V403R4_ADAPT exact_copies=%llu foreign_copies=%llu bytes=%llu original_policy=1 original_home_event=1 no_weight_change=1\n",(unsigned long long)p.adapt_copies,(unsigned long long)p.adapt_foreign,(unsigned long long)p.adapt_bytes);
 p.finished=true;std::fprintf(stderr,"V403_MODEL_EP COMPLETE windows=%llu publications=%llu foreign_groups=%llu foreign_rows=%llu ordered_V350_original=1\n",(unsigned long long)p.window,(unsigned long long)p.publications,(unsigned long long)p.foreign_groups,(unsigned long long)p.foreign_rows);return true;}

bool prepare(const void*key,int T,int lb,int le,const std::vector<strata::kernels::NativeExpertLayout>&layouts,
 const uint8_t*q8,void*local_scratch,float*hits,std::string&e){
 if(!owns(key))return true;auto&p=*active;int h=p.home(key);auto&o=p.owner[h];
 if(T<1||T>8||lb!=h*24||le!=lb+24||layouts.size()!=24||!q8||!local_scratch||!hits){e="V403 capture geometry refused";return false;}
 if(o.prepared[T])return o.source==q8&&o.local_scratch==local_scratch&&o.hits==hits;
 o.source=q8;o.local_scratch=local_scratch;o.hits=hits;
 Device peer(1-h);
 for(int k=0;k<24;++k){const auto&L=layouts[k];if(L.n_embd!=2560||L.n_ff!=640||L.bytes!=p.homes[h].payload||L.had_block||L.d_signs){e="V403 original native layout refused";return false;}
  if(!ck(cudaStreamBeginCapture(o.stream,cudaStreamCaptureModeThreadLocal),e,"foreign begin capture"))return false;
  std::string why;bool ok=ck(ingest(o.alias[1-h],o.plan[1-h],1-h,T,lb+k,k+1,o.stream),why,"foreign plan")&&
   ck(copy_q8_if_needed(o.plan[1-h],q8,o.q8,T,o.stream),why,"single native peer Q8");
  if(ok){auto*plan=o.plan[1-h];strata::kernels::native_expert_grouped(L,plan->ptr,plan->start,plan->counts,plan->dst,plan->tok,T*10,T*10,o.q8,o.scratch,hits,o.stream,0);ok=ck(cudaPeekAtLastError(),why,"original peer native40");}
  if(ok)ok=ck(mark_completed(o.alias[1-h],1-h,k+1,o.stream),why,"foreign completion");
  cudaGraph_t graph=nullptr;auto rc=cudaStreamEndCapture(o.stream,&graph);
  if(!ok||rc!=cudaSuccess){if(graph)cudaGraphDestroy(graph);e=why.empty()?std::string("V403 foreign end capture: ")+cudaGetErrorString(rc):why;return false;}o.graph[T][k]=graph;
 }
 o.prepared[T]=true;std::fprintf(stderr,"V403_HELPER_GRAPHS owner=%d helper=%d T=%d layers=24 separate_nonblocking_context=1 native40_unchanged=1\n",h,1-h,T);return true;
}
bool fork_peer(const void*key,int layer,int T,cudaStream_t stream,std::string&e){
 if(!owns(key)){e="V405 fork requires exact registered EP owner";return false;}
 auto&p=*active;int h=p.home(key);auto&o=p.owner[h];int k=layer-h*24;
 if(T<1||T>8||k<0||k>=24||!o.prepared[T]){e="V405 foreign graph not prepared before input-ready fork";return false;}
 cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;const cudaGraphNode_t*deps=nullptr;
 const cudaGraphEdgeData*edge=nullptr;size_t n=0;unsigned long long id=0;
 if(!ck(cudaStreamGetCaptureInfo(stream,&status,&id,&graph,&deps,&edge,&n),e,"input-ready capture frontier")||status!=cudaStreamCaptureStatusActive||!graph){e="V405 input-ready fork must be captured";return false;}
 if(o.capture_id[T][k]==id&&o.child[T][k]){e="V405 duplicate foreign fork in same original capture";return false;}
 cudaGraphNode_t child=nullptr;
 if(!ck(cudaGraphAddChildGraphNode(&child,graph,deps,n,o.graph[T][k]),e,"early foreign graph node"))return false;
 // Do not add this child to the owner stream yet: original union wait-copy
 // and local native kernels proceed independently. Join at original post().
 o.parent[T][k]=graph;o.child[T][k]=child;o.capture_id[T][k]=id;o.bound[T][k]=o.local_bound[T][k]=false;return true;
}
bool enqueue(const void*key,int layer,int T,const strata::kernels::NativeExpertLayout&L,const uint8_t*q8,void*scratch,float*hits,cudaStream_t stream,std::string&e){
 if(!owns(key)){e="V403 unbound requested enqueue";return false;}auto&p=*active;int h=p.home(key);auto&o=p.owner[h];int k=layer-h*24;
 if(T<1||T>8||k<0||k>=24||!o.prepared[T]||o.source!=q8||o.local_scratch!=scratch||o.hits!=hits){e="V403 unprepared model endpoints";return false;}
 cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;const cudaGraphNode_t*deps=nullptr;const cudaGraphEdgeData*edge=nullptr;size_t n=0;unsigned long long id=0;
 if(!ck(cudaStreamGetCaptureInfo(stream,&status,&id,&graph,&deps,&edge,&n),e,"capture frontier")||status!=cudaStreamCaptureStatusActive||!graph){e="V403 owner must be captured";return false;}
 // Child dependencies are default/full completion+memory edges. ADD preserves
 // the local stream's tail; replacing it could lose the original native branch.
 cudaGraphNode_t child=o.child[T][k];
 if(!child||o.parent[T][k]!=graph||o.capture_id[T][k]!=id){e="V405 missing matching input-ready foreign child";return false;}
 auto*plan=o.plan[h];if(!ck(ingest(o.alias[h],plan,h,T,layer,k+1,stream),e,"local plan"))return false;
 strata::kernels::native_expert_grouped(L,plan->ptr,plan->start,plan->counts,plan->dst,plan->tok,T*10,T*10,q8,scratch,hits,stream,0);
 if(!ck(cudaPeekAtLastError(),e,"original local native40")||!ck(mark_completed(o.alias[h],h,k+1,stream),e,"local completion"))return false;
 o.local_bound[T][k]=true;return true;
}
bool join_peer(const void*key,int layer,int T,cudaStream_t stream,std::string&e){
 if(!owns(key)){e="V405 unregistered late join";return false;}
 auto&p=*active;int h=p.home(key);auto&o=p.owner[h];int k=layer-h*24;
 if(T<1||T>8||k<0||k>=24||!o.local_bound[T][k]||o.bound[T][k]){e="V405 missing local completion or duplicate join";return false;}
 cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;const cudaGraphNode_t*deps=nullptr;
 const cudaGraphEdgeData*edge=nullptr;size_t n=0;unsigned long long id=0;
 if(!ck(cudaStreamGetCaptureInfo(stream,&status,&id,&graph,&deps,&edge,&n),e,"late consumer join frontier")||status!=cudaStreamCaptureStatusActive||graph!=o.parent[T][k]||id!=o.capture_id[T][k]||!o.child[T][k]){e="V405 late join capture mismatch";return false;}
 // ADD retains the original local-native, CPU-ready and shared-expert frontier;
 // V350 cannot read either raw operand bank before its actual producer completes.
 cudaGraphEdgeData full{};auto child=o.child[T][k];
 if(!ck(cudaStreamUpdateCaptureDependencies(stream,&child,&full,1,cudaStreamAddCaptureDependencies),e,"positive late peer join before original V350"))return false;
 o.bound[T][k]=true;return true;
}
bool completed(const void*key,std::string&e){
 if(!owns(key))return true;auto&p=*active;int h=p.home(key);auto&f=*p.owner[h].host;
 if(p.failure||host_value(f.error)||host_value(f.status[0])||host_value(f.status[1])||host_value(f.completed[0])!=24||host_value(f.completed[1])!=24||host_value(f.abort)){e=p.failure?p.failure:"V403 joined model frame incomplete/refused";return false;}
 if(h==1){const auto&root=*p.owner[0].host;if(host_value(root.error)||host_value(root.status[0])||host_value(root.status[1])||host_value(root.completed[0])!=24||host_value(root.completed[1])!=24||host_value(root.abort)){e="V403 earlier owner refused before final logits consumption";return false;}}return true;
}
void release(const void*key)noexcept{if(owns(key))active->release_all();}
}
