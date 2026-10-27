#include "selector_v7.hpp"
#include "selector_v7_cuda.hpp"
#include "run_devices.hpp"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace swift_selector_v7 { namespace {
namespace L=swift_selector_v7_leaf;
constexpr unsigned char Canary=0xa7;
static_assert(BytesPerQuery==4ull*(L::Parts*L::Keep+L::Parts+2*L::Packed+4+Cap));
struct Slot {
 void*raw=nullptr;uint64_t bytes=0;cudaStream_t stream=nullptr;
 L::Workspace workspace{};bool guards_ready=false;
};
struct Pool {
 int device=-1,queries=0;unsigned used=0,active=0;
 bool retiring=false;std::atomic<bool>failed{false};
 std::array<Slot,MaxStreams>slots{};
 // Raw CUDA allocations have NO destructor. Only release performs device frees.
};
struct Registry {std::mutex mutex;std::map<const void*,std::unique_ptr<Pool>>owners;};
Registry&registry(){static Registry*r=new Registry;return *r;}
thread_local Pool*current=nullptr;
[[noreturn]]void fail(const char*s){throw std::runtime_error(std::string("Swift strict selector V7: ")+s);}
void check(cudaError_t e,const char*s){if(e!=cudaSuccess)fail(s);}
bool span(const void*p,uint64_t bytes,uintptr_t&lo,uintptr_t&hi){
 lo=reinterpret_cast<uintptr_t>(p);if(!p||(lo&3)||bytes>UINTPTR_MAX-lo)return false;hi=lo+bytes;return true;
}
bool overlap(uintptr_t a,uintptr_t b,uintptr_t c,uintptr_t d){return a<d&&c<b;}
void enqueue_check(Pool*p,cudaError_t e,const char*s){if(e!=cudaSuccess){p->failed.store(true);fail(s);}}
}

bool prepare(const void*owner,void*stream,int queries,std::string&err)noexcept{
 try{
  if(!owner||queries<1||queries>MaxQueries)fail("owner/query capacity out of range");
  int device=-1;check(cudaGetDevice(&device),"prepare current device");
  std::string hardware_error;
  if(!threegpu_scope::hardware(hardware_error)||device<0||device>=threegpu_scope::count())
   fail("fresh current-index CUDA order/SM89 capability required");
  auto cs=static_cast<cudaStream_t>(stream);cudaStreamCaptureStatus capture;
  int stream_device=-1;check(cudaStreamGetDevice(cs,&stream_device),"prepare stream device");
  if(stream_device!=device)fail("foreign-device stream");
  check(cudaStreamIsCapturing(cs,&capture),"prepare capture status");
  if(capture!=cudaStreamCaptureStatusNone)fail("prepare must precede capture");
  auto&r=registry();std::lock_guard<std::mutex>lock(r.mutex);
  auto it=r.owners.find(owner);
  if(it==r.owners.end()){
   auto p=std::make_unique<Pool>();p->device=device;p->queries=queries;
   it=r.owners.emplace(owner,std::move(p)).first;
  }
  Pool&p=*it->second;
  if(p.device!=device||p.queries!=queries||p.active||p.retiring||p.failed.load())
   fail("changed/active/retiring/failed owner");
  for(unsigned i=0;i<p.used;++i)if(p.slots[i].stream==cs)return true;
  if(p.used==MaxStreams)fail("eight-stream limit");
  const uint64_t bytes=BytesPerQuery*uint64_t(queries);
  size_t free=0,total=0;check(cudaMemGetInfo(&free,&total),"scratch budget query");
  if(free<bytes+2*GuardBytes+(64ull<<20))fail("scratch plus 64MiB reserve unavailable");
  Slot&s=p.slots[p.used];s.stream=cs;s.bytes=bytes;
  check(cudaMalloc(&s.raw,bytes+2*GuardBytes),"scratch allocation");
  // Publish allocation ownership immediately, so failed initialization is still
  // drainable/freeable through release. No partially initialized Scope is usable.
  ++p.used;
  try{
   check(cudaMemsetAsync(s.raw,Canary,GuardBytes,cs),"left scratch guard");
   check(cudaMemsetAsync(static_cast<unsigned char*>(s.raw)+GuardBytes+bytes,Canary,GuardBytes,cs),"right scratch guard");
   check(cudaStreamSynchronize(cs),"prepare positive guard completion");
   s.guards_ready=true;
   auto*d=reinterpret_cast<int32_t*>(static_cast<unsigned char*>(s.raw)+GuardBytes);
   auto take=[&](uint64_t n){auto*out=d;d+=n;return out;};
   s.workspace.local_ids=take(uint64_t(queries)*L::Parts*L::Keep);
   s.workspace.counts=take(uint64_t(queries)*L::Parts);
   s.workspace.mapping=take(uint64_t(queries)*L::Packed);
   s.workspace.packed_steps=take(uint64_t(queries)*4);
   s.workspace.packed_ids=take(uint64_t(queries)*Cap);
   s.workspace.packed_scores=reinterpret_cast<float*>(take(uint64_t(queries)*L::Packed));
   if(reinterpret_cast<unsigned char*>(d)!=static_cast<unsigned char*>(s.raw)+GuardBytes+bytes)
    fail("workspace extent mismatch");
  }catch(...){p.failed.store(true);throw;}
  return true;
 }catch(const std::exception&e){err=e.what();return false;}
 catch(...){err="V7 prepare unknown exception";return false;}
}

bool release(const void*owner,std::string&err)noexcept{
 try{
  auto&r=registry();Pool*p=nullptr;
  {std::lock_guard<std::mutex>lock(r.mutex);auto it=r.owners.find(owner);
   if(it==r.owners.end())return true;p=it->second.get();
   if(p->active||p->retiring)fail("release while active or already retiring");
   p->retiring=true;
  }
  int saved=-1;check(cudaGetDevice(&saved),"release current device");
  check(cudaSetDevice(p->device),"release owning device");
  try{
   // First drain ALL streams, before ANY buffer free. A failed CUDA drain leaves
   // the retiring owner registered and allocations retained; no unsafe reuse.
   for(unsigned i=0;i<p->used;++i)check(cudaStreamSynchronize(p->slots[i].stream),"release positive stream drain");
   std::array<unsigned char,GuardBytes>left{},right{};
   for(unsigned i=0;i<p->used;++i){Slot&s=p->slots[i];if(!s.raw)continue;
    if(s.guards_ready){
     check(cudaMemcpy(left.data(),s.raw,GuardBytes,cudaMemcpyDeviceToHost),"read left scratch guard");
     check(cudaMemcpy(right.data(),static_cast<unsigned char*>(s.raw)+GuardBytes+s.bytes,GuardBytes,cudaMemcpyDeviceToHost),"read right scratch guard");
     for(unsigned j=0;j<GuardBytes;++j)if(left[j]!=Canary||right[j]!=Canary)fail("scratch canary mismatch");
    }
   }
   for(unsigned i=0;i<p->used;++i){Slot&s=p->slots[i];if(s.raw){check(cudaFree(s.raw),"release scratch free");s.raw=nullptr;}}
  }catch(...){p->failed.store(true);if(saved!=p->device)cudaSetDevice(saved);throw;}
  {std::lock_guard<std::mutex>lock(r.mutex);r.owners.erase(owner);}
  if(saved>=0)check(cudaSetDevice(saved),"restore release device");
  return true;
 }catch(const std::exception&e){err=e.what();return false;}
 catch(...){err="V7 release unknown exception";return false;}
}
void release(const void*owner)noexcept{
 std::string err;if(!release(owner,err)){std::fprintf(stderr,"V7 fatal cleanup: %s\n",err.c_str());std::abort();}
}

Scope::Scope(const void*owner){
 auto&r=registry();std::lock_guard<std::mutex>lock(r.mutex);auto it=r.owners.find(owner);
 if(it==r.owners.end())fail("unprepared owner");Pool*p=it->second.get();
 int device=-1;check(cudaGetDevice(&device),"Scope current device");
 if(p->device!=device||p->active||p->retiring||p->failed.load()||!p->used)fail("foreign/reentrant/failed owner");
 previous_=current;current_=p;current=p;++p->active;
}
Scope::~Scope()noexcept{
 auto&r=registry();std::lock_guard<std::mutex>lock(r.mutex);auto*p=static_cast<Pool*>(current_);
 if(current!=p||!p||p->active!=1)std::abort();--p->active;current=static_cast<Pool*>(previous_);
}
void select(const float*scores,const int32_t*steps,int64_t q,int64_t blocks,int64_t cap,
 const strata::kernels::QsaShapes&shape,int32_t*ids,void*stream,int64_t active){
 Pool*p=current;
 if(!p||p->failed.load()||q<1||q>p->queries||blocks<1||blocks>MaxBlocks||cap!=Cap||
    shape.idx_block!=4||shape.idx_top_k!=2048||shape.idx_dim!=128||shape.idx_n_head!=4||active< -1||active>blocks)
  fail("unsupported exact shape/owner; no alternative selector");
 Slot*slot=nullptr;auto cs=static_cast<cudaStream_t>(stream);
 for(unsigned i=0;i<p->used;++i)if(p->slots[i].stream==cs)slot=&p->slots[i];
 if(!slot||!slot->guards_ready)fail("unknown/unprepared stream");
 uintptr_t sl,sh,tl,th,il,ih,wl,wh;
 if(!span(scores,uint64_t(q)*blocks*4,sl,sh)||!span(steps,uint64_t(q)*16,tl,th)||
    !span(ids,uint64_t(q)*Cap*4,il,ih)||!span(slot->raw,slot->bytes+2*GuardBytes,wl,wh)||
    overlap(sl,sh,tl,th)||overlap(il,ih,sl,sh)||overlap(il,ih,tl,th)||
    overlap(wl,wh,sl,sh)||overlap(wl,wh,tl,th)||overlap(wl,wh,il,ih))fail("unaligned/aliased/overflowed spans");
 auto&w=slot->workspace;
 enqueue_check(p,L::chunks(scores,steps,int(q),blocks,w,0,cs),"local candidate enqueue; no retry");
 enqueue_check(p,L::pack(scores,steps,int(q),blocks,w,cs),"candidate pack enqueue; no retry");
 enqueue_check(p,L::final_select(w.packed_scores,w.packed_steps,int(q),L::Packed,cap,w.packed_ids,cs),"compact final enqueue; no retry");
 enqueue_check(p,L::translate(steps,int(q),cap,w,ids,cs),"ID translation enqueue; no retry");
}
bool inspect(const void*owner,PoolInfo&out)noexcept{
 auto&r=registry();std::lock_guard<std::mutex>lock(r.mutex);out=PoolInfo{};
 auto it=r.owners.find(owner);if(it==r.owners.end())return false;const Pool&p=*it->second;
 out.device=p.device;out.queries=p.queries;out.slots=int(p.used);out.active=p.active;
 out.failed=p.failed.load();out.retiring=p.retiring;
 for(unsigned i=0;i<p.used;++i)out.payload_bytes+=p.slots[i].bytes;return true;
}
unsigned live_pools()noexcept{auto&r=registry();std::lock_guard<std::mutex>lock(r.mutex);return unsigned(r.owners.size());}
} // namespace
