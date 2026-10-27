#include "resident_gather_v5.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "peer_matrix.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
namespace resident_pp_v5 {
namespace M=strata::prefill::mmq;
namespace {
void check(cudaError_t e,const char*why){if(e!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e));}
[[noreturn]]void fatal(const char*why,cudaError_t e=cudaErrorUnknown)noexcept{std::fprintf(stderr,"PP5 unsafe retirement refused: %s: %s\n",why,cudaGetErrorString(e));std::fflush(stderr);std::_Exit(2);}
struct Restore{int previous=-1;Restore(){check(cudaGetDevice(&previous),"PP5 save owner");}~Restore(){if(previous>=0&&cudaSetDevice(previous)!=cudaSuccess)fatal("restore owning context");}};
void device_span(const void*p,uint64_t n,int owner){
 require(p&&n&&uintptr_t(p)%16==0&&n<=UINTPTR_MAX-uintptr_t(p),"PP5 invalid device span");cudaPointerAttributes a{};check(cudaPointerGetAttributes(&a,p),"PP5 group span attributes");require(a.type==cudaMemoryTypeDevice&&a.device==owner,"PP5 group buffers must remain original owner VRAM");
}
bool overlap(const void*a,uint64_t na,const void*b,uint64_t nb){return uintptr_t(a)<uintptr_t(b)+nb&&uintptr_t(b)<uintptr_t(a)+na;}
}
struct GatherOwner::Impl {
 int owner;cudaStream_t stream;AllocatedShape shape;four_banks_v3::PhysicalLease banks;GroupBuffers buffers;bool failed=false,pending=false;uint64_t epoch;std::mutex submission;
 Impl(int d,cudaStream_t s,AllocatedShape sh,four_banks_v3::PhysicalLease b,GroupBuffers bf):owner(d),stream(s),shape(sh),banks(std::move(b)),buffers(std::move(bf)),epoch(banks.epoch()){
  shape.validate();require(owner>=0&&owner<2&&stream&&buffers.owner,"PP5 two real dense owners/explicit stream/arena owner required");Restore restore;check(cudaSetDevice(owner),"PP5 construction owner");outside_capture();int sd=-1;check(cudaStreamGetDevice(stream,&sd),"PP5 original stream owner");require(sd==owner,"PP5 foreign original stream");
  // Root must ALSO call this before all physical-bank/PP allocation. This
  // idempotent recheck precedes every resource this adapter could create.
  std::string e;const bool peer_ok=strata_peer_default::require_all(4,e);require(peer_ok,e.c_str());
  require(buffers.gu_bytes>=gu_workspace_bytes()&&buffers.dn_bytes>=dn_workspace_bytes()&&buffers.tail_bytes>=tail_workspace_bytes(),"PP5 unchanged original MMQ group buffers too small");
  device_span(buffers.gu,buffers.gu_bytes,owner);device_span(buffers.dn,buffers.dn_bytes,owner);device_span(buffers.tail,buffers.tail_bytes,owner);
  require(!overlap(buffers.gu,buffers.gu_bytes,buffers.dn,buffers.dn_bytes)&&!overlap(buffers.gu,buffers.gu_bytes,buffers.tail,buffers.tail_bytes)&&!overlap(buffers.dn,buffers.dn_bytes,buffers.tail,buffers.tail_bytes),"PP5 group destinations overlap");
  banks.validate_live();for(int d0=0;d0<4;++d0){
   require(banks.base(d0)&&banks.bytes(d0)&&uintptr_t(banks.base(d0))%256==0,"PP5 physical bank address alignment");
   device_span(banks.base(d0),banks.bytes(d0),d0);
   require(!overlap(banks.base(d0),banks.bytes(d0),buffers.gu,buffers.gu_bytes)&&!overlap(banks.base(d0),banks.bytes(d0),buffers.dn,buffers.dn_bytes)&&!overlap(banks.base(d0),banks.bytes(d0),buffers.tail,buffers.tail_bytes),"PP5 group destinations overlap immutable physical bank");
  }
 }
 void outside_capture()const{cudaStreamCaptureStatus c=cudaStreamCaptureStatusNone;check(cudaStreamIsCapturing(stream,&c),"PP5 capture state");require(c==cudaStreamCaptureStatusNone,"PP5 original host-routed Prefill must remain outside capture");}
 void drain(){Restore restore;check(cudaSetDevice(owner),"PP5 drain owner");check(cudaStreamSynchronize(stream),"PP5 positive original-stream consumer credit");pending=false;}
 ~Impl(){try{drain();}catch(const std::exception&e){std::fprintf(stderr,"PP5 retirement first error: %s\n",e.what());fatal("uncertain group/bank consumer drain");}}
 void run(const Group&g,OriginalConsumers consume,void*user){
  require(!failed&&consume&&g.layer>=0&&g.layer<48&&g.layer/24==owner&&g.tokens>=1&&g.tokens<=shape.arena_rows&&g.allocated_rows==shape.arena_rows&&g.physical_epoch==epoch&&g.n>=1&&g.n<=16,"PP5 group/owner/epoch/capacity mismatch");
  int actual=-1;check(cudaGetDevice(&actual),"PP5 current issuing owner");require(actual==owner,"PP5 issuing context differs from original stream owner");outside_capture();banks.validate_live();
  auto refs=banks.host_refs();M::GatherGroup group;group.first=0;group.n=g.n;
  require(g.absolute[0]==g.row0&&g.relative[0]==0&&g.row0>=0&&g.nrows>0&&g.row0+g.nrows<=g.tokens*K&&g.absolute[size_t(g.n)]==g.row0+g.nrows,"PP5 original group row extent");int maximum=0;
  for(int i=0;i<g.n;++i){const auto&r=g.physical[size_t(i)];const int key=g.layer*NE+g.experts[size_t(i)];require(key>=g.layer*NE&&key<(g.layer+1)*NE,"PP5 key outside original layer");const auto&live=refs[size_t(key)];
   require(r.key==live.key&&r.device==live.device&&r.physical_slot==live.physical_slot&&r.byte_offset==live.byte_offset&&r.blob_bytes==live.blob_bytes&&r.physical_epoch==live.physical_epoch&&r.physical_epoch==epoch,"PP5 forged/stale physical reference");
   require(r.byte_offset<=banks.bytes(r.device)&&r.blob_bytes<=banks.bytes(r.device)-r.byte_offset,"PP5 payload outside live physical bank");
   require(g.absolute[size_t(i)+1]>g.absolute[size_t(i)]&&g.relative[size_t(i)+1]==g.absolute[size_t(i)+1]-g.row0,"PP5 original group bounds changed");maximum=std::max(maximum,g.absolute[size_t(i)+1]-g.absolute[size_t(i)]);
   uintptr_t base=uintptr_t(banks.base(r.device));require(r.byte_offset<=UINTPTR_MAX-base,"PP5 source address overflow");group.blob[i]=reinterpret_cast<const uint8_t*>(base+r.byte_offset);
  }
  require(maximum==g.max_rows,"PP5 original MMQ max_rows changed");
  pending=true; // before first uncertain enqueue; exceptions poison and drain
  require(M::gather_native_group(group,UpOffset,GuBytes/2,DownOffset,DnBytes,buffers.gu,GuBytes,buffers.dn,DnBytes,stream,TailOffset,buffers.tail,TailZero),"PP5 native aligned resident gather refused; no scalar/RAM retry");
  consume(g,buffers,stream,user);check(cudaGetLastError(),"PP5 original native consumers enqueue");
 }
};
GatherOwner::GatherOwner(int d,cudaStream_t s,AllocatedShape a,four_banks_v3::PhysicalLease l,GroupBuffers b):p_(std::make_unique<Impl>(d,s,a,std::move(l),std::move(b))){}
GatherOwner::~GatherOwner()=default;
bool GatherOwner::with_group(const Group&g,OriginalConsumers c,void*u,std::string&e)noexcept{
 std::unique_lock<std::mutex>lock(p_->submission,std::try_to_lock);
 if(!lock.owns_lock()){e="PP5 concurrent host submission refused";return false;}
 try{p_->run(g,c,u);return true;}catch(const std::exception&x){e=x.what();p_->failed=true;try{p_->drain();}catch(const std::exception&d){std::fprintf(stderr,"PP5 original error: %s\n",e.c_str());fatal(d.what());}return false;}
}
bool GatherOwner::finish(std::string&e)noexcept{std::unique_lock<std::mutex>lock(p_->submission,std::try_to_lock);if(!lock.owns_lock()){e="PP5 concurrent finish refused";return false;}try{p_->drain();require(!p_->failed,"PP5 poisoned owner cannot resume");return true;}catch(const std::exception&x){e=x.what();p_->failed=true;return false;}}
} // namespace resident_pp_v5
