#include "native_adaptive_lease_v387.hpp"
#include <map>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <exception>
#include <sched.h>
namespace ds387 {
namespace {
std::mutex registry_mutex;
std::map<const strata::ds41::NativeExperts*,std::weak_ptr<Lease>>registry;
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
bool main255(){cpu_set_t a;CPU_ZERO(&a);return !sched_getaffinity(0,sizeof(a),&a)&&CPU_COUNT(&a)==1&&CPU_ISSET(255,&a);}
bool flag(const char*k){const char*v=std::getenv(k);need(!v||!std::strcmp(v,"0")||!std::strcmp(v,"1"),"387 requires literal0/1 flag");return v&&!std::strcmp(v,"1");}
void remove(const strata::ds41::NativeExperts*id,const Lease*lease){
 std::lock_guard<std::mutex>g(registry_mutex);auto it=registry.find(id);
 need(it!=registry.end(),"387 missing registered exact NativeExperts lease");
 auto p=it->second.lock();need(p&&p.get()==lease,"387 lease identity changed");registry.erase(it);
}
}
static_assert(sizeof(ds370::Receipt370)==672&&std::is_trivially_copyable_v<ds370::Receipt370>);
bool enabled(){return flag("DS1017_WHOLE_ENGINE_V387");}
bool adaptive_mode(){const char*v=std::getenv("DS1017_WHOLE_ADAPTIVE_V370");need(v&&(!std::strcmp(v,"0")||!std::strcmp(v,"1")),"387 explicit370 OFF/ON required");return !std::strcmp(v,"1");}
Lease::Lease(strata::ds41::NativeExperts&id,std::shared_ptr<ds370::Owner>owner,std::function<void()>detach)
 :identity_(&id),owner_(std::move(owner)),detach_(std::move(detach)){
 need(owner_&&bool(detach_),"387 real owner/detach closure required");const auto r=owner_->receipt();
 generation_=r.whole[ds299::Generation];format_=r.whole[ds299::FormatEpoch];need(generation_&&format_,"387 actual owner generation/format required");
}
Lease::~Lease()noexcept{if(registered_&&!closed_)std::terminate();}
void Lease::mark_registered(){registered_=true;}
bool Lease::closed()const{std::lock_guard<std::mutex>g(mutex_);return closed_;}
void Lease::arm(uint64_t gen,uint64_t fmt){std::lock_guard<std::mutex>g(mutex_);need(registered_&&!closed_&&!armed_&&gen==generation_&&fmt==format_,"387 arm actual registered binding once");need(!owner_->arm(gen,fmt),"387 owner arm rejected");armed_=true;}
unsigned Lease::gap(uint64_t gen,uint64_t fmt,uint64_t nonce,unsigned max){
 std::lock_guard<std::mutex>g(mutex_);need(main255()&&registered_&&!closed_&&armed_&&gen==generation_&&fmt==format_&&nonce>nonce_&&max==2,"387 gap binding/arm/monotonicnonce/cap2");
 const auto r=owner_->receipt();need(r.armed&&!r.closed&&!r.failed&&!r.whole[ds299::LiveTickets]&&!r.whole[ds299::Borrowed],"387 gap requires actual idle clean owner");
 // The first generation iteration has no completed40-layer target forward.
 // WholeOFF has no observer/policy history. Neither case submits any upload.
 if(!r.completed_forwards){nonce_=nonce;return 1;}
 need(r.completed_forwards>last_serviced_,"387 duplicate service for same completed forward");
 const int rc=owner_->gap_service(gen,fmt,nonce,max);need(rc==0,"387 owner gap service refused/failed");
 nonce_=nonce;last_serviced_=r.completed_forwards;return 0;
}
ds370::Receipt370 Lease::receipt()const{std::lock_guard<std::mutex>g(mutex_);return owner_->receipt();}
void Lease::close(const strata::ds41::NativeExperts&id){
 std::lock_guard<std::mutex>g(mutex_);need(&id==identity_,"387 exact typed NativeExperts required for close");if(closed_)return;
 need(registered_&&bool(detach_),"387 close requires registered live callbacks");
 // TaskCABI checks running/queued/borrowed state before detaching. NativeExperts
 // mutex is held by this closure; owner close positively drains BOTH streams.
 detach_();need(owner_->close()==0,"387 adaptive close drain failed");
 const auto r=owner_->receipt();need(r.closed&&!r.pending&&!r.stages_inflight&&!r.whole[ds299::LiveTickets]&&!r.whole[ds299::Borrowed],"387 actual drained terminal receipt required");
 remove(identity_,this);registered_=false;closed_=true;detach_={}; // no dangling Impl capture after unregister
}
std::shared_ptr<Lease>install(strata::ds41::NativeExperts&id,std::shared_ptr<ds370::Owner>owner,std::function<void()>detach){
 auto lease=std::make_shared<Lease>(id,std::move(owner),std::move(detach));std::lock_guard<std::mutex>g(registry_mutex);
 for(auto it=registry.begin();it!=registry.end();)if(it->second.expired())it=registry.erase(it);else ++it;
 need(registry.empty(),"387 permits one live actual NativeExperts registration per child");
 need(registry.emplace(&id,lease).second,"387 duplicate actual NativeExperts");lease->mark_registered();return lease;
}
std::shared_ptr<Lease>lookup(const strata::ds41::NativeExperts&id){std::lock_guard<std::mutex>g(registry_mutex);auto it=registry.find(&id);if(it==registry.end())return{};auto p=it->second.lock();need(bool(p),"387 expired registered lease violates lifetime");return p;}
void cleanup(const strata::ds41::NativeExperts*id)noexcept{if(!id)return;try{if(auto p=lookup(*id))p->close(*id);}catch(...){std::terminate();}}
}
