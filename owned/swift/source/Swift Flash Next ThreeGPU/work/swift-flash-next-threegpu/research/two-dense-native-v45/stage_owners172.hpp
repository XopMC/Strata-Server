#include "run_devices.hpp"
#pragma once
// Private MAIN/component ownership; no public numerical class/ABI change. No early/manual stage teardown contract.
#include "strata/core/session.hpp"
#include "strata/core/pin_budget.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace stage_owners172 {
inline bool enabled(){static const bool on=[] {const char* p=std::getenv("STRATA_STAGE_ARENA_OWNERS_V172");return p&&std::strcmp(p,"1")==0;}();return on;}
enum class Event { DrainBeforeMembers, ReleaseRope, FreeHostStep, FreeHostPos, DeleteQsaArray, FreeSession, FreeCanonical, RestoreCaller };
using Audit=void(*)(Event,const void*,void*) noexcept;
[[noreturn]]inline void fatal(const char* what,cudaError_t e=cudaErrorInvalidValue)noexcept{
 std::fprintf(stderr,"V172 stage owner %s: %s; unsafe cleanup refused\n",what,cudaGetErrorString(e));std::fflush(stderr);std::_Exit(1);
}
// Declare FIRST in an owning stage: dies LAST, after Prefill/Verifier/cache/head/dense/table members.
class Memory {
 int device_=-1,previous_=-1;bool prepared_=false;
 void* canonical_=nullptr;size_t canonical_bytes_=0;void* session_=nullptr;size_t session_bytes_=0;
 strata::core::SessionState release_view_{};
 Audit audit_=nullptr;void* audit_user_=nullptr;
 void note(Event e,const void* p=nullptr)noexcept{if(audit_)audit_(e,p,audit_user_);}
 void free_host(int32_t*& p,Event e)noexcept{
  if(!p)return;note(e,p);
  // host_alloc_guarded has no public matching free wrapper. Use its pinned, source-pinned
  // accounting under the same mutex; uncharge ONLY after a successful actual free.
  auto& budget=strata::core::pin_detail::st();std::lock_guard<std::mutex> lock(budget.mu);
  const auto ce=cudaFreeHost(p);if(ce!=cudaSuccess)fatal("host staging free",ce);
  strata::core::pin_detail::sim_uncharge(budget,p);p=nullptr;
 }
public:
 Memory()=default;Memory(const Memory&)=delete;Memory&operator=(const Memory&)=delete;
 void configure(int device,Audit audit=nullptr,void* user=nullptr)noexcept{
  if(device_<0){if(device<0||device>=threegpu_scope::count())fatal("device outside current run");device_=device;audit_=audit;audit_user_=user;}
  else if(device_!=device)fatal("owner device changed");
 }
 bool configured()const noexcept{return device_>=0;}
 size_t canonical_bytes()const noexcept{return canonical_bytes_;}
 size_t session_bytes()const noexcept{return session_bytes_;}
 void canonical(void* p,size_t n)noexcept{if(!configured()||canonical_||!p||!n)fatal("invalid canonical adoption");canonical_=p;canonical_bytes_=n;}
 void session(void* p,size_t n)noexcept{if(!configured()||session_||!p||!n)fatal("invalid session adoption");session_=p;session_bytes_=n;}
 void capture_session(const strata::core::SessionState& s)noexcept{
  if(!configured()||!session_)return;
  if(s.qsa_states&&(s.qsa_ord0<0||s.qsa_alloc<0||s.qsa_ord0+s.qsa_alloc>12))fatal("invalid global QSA ownership range");
  if(release_view_.qsa_states&&release_view_.qsa_states!=s.qsa_states)fatal("session metadata owner replaced");
  // Snapshot only ownership fields. Do not retain a pointer to a later-destroyed SessionState member.
  release_view_.qsa_states=s.qsa_states;release_view_.qsa_ord0=s.qsa_ord0;release_view_.qsa_alloc=s.qsa_alloc;
 }
 void before_members()noexcept{
  if(!configured()||prepared_)return;
  auto e=cudaGetDevice(&previous_);if(e!=cudaSuccess)fatal("get previous device",e);
  e=cudaSetDevice(device_);if(e!=cudaSuccess)fatal("select stage for destruction",e);
  e=cudaDeviceSynchronize();if(e!=cudaSuccess)fatal("drain before member destruction",e);
  prepared_=true;note(Event::DrainBeforeMembers);
  // Intentionally leave stage device current until Memory dies AFTER all dependent members.
 }
 ~Memory()noexcept{
  if(!configured())return;if(!prepared_)before_members();
  auto e=cudaSetDevice(device_);if(e!=cudaSuccess)fatal("select stage for final arena free",e);
  // Member destructors have finished. Drain any work they themselves issued, too.
  e=cudaDeviceSynchronize();if(e!=cudaSuccess)fatal("drain after member destruction",e);
  // No graph/Prefill/MTP may still read session or weights.
  if(release_view_.qsa_states){
   strata::core::session_release(release_view_);note(Event::ReleaseRope);
   for(int64_t j=0;j<release_view_.qsa_alloc;++j){auto& q=release_view_.qsa_states[release_view_.qsa_ord0+j];
    free_host(q.host_step,Event::FreeHostStep);free_host(q.host_pos,Event::FreeHostPos);}
   note(Event::DeleteQsaArray,release_view_.qsa_states);delete[] release_view_.qsa_states;release_view_.qsa_states=nullptr;
  }
  // Process-lifetime authoritative QSA host KV is NOT owned/freed here; fields are mapped aliases.
  if(session_){note(Event::FreeSession,session_);e=cudaFree(session_);if(e!=cudaSuccess)fatal("session arena free",e);session_=nullptr;}
  if(canonical_){note(Event::FreeCanonical,canonical_);e=cudaFree(canonical_);if(e!=cudaSuccess)fatal("canonical arena free",e);canonical_=nullptr;}
  e=cudaSetDevice(previous_);if(e!=cudaSuccess)fatal("restore caller device",e);note(Event::RestoreCaller);
 }
};
// Declare LAST in the stage: dies FIRST. A destructor BODY would run before member destructors too,
// but must not free resources; this guard separates drain/device selection from last-owner frees.
struct DrainFirst {Memory& memory;~DrainFirst()noexcept{memory.before_members();}};
// Captures even a session_init exception/zero return after partial global QSA-array initialization.
struct CaptureSession {Memory& memory;strata::core::SessionState& state;~CaptureSession()noexcept{memory.capture_session(state);}};
}
