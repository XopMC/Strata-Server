#pragma once
#include "../../whole-expert-resident-owner-v299/source/whole_owner_v299.hpp"
#include "policy_v370.hpp"
#include <array>
namespace ds370 {
using Descriptor=ds299::Descriptor;
struct Config370 {
 bool adaptive=false,component_only=false;
 //Productionliteral1078/1070; fixtureliteral16/8 with one real layer.
 int component_layer=0;size_t budget=20ull<<30;
 //Component-only deterministic delayedvisibility/fault. NEVERproductionflags.
 unsigned component_ready_delay_tasks=0;bool component_fail_after_submit=false;
};
struct Receipt370 {
 uint64_t version=370,struct_bytes=sizeof(Receipt370),adaptive=0,component_only=0;
 std::array<uint64_t,ds299::StatCount>whole{};
 uint64_t catalog=0,physical=0,active=0,spares=0,static_entries=0,pinned_bytes=0;
 uint64_t armed=0,completed_forwards=0,observer_calls=0,observer_failures=0;
 uint64_t service_calls=0,service_refusals=0,service_wall_ns=0,service_thread_ns=0,planner_wall_ns=0,planner_thread_ns=0,planner_group_visits=0,planner_limited=0;
 uint64_t stage_wall_ns=0,stage_thread_ns=0,started=0,upload_completed=0,published=0,nonready_fallbacks=0,protected_ready=0;
 uint64_t H2D_bytes=0,inverse_read_bytes=0,inverse_write_bytes=0,upload_event_ns=0;
 uint64_t pending=0,stages_inflight=0,close_cancelled=0,positive_drains=0,closed=0,failed=0;
};
struct PolicyHooks370 {
 uint64_t struct_bytes=sizeof(PolicyHooks370),version=370;void*owner=nullptr;
 void(*retain)(void*)noexcept=nullptr;void(*release)(void*)noexcept=nullptr;
 int(*before)(void*,uintptr_t,unsigned,const uint8_t*,unsigned)noexcept=nullptr;
 int(*after)(void*,int)noexcept=nullptr;
};
class Owner final:public std::enable_shared_from_this<Owner>{
public:
 static std::shared_ptr<Owner>create(std::shared_ptr<const strata::ds41::BorrowedWeights>,std::vector<Descriptor>,const Ds276ExpTable&,Config370);
 ~Owner()noexcept;Owner(const Owner&)=delete;Owner&operator=(const Owner&)=delete;
 const DsWholeHooksV299&hooks()const noexcept;
 const PolicyHooks370&policy_hooks()const noexcept;
 int arm(uint64_t expected_generation,uint64_t expected_format_epoch)noexcept;
 //Existingmain255 only, ordinary idle AFTERpriorverify/commit. NoGPUhostwait.
 int gap_service(uint64_t generation,uint64_t format_epoch,uint64_t control_nonce,unsigned max_starts)noexcept;
 Receipt370 receipt()const;
 int close()noexcept;
private:
 struct Impl;std::unique_ptr<Impl>p_;
 Owner(std::shared_ptr<const strata::ds41::BorrowedWeights>,std::vector<Descriptor>,const Ds276ExpTable&,Config370);
 static void retain(void*)noexcept;static void release(void*)noexcept;
 static int contains(void*,int,uintptr_t,uintptr_t,uintptr_t)noexcept;
 static int begin(void*,const DsWholeCommandV299*,void**)noexcept;
 static int finish(void*,void*,uint64_t,const uint16_t**,uint64_t)noexcept;
 static int release_view(void*,void*,uint64_t)noexcept;static int abort(void*,void*,uint64_t)noexcept;
 static int before(void*,uintptr_t,unsigned,const uint8_t*,unsigned)noexcept;
 static int after(void*,int)noexcept;
};
}
