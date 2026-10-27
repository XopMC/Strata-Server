#pragma once
#include "whole_hooks_v299.hpp"
#include "../../bf16-gate-libmvec-table-v276/source/exp_table_v276.h"
#include "strata/ds41/borrowed_weights.hpp"
#include <array>
#include <memory>
#include <vector>
namespace ds299 {
struct Descriptor {int layer,expert;uint64_t frequency;};
struct Budgets {
 // Requestedpayload caps PERrank, including303workspace+tables+weights
 // AND separate6635520 startup inverse scratch. Driverreserve is additional.
 // GPU0<=2GiB andGPU1<=20GiB; combined<=22GiB afterfreshdualGPU instruction.
 size_t gpu0=0,gpu1=20ull<<30;
 // GPU0 allocations are refused without explicit external proof gate. The
 // separate bounded doorbell/concurrency and all-target-graph memory receipt
 // must be bound by a future Root adapter before setting this field true.
 bool gpu0_target_resources_proved=false;
};
enum Stat:size_t {Version,Layout,Generation,FormatEpoch,Entries0,Entries1,
 FinalBytes0,FinalBytes1,DeclaredPeak0,DeclaredPeak1,RequestedPeak0,RequestedPeak1,FreeAfterWorkspace0,FreeAfterWorkspace1,WeightBytes0,WeightBytes1,
 Begin,Submitted,Finished,Released,Aborted,Failed,LiveTickets,Borrowed,
 Groups0,Groups1,Routes0,Routes1,H2D0,H2D1,D2H0,D2H1,IssuerRebinds,
 InverseCalls,InverseBytes,StartupScratchLive,StartupJobs0,StartupJobs1,Completed0,Completed1,GuChunks0,GuChunks1,DownChunks0,DownChunks1,Closed,StatCount};
class Owner final:public std::enable_shared_from_this<Owner>{
public:
 // Actual typed145 projection lookup on this SAME BorrowedWeights. No raw void
 // ownership token/no registry injection/no caller-substituted descriptor bytes.
 // Immutable global frequency selection; ties layer/expert. Disjoint residence.
 static std::shared_ptr<Owner>create(std::shared_ptr<const strata::ds41::BorrowedWeights>,
   std::vector<Descriptor>,const Ds276ExpTable&,Budgets);
 ~Owner()noexcept;
 Owner(const Owner&)=delete;Owner&operator=(const Owner&)=delete;
 const DsWholeHooksV299&hooks()const noexcept;
 std::array<uint64_t,StatCount>stats()const noexcept;
private:
 struct Impl;std::unique_ptr<Impl>p_;
 Owner(std::shared_ptr<const strata::ds41::BorrowedWeights>,std::vector<Descriptor>,const Ds276ExpTable&,Budgets);
 static void retain(void*)noexcept;static void release(void*)noexcept;
 static int contains(void*,int,uintptr_t,uintptr_t,uintptr_t)noexcept;
 static int begin(void*,const DsWholeCommandV299*,void**)noexcept;
 static int finish(void*,void*,uint64_t,const uint16_t**,uint64_t)noexcept;
 static int release_view(void*,void*,uint64_t)noexcept;
 static int abort(void*,void*,uint64_t)noexcept;
};
}
