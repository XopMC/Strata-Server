#pragma once
#include "../four-gpu-physical-banks-v3/physical_banks_v3.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <functional>
#include <memory>
#include <string>
namespace four_loader_v9 {
namespace B=four_banks_v3;
enum class HostKind {Pinned,Pageable};
struct SourceSpan {const unsigned char*data=nullptr;uint64_t bytes=0;HostKind kind=HostKind::Pageable;};
struct Source {
 std::shared_ptr<const void>immutable_arena_owner;
 std::function<SourceSpan(int global_key)>get;
 uint64_t source_epoch=0;
 bool one_immutable_arena=false;
};
// Values are REMAINING allocations/peak workspace not already subtracted by
// cudaMemGetInfo. Already-allocated dense/head/session bytes are not re-added.
// Every category is explicit; zero means caller deliberately accounts it as
// already allocated/not used. This is NOT a complete model fit assertion.
struct FutureNeed {
 uint64_t dense=0,prefill_peak=0,head=0,session=0,controller_other=0,safety_reserve=0;
 bool declared=false;
};
struct Options {
 std::array<FutureNeed,4>future;
 uint64_t physical_epoch=0;
 unsigned staging_slots_per_device=2; //1..2, lazy only for pageable spans
 uint64_t pinned_staging_limit=32ull<<20;
};
struct BudgetReceipt {
 uint64_t free_before=0,total=0,owned_request=0,future_remaining=0;
 FutureNeed future;
};
struct Receipt {
 std::array<BudgetReceipt,4>budget;
 std::array<uint64_t,4>payload_bytes{},pinned_direct_bytes{},bounced_bytes{};
 uint64_t source_epoch=0,physical_epoch=0,peak_pinned_staging=0;
 uint32_t keys=0;
 bool four_uploads_complete=false,outer_guards_identical=false,metadata_readback=false;
};
// Requires caller's fresh current0/1/2/3 visibility snapshot (shared policy).
// Does not select UUIDs/devices or change driver/system state.
void admit_current_four();
class BankOwner {
 struct Session;std::shared_ptr<Session>s_;
 explicit BankOwner(std::shared_ptr<Session>);
public:
 BankOwner(const BankOwner&)=delete;BankOwner&operator=(const BankOwner&)=delete;
 ~BankOwner();
 static std::unique_ptr<BankOwner>load(const Source&,const Options&);
 B::PhysicalLease lease()const;
 B::PhysicalDeviceView view(int current_device)const;
 std::shared_ptr<const void>retain()const; // retains banks AND refs allocations
 const Receipt&receipt()const;
 // Caller destroys/drains original consumer graphs and releases all external
 // leases/retain tokens FIRST. No global/device sync or borrowed-stream use.
 void close(bool all_external_consumers_positively_drained);
};
#ifdef STRATA_BANK_LOADER_V9_TESTING
namespace testing {
// SAME allocate/upload/guard engine with 2 synthetic keys per device. No V3
// full-coverage lease/view is exposed; this tests transport, not full capacity.
void exercise_mid_upload_failure(const Source&,const Options&);
void exercise_small(const Source&,const Options&,const std::function<void(int,int,const unsigned char*,uint64_t,cudaStream_t)>&check_device);
}
#endif
}
