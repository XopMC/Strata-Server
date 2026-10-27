#pragma once
#include "../gpu-gu-borrowed-post-v18-r3/gu_hooks_v18.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ds1017_gu_resident_owner_v19 {
struct Descriptor {
 int layer=0,expert=0;
 uintptr_t gate_key=0,up_key=0;
 const uint8_t*gate=nullptr,*up=nullptr;
 size_t gate_bytes=0,up_bytes=0;
 int rows=2304,columns=5120,bits=0;
 uint64_t frequency=0;
};
// Raw exact counters. No time/TPS or whole-FFN hit claims.
enum Stat : size_t {
 Version,ResidentEntries,RequestedDeviceBytes,WeightDeviceBytes,UploadH2DBytes,
 ContainsHits,ContainsMisses,ContainsInvalid,BeginCalls,SubmittedCommands,
 ReleasedViews,AbortedCommands,Failures,LiveTickets,BorrowedViews,
 CommandH2DBytes,ResultD2HBytes,SubmittedGroups,SubmittedRoutePairs,
 Count1Groups,Count2Groups,Count6Groups,IssuerRebinds,Closed,
 BudgetBytes,StagingPinnedBytes,RequestPinnedBytes,ResultPinnedBytes,
 ScratchSentinelBytes,UploadCopies,VRAMFreeBefore,VRAMFreeAfter,StatCount
};
class Owner final : public std::enable_shared_from_this<Owner> {
public:
 // Validate all descriptors, select stable highest-frequency ranks that fit the
 // TOTAL device budget (ties layer/expert), then upload only selected pairs.
 // Root supplies same BorrowedWeights retention token; no second CPU bank.
 // Requires exact admitted pair, explicit owner/V13 opt-ins, ordinary uncaptured
 // startup, GPU1 free bytes >= exact allocations +512MiB reserve, budget<=22GiB.
 static std::shared_ptr<Owner> create(std::vector<Descriptor>,size_t budget,
     std::shared_ptr<const void> retained_weights);
 ~Owner() noexcept;
 Owner(const Owner&)=delete;Owner&operator=(const Owner&)=delete;
 // Stable authoritative R3 ABI. attach invokes retain; detach invokes release.
 // No extra initial hook ref is transferred by this accessor. Root may keep its
 // shared_ptr too. Hooks retain this Owner until detached; active tickets require
 // serialized N1 lifecycle and finish/release or abort before detach/close.
 const DsGuHooksV18& hooks() const noexcept;
 std::array<uint64_t,StatCount> stats() const noexcept;
private:
 struct Impl;std::unique_ptr<Impl> impl_;
 explicit Owner(std::vector<Descriptor>,size_t,std::shared_ptr<const void>);
 static void retain_hook(void*) noexcept;static void release_hook(void*) noexcept;
 static int contains_hook(void*,int,uintptr_t,uintptr_t) noexcept;
 static int begin_hook(void*,const DsGuCommandV18*,void**) noexcept;
 static int finish_hook(void*,void*,uint64_t,const float**,uint64_t) noexcept;
 static int release_view_hook(void*,void*,uint64_t) noexcept;
 static int abort_hook(void*,void*,uint64_t) noexcept;
};
}
