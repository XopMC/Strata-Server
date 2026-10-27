#pragma once
// Original PUBLIC Owner19/Descriptor definitions are included, never redeclared.
#include "../gpu-gu-resident-owner-v19/resident_owner_v19.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace ds1017_gu_paired_owner_v147 {
using Owner=ds1017_gu_resident_owner_v19::Owner;
using Descriptor=ds1017_gu_resident_owner_v19::Descriptor;
struct Sidecar147 {
 int layer=0,expert=0;
 uintptr_t gate_key=0,up_key=0;
 uint32_t layout_tag=139,reserved=0;
 uint64_t owner_generation=0,bank_format_epoch=0;
};
class UploadStats147;
enum Stat147:size_t {
 Version147,Layout147,OwnerGeneration147,FormatEpoch147,ValidatedDescriptors147,
 ResidentEntries147,CacheBudget147,StartupScratchBudget147,DeclaredPeakBudget147,
 FinalRequestedDeviceBytes147,PeakRequestedDeviceBytes147,StartupScratchPeak147,
 StartupScratchLive147,InverseSubmitted147,InverseCompleted147,InverseInputBytes147,
 InverseOutputBytes147,PairedH2DBytes147,ValidatedFooterWords147,InitComplete147,
 Closed147,Failures147,LastInverseStatus147,OriginalStatWords147,StatCount147
};
struct Created147 {
 std::shared_ptr<Owner> owner;
 std::shared_ptr<const UploadStats147> upload_stats;
 size_t declared_cache_budget=0,declared_startup_scratch_budget=0;
};
/* Explicit typed factory only. Sidecars SAMEorderasdescriptors, each exact keys,
   layer/expert,tag139,positive common owner_generation+bank_format_epoch.
   Original readonly paired buffers and real BorrowedWeights sharedtoken retained
   until Owner drains/closes. No TLS/global next-owner registry/rawtoken casts.
   Optins DS1017_GPU_GU_PAIR32_OWNER_V147=139 and141inverse=1 are mandatory.
   cache_budget is original19 FINALcachebudget (same membership/capacity).
   startup_scratch_budget MUST explicitly equal6635520; declaredpeak=cache+scratch
   <=22GiB. Actualrequestedpeak=original19allocations+scratch; free guard includes
   that peak+512MiB. Never claim peak<=old cachebudget. Extra scratch freedafterinit.
   In private147 DSO plain/untyped Owner::create failsclosed before CUDA. */
Created147 create_paired147(std::vector<Descriptor>descriptors,
 std::vector<Sidecar147>sidecar,size_t cache_budget,size_t startup_scratch_budget,
 std::shared_ptr<const void>retained_weights);
/* Separate24wordreceipt; originalOwner19 32wordenum/ABI is byte-identical.
   Counts are actual submissions/completions; requestedbytes are allocator request
   bytes, NOT a claim about driver-rounded VRAM use. No TG/TPS claim. */
std::array<uint64_t,StatCount147> stats147(const std::shared_ptr<const UploadStats147>&)noexcept;
}
