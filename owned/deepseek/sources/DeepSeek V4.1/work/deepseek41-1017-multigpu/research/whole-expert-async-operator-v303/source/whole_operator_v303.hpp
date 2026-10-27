#pragma once
#include <cuda_runtime_api.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "../../cpu-pair32-static-v143/source/native_a8_prequant_views_v118.hpp"
#include "../../bf16-gate-libmvec-table-v276/source/exp_table_v276.h"
namespace ds303 {
constexpr int Dim=5120,Hidden=2304,TopK=6,MaxRows=6,MaxRoutes=36,MaxGroups=36,Chunks=3;
struct Group {
 uint32_t slot=0,expert=0,count=0,reserved=0;
 uint32_t query[6]{},route_slot[6]{};float alpha[6]{};
};
static_assert(sizeof(Group)==88);
struct Command {
 uint64_t epoch=0;uint32_t rows=0,groups=0;
 Ds41A8ViewV118 view[6]{};Group group[MaxGroups]{};
};
struct Config {
 int logical_device=-1;uint32_t resident_slots=0;
 const uint8_t*const*gu_table=nullptr;const uint8_t*const*down_table=nullptr;
 const Ds276ExpTable*cpu_table=nullptr;
 cudaStream_t stream=nullptr; // caller-owned, MUST NONBLOCKING/current rank
 std::shared_ptr<const void> device_owner; // resources token, NOT Operator/Owner cycle
};
struct Borrowed {const uint16_t*data=nullptr;uint64_t words=0,epoch=0;uint32_t rows=0;};
enum Stat:size_t {
 Version,Device,Slots,LiveDeviceBytes,PeakDeviceBytes,LivePinnedBytes,PeakPinnedBytes,
 Active,BorrowedView,Poisoned,Closed,Begins,Groups,Routes,GuChunks,DownChunks,
 CompletedEvents,Finishes,Releases,Aborts,Failures,LastEpoch,ActiveEpoch,IssuerTid,
 Rows,InputCopyBytes,OutputCopyBytes,TablesValidated,GuVersion,HiddenVersion,DownVersion,Reserved,StatCount
};
static_assert(StatCount==32);
class Operator {
 struct Impl;std::unique_ptr<Impl>p_;
 explicit Operator(std::unique_ptr<Impl>);
public:
 /* Pure ledger, no CUDA or allocation. Original requests are65920/176896B;
    CPP asserts these literal ABI sizes and private Plan736B before build.
    Excludes caller's bank/tables/stream and separately-budgeted141 inverse.
    Includes immutable320KiB table and EVERY owned device/pinned allocation. */
 static constexpr size_t workspace_device_bytes() noexcept {
  return 65536*5+(65920+176896)*Chunks+736+
   2*MaxRoutes*Hidden*4+MaxRoutes*Dim*4+3*MaxRoutes*Hidden*4+
   MaxRoutes*Hidden*2+MaxRoutes*Hidden+MaxRoutes*(Hidden/32)*4+
   MaxRoutes*(Hidden/4)*4+MaxRows*TopK*Dim*2+8;
 }
 static constexpr size_t workspace_pinned_bytes() noexcept {
  return (65920+176896)*Chunks+736+MaxRows*TopK*Dim*2+4;
 }
 static std::unique_ptr<Operator> create(Config);
 ~Operator();Operator(const Operator&)=delete;Operator&operator=(const Operator&)=delete;
 /* Same issuer within a ticket; idle issuer may rebind. groups1..36, total
    routes<=36, Count1..6. Strict expert/query order, unique physicalslots and
    originalroute slots; positivealpha only, route_slot/6==query. All guards and
    host snapshots BEFORE enqueue. No allocations/defaultstream/device sync.
    After partial enqueue failure: drain/poison, never retry/fallback. */
 void begin(const Command&);
 Borrowed finish(uint64_t epoch); // event wait, immutable pinned view remains active
 void release(uint64_t epoch);   // ONLY after synchronous caller copy/CPU join
 void abort(uint64_t epoch);     // drain active ticket on same issuer
 int shutdown() noexcept;        // idle ONLY (release/abort first); own-stream drain/free
 // Failed/unproved close retains resources and is fatal at destruction. No
 // active borrow is revoked or silently freed; no device-wide sync/reset.
 std::array<uint64_t,StatCount>stats()const noexcept;
};
}
