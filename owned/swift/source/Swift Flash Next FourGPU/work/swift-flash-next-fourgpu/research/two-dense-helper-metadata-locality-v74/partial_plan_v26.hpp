#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace swift_ep_v26 {
inline constexpr int Layers=48, Experts=512, K=10, MaxT=8, Cap=80;
enum Error : uint32_t {
    BadShape=1u, BadLayer=2u, BadId=4u, BadSlot=8u,
    BadOffset=16u, BadEpoch=32u, BadDestination=64u
};
// Device-resident, stable address. Producer may update its CONTENTS between
// fully drained frames; a captured planner reads the current contents.
struct alignas(16) Request {
    int32_t T=1, k=K, layer=0, row_base=0;
    uint32_t epoch=1, reserved[3]{};
};
// Nonowning, single physical device. All metadata must be immutable during
// consumption. offsets has slots+1 live byte offsets; local has48*512 slots/-1.
// min_blob_bytes is actual ExpertLayout::blob_bytes(layer), not slot stride.
// Caller validates unique physical ownership, cache epoch and stream/device.
struct CacheView {
    const unsigned char* base=nullptr;
    uint64_t bytes=0;
    const uint64_t* offsets=nullptr;
    const int32_t* local=nullptr;
    int32_t slots=0;
    uint64_t min_blob_bytes=0;
};
// Exact native_expert_grouped adapter:
// (ptr,start,&groups,dst,tok,T*K,T*K,original_Q8,scratch,compact_FP32,stream).
// Every live dst[i]==i; original[i] is its original WINDOW row, not compact row.
// Starts/groups/entries and all inactive bytes are zeroed on every execution.
// Any error leaves groups=entries=0 and no pointers; caller must surface error.
struct alignas(16) Plan {
    unsigned long long ptr[Cap];
    int32_t start[Cap+1], dst[Cap], tok[Cap], original[Cap];
    int32_t groups, entries;
    uint32_t error, epoch;
};
static_assert(sizeof(Request)==32 && sizeof(Plan)==1952);
static_assert(std::is_standard_layout_v<Plan> && std::is_trivially_copyable_v<Plan>);
// No allocation, host wait, device query, upload, weight read, or fallback.
// ids has >=80 accessible int32s (only T*K are read); local/offsets are GPU
// pointers on the owning device. Inputs/output must not overlap. The caller
// orders original IDs/cache-map readiness and later raw-return consumption.
cudaError_t partial_plan(const int32_t* ids, const Request* request,
                         CacheView cache, Plan* out, cudaStream_t stream) noexcept;
} // namespace swift_ep_v26
