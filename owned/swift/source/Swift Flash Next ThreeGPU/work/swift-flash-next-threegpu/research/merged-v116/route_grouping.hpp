#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace swift_route_v109r1 {
constexpr int kExperts = 512, kRanks = 10, kMaxTokens = 32768;
constexpr unsigned kBadRoute = 1u, kBadIdentity = 2u;
struct Plan {
    int tokens = 0, experts = 0, ranks = 0, device = -1;
    int routes = 0;
    std::size_t sorted_keys_at = 0, sorted_indices_at = 0, count_at = 0, offset_at = 0, error_at = 0;
    std::size_t cub_at = 0, cub_bytes = 0, required_bytes = 0;
};
struct Results {
    const int32_t* count = nullptr;    // 512 ints
    const int32_t* offset = nullptr;   // 513 ints, including R sentinel
    const unsigned* error = nullptr;   // MUST be downloaded and checked before any MMQ
};
// Host setup/query only: reads current device / CUB dispatch requirements, launches no kernels.
// No allocations. Query per actual T/E/K/device; no plan from another GPU may be reused.
// T=0 is supported if the caller provides the small fixed scratch (an empty logits span cannot fit it).
cudaError_t query(int tokens, int experts, int ranks, cudaStream_t stream, Plan* out);
// Capacity is the caller's actual dead-logits span, not an invented extra region. 256B base alignment.
// Returns false before enqueue if query + fixed scratch do not fit. No fallback code lives in this API.
bool fits(const Plan&, const void* scratch, std::size_t capacity);
Results results(const Plan&, const void* scratch);
// Async, allocation-free, stream-local. Inputs, outputs, and scratch must be pairwise disjoint.
// Inputs are read-only; identity[i] must equal i. All output table elements beyond R stay unchanged.
// Invalid IDs/identity: error bits set, cnt/off zeroed, slot/src entirely unchanged. No hidden clamp.
// A successful host return means launch submission only. Caller MUST D2H error + cnt and synchronize,
// then abort/fallback before MMQ on any error. Never enqueue dependent MMQ speculatively.
// Plan must have come from query on the current device. Query is outside graph capture; enqueue is captureable.
cudaError_t enqueue(const Plan&, void* scratch, std::size_t capacity,
                    const int32_t* ids, const int32_t* identity,
                    int32_t* slot, int32_t* src, cudaStream_t stream);
}  // namespace swift_route_v109r1
