#include "route_grouping.hpp"
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>
#include <limits>

namespace swift_route_v109r1 {
namespace {
constexpr std::size_t align256(std::size_t x) { return (x + 255) & ~std::size_t(255); }
bool shape(const Plan& p) {
    return p.tokens >= 0 && p.tokens <= kMaxTokens && p.experts == kExperts &&
           p.ranks == kRanks && p.routes == p.tokens * kRanks;
}
bool disjoint(const void* a, std::size_t an, const void* b, std::size_t bn) {
    if (!an || !bn) return true;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    if (!x || !y || x > std::numeric_limits<std::uintptr_t>::max() - an ||
        y > std::numeric_limits<std::uintptr_t>::max() - bn) return false;
    return x + an <= y || y + bn <= x;
}
__global__ void validate(const int32_t* ids, const int32_t* identity, int n, unsigned* error) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    unsigned e = 0;
    if (ids[i] < 0 || ids[i] >= kExperts) e |= kBadRoute;
    if (identity[i] != i) e |= kBadIdentity;
    if (e) atomicOr(error, e);
}
__device__ int lower(const int32_t* key, int n, int value) {
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (key[mid] < value) lo = mid + 1; else hi = mid;
    }
    return lo;
}
__global__ void counts_offsets(const int32_t* key, int n, const unsigned* error,
                               int32_t* count, int32_t* offset) {
    const int e = int(blockIdx.x * blockDim.x + threadIdx.x);
    const bool bad = *error != 0;
    if (e < kExperts) {
        const int start = bad ? 0 : lower(key, n, e);
        const int stop = bad ? 0 : lower(key, n, e + 1);
        count[e] = stop - start;
        offset[e] = start;
    }
    if (e == 0) offset[kExperts] = bad ? 0 : n;
}
__global__ void inverse(const int32_t* sorted_i, int32_t* slot, int32_t* src, int n, const unsigned* error) {
    if (*error) return;
    const int p = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n) return;
    const int i = sorted_i[p];
    // validate already proved a permutation0..n-1; retain the bound as an OOB backstop.
    if (i < 0 || i >= n) return;
    slot[i] = p;
    src[p] = i / kRanks;
}
}  // namespace

cudaError_t query(int tokens, int experts, int ranks, cudaStream_t stream, Plan* out) {
    if (!out) return cudaErrorInvalidValue;
    *out = Plan{};
    if (tokens < 0 || tokens > kMaxTokens || experts != kExperts || ranks != kRanks)
        return cudaErrorInvalidValue;
    Plan p;
    p.tokens = tokens; p.experts = experts; p.ranks = ranks; p.routes = tokens * ranks;
    auto e = cudaGetDevice(&p.device);
    if (e != cudaSuccess) return e;
    std::size_t cursor = 0;
    p.sorted_keys_at = cursor; cursor += align256(std::size_t(p.routes) * sizeof(int32_t));
    p.sorted_indices_at = cursor; cursor += align256(std::size_t(p.routes) * sizeof(int32_t));
    p.count_at = cursor; cursor += align256(kExperts * sizeof(int32_t));
    p.offset_at = cursor; cursor += align256((kExperts + 1) * sizeof(int32_t));
    p.error_at = cursor; cursor += align256(sizeof(unsigned));
    p.cub_at = cursor;
    if (p.routes) {
        // Query only; nullptr storage causes no sorting work. Pointer API does not overwrite inputs.
        e = cub::DeviceRadixSort::SortPairs(nullptr, p.cub_bytes,
            static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            p.routes, 0, 9, stream);
        if (e != cudaSuccess) return e;
    }
    if (p.cub_bytes > std::numeric_limits<std::size_t>::max() - cursor) return cudaErrorInvalidValue;
    p.required_bytes = cursor + p.cub_bytes;
    *out = p;
    return cudaSuccess;
}
bool fits(const Plan& p, const void* scratch, std::size_t capacity) {
    if (!shape(p) || p.device < 0 || !scratch ||
        (reinterpret_cast<std::uintptr_t>(scratch) & 255u)) return false;
    std::size_t cursor = 0;
    if (p.sorted_keys_at != cursor) return false;
    cursor += align256(std::size_t(p.routes) * sizeof(int32_t));
    if (p.sorted_indices_at != cursor) return false;
    cursor += align256(std::size_t(p.routes) * sizeof(int32_t));
    if (p.count_at != cursor) return false;
    cursor += align256(kExperts * sizeof(int32_t));
    if (p.offset_at != cursor) return false;
    cursor += align256((kExperts + 1) * sizeof(int32_t));
    if (p.error_at != cursor) return false;
    cursor += align256(sizeof(unsigned));
    if (p.cub_at != cursor || p.cub_bytes > std::numeric_limits<std::size_t>::max() - cursor ||
        p.required_bytes != cursor + p.cub_bytes || p.required_bytes > capacity) return false;
    return reinterpret_cast<std::uintptr_t>(scratch) <=
        std::numeric_limits<std::uintptr_t>::max() - p.required_bytes;
}
Results results(const Plan& p, const void* scratch) {
    if (!scratch) return {};
    const auto* b = static_cast<const unsigned char*>(scratch);
    return {reinterpret_cast<const int32_t*>(b + p.count_at),
            reinterpret_cast<const int32_t*>(b + p.offset_at),
            reinterpret_cast<const unsigned*>(b + p.error_at)};
}
cudaError_t enqueue(const Plan& p, void* scratch, std::size_t capacity,
                    const int32_t* ids, const int32_t* identity, int32_t* slot, int32_t* src,
                    cudaStream_t stream) {
    if (!fits(p, scratch, capacity)) return cudaErrorInvalidValue;
    int current = -1;
    auto e = cudaGetDevice(&current);
    if (e != cudaSuccess) return e;
    if (current != p.device) return cudaErrorInvalidDevice;
    const void* ptr[] = {ids, identity, slot, src, scratch};
    const std::size_t rbytes = std::size_t(p.routes) * sizeof(int32_t);
    const std::size_t length[] = {rbytes, rbytes, rbytes, rbytes, p.required_bytes};
    for (int a = 0; a < 5; ++a)
        for (int b = a + 1; b < 5; ++b)
            if (!disjoint(ptr[a], length[a], ptr[b], length[b])) return cudaErrorInvalidValue;
    if (p.routes && (!ids || !identity || !slot || !src)) return cudaErrorInvalidValue;
    auto* base = static_cast<unsigned char*>(scratch);
    auto* key = reinterpret_cast<int32_t*>(base + p.sorted_keys_at);
    auto* sorted_i = reinterpret_cast<int32_t*>(base + p.sorted_indices_at);
    auto* count = reinterpret_cast<int32_t*>(base + p.count_at);
    auto* offset = reinterpret_cast<int32_t*>(base + p.offset_at);
    auto* error = reinterpret_cast<unsigned*>(base + p.error_at);
    e = cudaMemsetAsync(error, 0, sizeof(unsigned), stream);
    if (e != cudaSuccess) return e;
    if (p.routes) {
        validate<<<(p.routes + 255) / 256, 256, 0, stream>>>(ids, identity, p.routes, error);
        e = cudaGetLastError(); if (e != cudaSuccess) return e;
        std::size_t temp = p.cub_bytes;
        // Invalid route keys may be reordered but never index a table. Error-gated kernels below
        // leave outputs untouched; host MUST observe error before admitting any MMQ.
        e = cub::DeviceRadixSort::SortPairs(base + p.cub_at, temp, ids, key, identity, sorted_i,
                                           p.routes, 0, 9, stream);
        if (e != cudaSuccess) return e;
    }
    counts_offsets<<<2, 256, 0, stream>>>(key, p.routes, error, count, offset);
    e = cudaGetLastError(); if (e != cudaSuccess) return e;
    if (p.routes) inverse<<<(p.routes + 255) / 256, 256, 0, stream>>>(sorted_i, slot, src, p.routes, error);
    return cudaGetLastError();
}
}  // namespace swift_route_v109r1
