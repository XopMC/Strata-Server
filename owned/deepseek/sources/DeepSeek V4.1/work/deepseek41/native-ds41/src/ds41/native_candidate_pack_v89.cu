#include "strata/ds41/native_candidate_pack.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpsPerBlock = kThreads / 32;
constexpr unsigned kFullWarp = 0xffffffffu;

template <class T>
bool aligned(const T* pointer) {
    return pointer && reinterpret_cast<uintptr_t>(pointer) % alignof(T) == 0;
}

bool valid_rows(int rows) {
    return rows >= 1 && rows <= strata::ds41::kNativeCandidatePackMaxRows;
}

size_t word_count(int rows) {
    return size_t(rows) * strata::ds41::kNativeCandidateValidWordsPerRow;
}

__global__ void pack_rows(const int32_t* input, uint16_t* payload,
                          uint32_t* validity, size_t words) {
    const size_t word = size_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const unsigned lane = threadIdx.x & 31;
    // Every row has64 full warps, so this condition is warp-uniform and all
    // participating lanes reach ballot with the same full-warp mask.
    if (word >= words) return;
    const size_t at = word * 32 + lane;
    const int32_t value = input[at];
    assert(value == -1 || (value >= 0 && value <= 65535));
    const bool valid = value != -1;
    payload[at] = valid ? uint16_t(value) : uint16_t(0);
    const uint32_t mask = __ballot_sync(kFullWarp, valid);
    if (lane == 0) validity[word] = mask;
}

__global__ void unpack_rows(const uint16_t* payload, const uint32_t* validity,
                            int32_t* output, size_t words) {
    const size_t word = size_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const unsigned lane = threadIdx.x & 31;
    if (word >= words) return;
    const size_t at = word * 32 + lane;
    const uint32_t local = lane == 0 ? validity[word] : uint32_t(0);
    const uint32_t mask = __shfl_sync(kFullWarp, local, 0);
    output[at] = (mask & (uint32_t(1) << lane)) ? int32_t(payload[at]) : int32_t(-1);
}

void check_status(int code) {
    if (code == -1) throw std::invalid_argument("native candidate pack: invalid bounded geometry/buffers");
    if (code != 0) throw std::runtime_error(std::string("native candidate pack: ") +
        cudaGetErrorString(static_cast<cudaError_t>(code)));
}

}  // namespace

extern "C" int ds41_candidate_pack(const int32_t* input, uint16_t* payload,
                                   uint32_t* validity, int rows, CUstream_st* stream) {
    if (!valid_rows(rows) || !aligned(input) || !aligned(payload) || !aligned(validity)) return -1;
    const size_t words = word_count(rows);
    const unsigned blocks = unsigned((words + kWarpsPerBlock - 1) / kWarpsPerBlock);
    pack_rows<<<blocks, kThreads, 0, stream>>>(input, payload, validity, words);
    return int(cudaGetLastError());
}

extern "C" int ds41_candidate_unpack(const uint16_t* payload, const uint32_t* validity,
                                     int32_t* output, int rows, CUstream_st* stream) {
    if (!valid_rows(rows) || !aligned(payload) || !aligned(validity) || !aligned(output)) return -1;
    const size_t words = word_count(rows);
    const unsigned blocks = unsigned((words + kWarpsPerBlock - 1) / kWarpsPerBlock);
    unpack_rows<<<blocks, kThreads, 0, stream>>>(payload, validity, output, words);
    return int(cudaGetLastError());
}

namespace strata::ds41 {

void native_candidate_pack(const int32_t* input, uint16_t* payload,
                           uint32_t* validity, int rows, CUstream_st* stream) {
    check_status(ds41_candidate_pack(input, payload, validity, rows, stream));
}

void native_candidate_unpack(const uint16_t* payload, const uint32_t* validity,
                             int32_t* output, int rows, CUstream_st* stream) {
    check_status(ds41_candidate_unpack(payload, validity, output, rows, stream));
}

}  // namespace strata::ds41
