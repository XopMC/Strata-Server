#pragma once
#include <cstddef>
#include <cstdint>

struct CUstream_st;

namespace strata::ds41 {

inline constexpr int kNativeCandidateIdsPerRow = 2048;
inline constexpr int kNativeCandidateValidWordsPerRow = 64;
inline constexpr int kNativeCandidatePackMaxRows = 524288;
inline constexpr size_t kNativeCandidatePackedBytesPerRow =
    size_t(kNativeCandidateIdsPerRow) * sizeof(uint16_t) +
    size_t(kNativeCandidateValidWordsPerRow) * sizeof(uint32_t);
static_assert(kNativeCandidatePackedBytesPerRow == 4352);

// Lossless domain: {-1} union [0,65535]. The caller must admit context<=524288
// and block size8 before using this codec; block ID65535 is a valid ID.
// Payload is rows*2048 uint16 values. Validity is rows*64 uint32 words;
// bit(i%32) of word(i/32) distinguishes a valid payload from sentinel -1.
// Rows/slots keep their original order. Buffers must be device-accessible,
// naturally aligned, large enough and non-overlapping. Input remains read-only.
// To address pass row first, offset payload by first*2048 and validity by
// first*64. The int32 scratch input/output starts at row0 for this call.
// The functions allocate nothing, enqueue on stream and never synchronize.
void native_candidate_pack(const int32_t* input, uint16_t* payload,
                           uint32_t* validity, int rows, CUstream_st* stream = nullptr);
void native_candidate_unpack(const uint16_t* payload, const uint32_t* validity,
                             int32_t* output, int rows, CUstream_st* stream = nullptr);

}  // namespace strata::ds41

// Probe ABI. Return0 on a successful asynchronous launch, -1 for invalid
// geometry/null/misaligned pointers, otherwise the CUDA launch error code.
// Successful launch does not establish asynchronous completion; the caller
// owns stream synchronization and the legal input-ID domain above.
extern "C" int ds41_candidate_pack(const int32_t* input, uint16_t* payload,
                                   uint32_t* validity, int rows, CUstream_st* stream);
extern "C" int ds41_candidate_unpack(const uint16_t* payload, const uint32_t* validity,
                                     int32_t* output, int rows, CUstream_st* stream);
