#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Source-only private leaf ABI. No owner, allocation, device selection or fallback.
namespace whole_row_tp_v410 {
inline constexpr int Width=2560, Hidden=640, K=10, Capacity=80, MaxTokens=8;
inline constexpr std::uint64_t GuRowBytes=1440, DownRowBytes=360;
inline constexpr std::uint64_t InputQ8RowBytes=2880, HiddenQ8RowBytes=720;
struct ConstSpan { const void* data=nullptr; std::uint64_t bytes=0; };
struct Span { void* data=nullptr; std::uint64_t bytes=0; };
// One geometry is common to ALL groups participating in these launches.
// hidden_begin is 0 or 320, hidden_rows is 320. Output begin/count are positive
// 32-aligned ranges within the ORIGINAL 2560 outputs. Typical cuts:1472/1504.
struct Rows { int hidden_begin=0, hidden_rows=320, output_begin=0, output_rows=0; };
// GPU array indexed by ORIGINAL group g, not logical slot or route row.
// Exactly the four packed slices from row_fragment_v410::Layout:
// gate[460800],up[460800],down[output_rows*360],scales[16]. No full blob.
// All spans are local pointers into allocation in that order, without gaps.
// allocation must be 16-aligned; bytes == 921600+output_rows*360+16.
struct Fragment { ConstSpan allocation, gate, up, down, scales; };
struct Plan {
 ConstSpan fragments; // GPU Fragment[capacity_groups], initialized host image
 ConstSpan counts;    // int32[4]:{groups,entries,0,0}; original native subset
 ConstSpan start;     // int32[capacity_groups+1], start[0]==0,start[groups]==entries
 ConstSpan dst;       // int32[capacity_entries], original rank row in [0,T*10)
 ConstSpan tok;       // int32[capacity_entries], original token == dst/10
 Span error;          // aligned LOCAL DEVICE uint32, caller clears before frame
 int tokens=0, capacity_groups=0, capacity_entries=0;
};
// Full original pitches/extents, even when a leaf owns only a row range.
// gate/up: capacity_entries*640 floats; hq: capacity_entries*20*36 bytes;
// raw: tokens*10*2560 floats; input_q8: tokens*80*36 bytes.
struct Gu { Plan plan; Rows rows; ConstSpan input_q8; Span gate, up; };
struct HiddenQ8 { Plan plan; Rows rows; ConstSpan gate, up; Span hq; };
struct Down { Plan plan; Rows rows; ConstSpan hq; Span raw; };
struct Combine {
 ConstSpan cpu, hits;  // each tokens*10*2560 floats, ORIGINAL rank-major pitch
 ConstSpan dst;        // int32[>=tokens*10], same original union order
 ConstSpan count;      // int32 native entry count (Plan.counts+1 is valid)
 ConstSpan weights;    // tokens*10 floats, unchanged rank order
 ConstSpan shared;     // tokens*2560 floats
 Span output;          // tokens*2560 floats, writes owned output range ONLY
 Span parts_receipt;   // optional full-pitch tokens*10*2560 float receipt
 Span error;           // local DEVICE uint32, caller clears before frame
 int tokens=0, output_begin=0, output_rows=0;
};
enum Error : std::uint32_t { BadCounts=1u, BadStart=2u, BadRoute=4u,
 BadFragment=8u, BadUnion=16u };
// Pure host checks; no CUDA queries or payload reads. The caller must validate
// every host Fragment via fragment_admitted BEFORE uploading the exact image,
// and retain that immutable image, packed payloads and maps through full drain.
bool rows_admitted(Rows) noexcept;
bool fragment_admitted(const Fragment&, Rows) noexcept;
// Nonzero host error means NOTHING enqueued; unsupported spans/configs refuse.
// A device metadata error sets sticky error bits; reject before logits/reuse.
// No function clears error or performs a fallback. Caller owns final error check.
cudaError_t gu(const Gu&, cudaStream_t);
cudaError_t hidden_q8(const HiddenQ8&, cudaStream_t);
cudaError_t down(const Down&, cudaStream_t);
cudaError_t combine(const Combine&, cudaStream_t);
// For graph context/kernel inventory; never invoke these as alternate leaves.
void* gu_function() noexcept;
void* hidden_q8_function() noexcept;
void* down_function() noexcept;
void* combine_function() noexcept;
static_assert(sizeof(ConstSpan)==16 && sizeof(Span)==16 && sizeof(Rows)==16);
static_assert(sizeof(Fragment)==80 && alignof(Fragment)==8);
static_assert(std::is_trivially_copyable_v<Fragment> && std::is_trivially_copyable_v<Plan>);
}
