#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Exact original244 FP8 arithmetic, paired139 byte-address reader only.
   One projection or literal gate/up/down triple; 72B POD per projection.
   Only bit0 GU2304x5120 / DOWN5120x2304, source6635520B, q=rows*cols,
   scales=rows*4. All source/output ranges must be disjoint actual cudaMalloc
   ranges on current explicit expected logical0/1. Interior pointers supported.
   Explicit DS1017_PREFILL_DIRECT_FP8_V531=1 must precede first call (latched).
   No allocation/sync/P2P/devicechange; capture refused. Caller retains ALL
   ranges and host descriptor until enqueue returns; device ranges until stream
   completion. Three descriptors are validated completely BEFORE any launch.
   Original244 DSO/API and NativeFp8 public classes are untouched. */
typedef struct Ds531Projection {
 uint64_t struct_bytes;uint32_t layout_tag;int32_t source_bits;
 const void*source;uint64_t source_bytes;
 void*fp8;uint64_t fp8_bytes;
 void*row_scales;uint64_t row_scale_bytes;
 int32_t rows,columns;
} Ds531Projection;
enum Ds531Status {
 DS531_POLICY=-53101,DS531_ARGUMENT=-53102,DS531_ALIAS=-53103,
 DS531_DEVICE=-53104,DS531_POINTER=-53105,DS531_CAPTURE=-53106,
 DS531_CUDA_GUARD=-53107
};
int ds531_pair32_fp8_encode(const Ds531Projection*projections,size_t count,
 int expected_device,uint64_t stream);
const char*ds531_pair32_fp8_error(void);
/* Root's bounded SOURCEfixture only, no model/bank ownership. Synthetic mode
   tests six actual-shape numeric images + one triple, original244 oracle.
   Actual entries accept ONE retained readonly old16 or paired139 matrix, per
   explicit parent format. Fixture makes bounded private copies; never changes
   source/parent. Current-index pair lease/logical-order admission is external.
   At most128MiB GPU /96MiB CPU; no cudaSetDevice/reset. */
typedef struct Ds531Report {
 uint64_t schema,struct_bytes,stage,expected_device,cases,baseline_kernel_calls;
 uint64_t candidate_kernel_calls,compared_fp8_bytes,compared_scale_words;
 uint64_t guard_bytes,source_unchanged_checks,invalid_refusals;
 uint64_t host_peak_bytes,device_peak_bytes,zero_codes,signed_zero_codes;
 uint64_t small_scales,large_scales,triple_cases,reserved[3];
 char error[256];
} Ds531Report;
int ds531_component_synthetic(int expected_device,Ds531Report*out);
int ds531_component_retained(const void*old16,size_t old16_bytes,int rows,int columns,
 int expected_device,Ds531Report*out);
int ds531_component_retained_pair139(const void*paired,size_t paired_bytes,int rows,int columns,
 int expected_device,Ds531Report*out);
#ifdef __cplusplus
}
static_assert(sizeof(Ds531Projection)==72);
#endif
