#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Layout139 is the literal115 pair32 byte layout, not a numeric quant type.
   GU2304x5120 or DOWN5120x2304 only, source/dest6635520 bytes each.
   Footer bytes are opaque; even255/noncanonical words copy bit-exactly.
   expected_device0/1 must be current, exact admitted UUID. NO device hops.
   Both buffers must be disjoint ranges within actual cudaMalloc allocations
   on expected_device; host, managed, mapped and undersized ranges failclosed.
   Caller-owned stream must be on that device; active capture is refused.
   No allocations, sync, copies, numeric conversions, model reads or fallback.
   Success enqueues one byte kernel only. Caller MUST retain source/dest and
   same device/stream until actual stream completion before reuse/free. */
enum Ds141Status {
 DS141_POLICY=-14101,DS141_ARGUMENT=-14102,DS141_OVERLAP=-14103,
 DS141_DEVICE=-14104,DS141_POINTER=-14105,DS141_CAPTURE=-14106,
 DS141_CUDA_GUARD=-14107
};
int ds141_pair32_to_output16(const void*source,size_t source_bytes,int rows,int columns,
 uint32_t layout_tag,void*destination,size_t destination_bytes,
 int expected_device,uint64_t stream);
/* Synthetic, bounded, synchronous fixture only. Caller selects current device
   and retains pair lease; invoke device0 then1 sequentially, NOT concurrently.
   No model/bank pointers/files. The fixture allocates <=16MiB on that device.
   No cudaSetDevice/reset/retry. Result stage records exact failure boundary. */
typedef struct Ds141Report {
 uint64_t schema,struct_bytes,stage,expected_device,cases,kernel_calls;
 uint64_t compared_bytes,guard_bytes,footer_words,footer255_words;
 uint64_t invalid_refusals,host_working_peak_bytes,device_working_peak_bytes;
 uint64_t source_unchanged_checks,reserved[2];
 char error[256];
} Ds141Report;
int ds141_inverse_component(int expected_device,Ds141Report*report);
#ifdef __cplusplus
}
#endif
