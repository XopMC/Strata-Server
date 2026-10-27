#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Actual native Engine Handle only, mutex-serialized N1. Never Engine or pool pointer casts.
   370 receipt is authoritative sizeof(ds370::Receipt370)==672 (84 u64),
   not a duplicate new numerical schema. Read before Engine destroy only. */
int ds387_native_engine_adaptive_arm(void*handle,uint64_t generation,uint64_t format_epoch);
/* rc0+disposition0: Owner.gap_service succeeded. rc0+disposition1: validated
   armed initial-no-forward (or wholeOFF) skip. All other outcomes fail closed.
   mainCPU255; nonce strictly increases even skipped calls; max_starts==2. */
int ds387_native_engine_adaptive_gap(void*handle,uint64_t generation,uint64_t format_epoch,
                                   uint64_t nonce,unsigned max_starts,unsigned*disposition);
int ds387_native_engine_adaptive_receipt(void*handle,void*out,size_t bytes672);
/* Terminal close: detach policy then whole hooks, positive owner upload/compute
   drain, exact NativeExperts lease unregister; Engine stillalive but unusable.
   Copies real closed receipt; normal ds41_native_engine_destroy follows. */
int ds387_native_engine_adaptive_close(void*handle,void*out,size_t bytes672);
#ifdef __cplusplus
}
#endif
