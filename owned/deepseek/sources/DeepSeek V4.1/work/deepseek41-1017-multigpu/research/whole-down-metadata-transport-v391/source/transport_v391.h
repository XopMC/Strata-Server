#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* DSO-local process aggregate, idle read recommended; exact16words below.
   Counters describe SUCCESSFUL complete enqueues; failed enqueue count invalidates
   byte-accounting completeness (partial submission never claimed as zero traffic).
   No reset, pool/Engine pointer, guessed layout, or env-only branch proof. */
int ds391_whole_down_transport_info(uint64_t*out,int words)
#ifdef __cplusplus
 noexcept
#endif
;
#ifdef __cplusplus
}
#endif
/* 0version391,1words16,2liveOperators,3created,4createdOFF,5createdON,
6successfulEnqueues,7successfulONenqueues,8fullDOWNchunks,9prefixDOWNchunks,
10DOWN_H2D_bytes,11oldEquivalentDOWN_H2D_bytes,12avoidedDOWN_H2D_bytes,
13hostDOWNclearBytes,14failedEnqueues,15reserved0. Public303 stats schema unchanged. */
