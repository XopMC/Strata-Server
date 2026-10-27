#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Exact loaded Task427 DSO, N1 weak receipt, idleonly. rc2 absent/expired/active.
16words:427,latchedmode,active_window_ns,publications,helperclaims,completed,
helperspiniterations,mainspiniterations,helperparks,mainparks,lastpublished,
lastcompleted,errors,active(0 idle),helperCPU127,mainCPU254.
No Engine/pool pointer or inferred environment evidence. */
int ds427_moe_epoch_info(uint64_t*out,int words);
#ifdef __cplusplus
}
#endif
