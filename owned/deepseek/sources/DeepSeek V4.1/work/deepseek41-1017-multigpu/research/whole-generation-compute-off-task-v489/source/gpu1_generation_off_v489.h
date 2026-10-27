#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Actual split-home Task N1 weak receipt, read only while idle. No reset.
// rc0: populated; rc1: invalid args; rc2: absent/expired/active/uninitialized,
// zero output for valid args on rc2. Calls are cumulative across warmup/generation.
// words: schema489,16,latchedMode,entered,successful,failed,successfulRows,
// successfulPositiveRoutes,actualContainsCalls,rawPresentGroups,rawPresentRoutes,
// forcedMissGroups,forcedMissRoutes,formedCommandGroups,formedCommandRoutes,active.
// Decision counters include failed attempts and DO NOT claim GPU execution.
int ds489_moe_gpu1_generation_info(uint64_t* out,int words);
#ifdef __cplusplus
}
#endif
