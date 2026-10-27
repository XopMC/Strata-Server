#pragma once
#include <stdint.h>
#include "../../cpu-pair32-static-v143/source/pair_pool_v143.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Private CPU418: original143 math, singleton-home pools. Caller owns both
   opaque pool handles and retains all registered139 bytes until AFTER Task
   engine close. Same DSO/registry for both; no model bytes copied. */
void* ds418_home_pool_create(const int*cpus,int count,int home,int*status);
int ds418_home_pool_info(void*pool,uint64_t*out,int words); /*8:418,home+1,workers,N0,N1,generation,entries,bytes*/
int ds418_home_pool_cpus(void*pool,int*out,int count);
int ds418_paired_descriptor(const void*registered_key,Ds143PairDescriptor*out);
/* New Task418 typed creation. gemv_function belongs to SAME CPU418 DSO as both
   pool handles. CPU lists are queried from genuine handles, not this struct.
   Coordinator1 is the original synchronous Task caller254; Task owns exactly
   ONE helper coordinator127. Caller must invoke run on254 and stay N1/idle. */
typedef struct Ds418Pools {
 uint64_t struct_bytes,version;
 void*pool[2];uintptr_t gemv_function;
 int32_t workers[2],coordinator[2];
 uint64_t reserved[2];
} Ds418Pools;
void* ds418_moe_engine_create(const Ds418Pools*binding);
/* Actual Task-engine handle only, idle16words. No Engine wrapper cast.
   418,enabled,workers0,workers1,coord0,coord1,jobs,home0GU,home0post,home0DOWN,
   home1GU,home1post,home1DOWN,errors,active,component419(0 production /1 explicit18expert fixture).
   Getter/reset is not an environment assertion; successful jobs counted after
   BOTH homes have ACKed, errors after both drain. */
int ds418_moe_engine_info(void*actual_task_engine,uint64_t*out,int words);
/* Same16words, single-enabled-engine weak receipt token; idle only. rc2 when
   absent/expired/active. No Engine/pool address is accepted or fabricated. */
int ds418_moe_process_info(uint64_t*out,int words);
#ifdef __cplusplus
}
static_assert(sizeof(Ds418Pools)==72);
#endif
