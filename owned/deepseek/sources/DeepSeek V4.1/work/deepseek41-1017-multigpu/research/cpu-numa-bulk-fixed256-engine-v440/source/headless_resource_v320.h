#pragma once
#include "strata/ds41/native_engine_causal_batch_v310.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Existing SAME-DSO ds41_checkpoint_create handle, same real retained bank and
   hash as the fresh headful binding. No fake checkpoint/foreign registry/PID.
   Explicit DS1017_HEADLESS_PREFILL_RESOURCE_V320=1 and role flag required.
   Existing controller pair lease/freshchild only. Constructor+close, NO PP/model
   arithmetic. Selects GPU1 then restores callerGPU0; exactpair mask validated.
   stats32: [320,stage,524288,freeBefore1,freeConstructed1,freeDestroyed1,budget,
   reserve,headless_info16...,pinnedEmbedding,basePlan,fullLayerPlan,bankCapacity,
   ownerGeneration,formatEpoch,destroyed,restored0]. Stages1args/2pair/3construct/
   4constructed/5closed/6destroyed. Failure leaves actual stage/error, nonzero.
   Memory values are observed samples, NOT peak/no-leak claims. */
int ds320_headless_resource_component(void*checkpoint,const ds41_native_hash*hash,
 uint64_t budget,uint64_t reserve,uint64_t*stats,int count,char*error,size_t error_bytes);
#ifdef __cplusplus
}
#endif
