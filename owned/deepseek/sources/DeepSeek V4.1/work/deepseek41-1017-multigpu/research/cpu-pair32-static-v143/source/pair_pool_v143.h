#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All members describe immutable retained paired139 bytes, not a raw model.
   role0=w1/gate,1=w3/up,2=w2/down. Exact name/role/shape must agree.
   data2MiBaligned, bytes6635520, bits0, tag139, positive generations.
   One owner_generation/bank_format_epoch per fresh DSO registry. */
typedef struct Ds143PairDescriptor {
 uint64_t struct_bytes;
 uint32_t layout_tag;int32_t bits;
 const void* data;uint64_t bytes;
 int32_t rows,columns,layer,expert,role,reserved;
 uint64_t owner_generation,bank_format_epoch;
} Ds143PairDescriptor;
/* Requires env DS1017_CPU_PAIR32_LAYOUT_V143=139. Validates ALLfooter words<=254
   once at tagged registration; no numerical decode. Caller retains these exact
   readonly buffers until every pool is destroyed. No ownership/copy or unregister. */
int ds143_pair32_cache_register(const char*name,const Ds143PairDescriptor*descriptor);
/* New pool getter, idleonly12words:
 [143,139,32,576,288,1(static),workers,entries,bytes,owner_generation,format_epoch,0].
 Info requires a successfully opted-in created pool, not inferred from env. */
int ds143_pair32_pool_info(void*pool,uint64_t*values,int words);
#ifdef __cplusplus
}
static_assert(sizeof(Ds143PairDescriptor)==72);
#endif
