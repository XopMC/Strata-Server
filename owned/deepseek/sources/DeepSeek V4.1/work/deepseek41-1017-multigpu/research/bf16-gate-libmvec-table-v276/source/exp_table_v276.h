#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Caller owns immutable values[65536] and domain[65536], no retained pointers.
 Canonical input is already BF16-rounded and clamp(g,upper=10) applied.
 All finite negative codes and positive codes<=0x4120 are eligible. exp overflow
 is legal; output underflow is excluded by g<=10. Invalid codes are NOT zeros. */
typedef struct Ds276ExpTable {
 uint32_t struct_bytes,version,entries,valid_entries;
 uint32_t limit_f32_bits,mxcsr_controls,reserved0,reserved1;
 const uint32_t*values;const uint8_t*domain;
 uint64_t content_fnv64,resolved_libmvec;
} Ds276ExpTable;
/* stats16: [276,valid_entries,vectorcalls,lanewords,mxcsr_before,mxcsr_after,
 control_before,control_after,exp_overflow_entries,0...]. No FP-mode writes. */
int ds276_exp_table_build(Ds276ExpTable*,uint32_t*values,size_t entries,uint8_t*domain,size_t domain_entries,uint64_t*stats,int words);
/* Full content/domain check once before immutable use/upload; not a hot lookup. */
int ds276_exp_table_validate(const Ds276ExpTable*);
/* Bounded16-lane CPU lookup, all guards BEFORE output writes; requires caller
 to keep a successfully validated table immutable/alive. Does not run libmvec. */
int ds276_exp_lookup16(const Ds276ExpTable*,const uint16_t*gates,uint32_t*out);
#ifdef __cplusplus
}
static_assert(sizeof(Ds276ExpTable)==64);
#endif
