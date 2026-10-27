#pragma once
#include <cstdint>
#include <cstddef>
#include "native_a8_prequant_views_v118.hpp"
constexpr uint32_t DsGuVersionV18=18;
struct DsGuGroupV18 {
 uint32_t expert,count,output_words,reserved;
 uintptr_t gate_key,up_key;
 uint32_t query[6],route_slot[6];
};
static_assert(sizeof(DsGuGroupV18)==80);
struct DsGuCommandV18 {
 uint32_t struct_size,version;uint64_t epoch;
 uint32_t rows,dim,inter,groups,output_words,reserved;
 Ds41A8ViewV118 view[6];DsGuGroupV18 group[12];
};
struct DsGuHooksV18 {
 uint32_t struct_size,version;void*owner;
 void(*retain)(void*);void(*release)(void*);
 int(*contains)(void*,int expert,uintptr_t gate_key,uintptr_t up_key);
 int(*begin)(void*,const DsGuCommandV18*,void**ticket);
 int(*finish)(void*,void*ticket,uint64_t epoch,const float**flat_output,uint64_t output_words);
 int(*release_view)(void*,void*ticket,uint64_t epoch);
 int(*abort)(void*,void*ticket,uint64_t epoch);
};
// All callbacks are noexcept across ABI. begin snapshots borrowed views before
// returning and drains internally if it fails after enqueue; no failure retry.
// Owner is retained from successful attach until detach/Engine destruction.
extern "C" int ds41_moe_gu_attach_v18(void*engine,const DsGuHooksV18*hooks);
extern "C" int ds41_moe_gu_stats_v6(void*task,uint64_t*out,int count);

extern "C" int ds41_moe_gu_test_cpu_fail_v6(void*task); // test flag required, never production admission

// finish borrows immutable pinned result while ticket/mutex remains live.
// release_view only after synchronous post join; abort drains on any error.
extern "C" int ds41_moe_gu_trace_enable_v18(void*task,int enable);
