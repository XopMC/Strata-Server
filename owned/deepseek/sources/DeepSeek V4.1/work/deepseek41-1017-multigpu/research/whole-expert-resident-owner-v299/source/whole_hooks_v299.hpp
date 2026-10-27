#pragma once
#include <cstddef>
#include <cstdint>
#include "../../cpu-pair32-static-v143/source/native_a8_prequant_views_v118.hpp"
constexpr uint32_t DsWholeVersionV299=299;
constexpr int DsWholeDeclinedBeforeSubmitV299=10;
struct DsWholeGroupV299 {
 uint32_t expert,count,reserved[2];
 uintptr_t gate_key,up_key,down_key;
 uint32_t query[6],route_slot[6];float alpha[6];
};
struct DsWholeCommandV299 {
 uint32_t struct_size,version;uint64_t epoch;
 uint32_t rows,dim,inter,groups,route_words,reserved;
 Ds41A8ViewV118 view[6];DsWholeGroupV299 group[36];
};
struct DsWholeHooksV299 {
 uint32_t struct_size,version;void*owner;
 void(*retain)(void*);void(*release)(void*);
 // -1 invalid identity;0 miss;1 residentGPU0;2 residentGPU1. Immutable after create.
 int(*contains)(void*,int expert,uintptr_t gate,uintptr_t up,uintptr_t down);
 // All input views/metadata snapshotted BEFORE return. rc10 only if NO enqueue
 // occurred and NO ticket/output was published; caller may use original CPU path.
 int(*begin)(void*,const DsWholeCommandV299*,void**ticket);
 // Returns original sorted-route BF16 [rows][6][5120], never a GPU subtotal.
 // All nonselected route slots are zero. Ticket/mutex/host output stay borrowed.
 int(*finish)(void*,void*,uint64_t,const uint16_t**route_bf16,uint64_t words);
 int(*release_view)(void*,void*,uint64_t);
 int(*abort)(void*,void*,uint64_t);
};
extern "C" int ds41_moe_whole_attach_v299(void*engine,const DsWholeHooksV299*);
extern "C" int ds41_moe_whole_stats_v299(void*task,uint64_t*out,int words);
// Same N1 Engine/Task lifetime as18. No attach/detach/getter concurrently with run.
// Callbacks are noexcept across ABI. Hook retention outlives all Task callbacks.
