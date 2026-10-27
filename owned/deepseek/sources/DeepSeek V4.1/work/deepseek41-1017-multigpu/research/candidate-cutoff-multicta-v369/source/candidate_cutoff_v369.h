#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Caller owns ALL device spans until its explicit stream completes. No alias.
// Selected SM89 logical0/1 only, qualified by component/caller; no device hop.
// No internal allocation, sync, pointer queries, cache or capture-time host IO.
// enable0 = literal172; enable1 = four-byte multiCTA cutoff. Values/MAX unchanged.
size_t ds369_candidate_workspace_bytes(int64_t capacity,int block);
int ds369_candidate_static(const float*scores,int64_t n,int topk,int block,
 uint8_t*mask,void*workspace,size_t workspace_bytes,void*nonblocking_stream,int enable);
int ds369_candidate_dynamic(const float*scores,const int32_t*position,int ratio,
 int64_t capacity,int topk,int block,uint8_t*mask,void*workspace,size_t workspace_bytes,
 void*nonblocking_stream,int enable);
// Ordinary-idle diagnostics only. Actual CUDA attributes/current device, not TPS.
int ds369_candidate_resource_info(uint64_t*out,int words32);
#ifdef __cplusplus
}
struct Ds369State {
 uint64_t positions,blocks,remaining;
 uint32_t prefix,prefix_mask;
 int32_t k,mode,status,reserved0;
 uint64_t reserved[2];
};
static_assert(sizeof(Ds369State)==64,"369 fixed callerworkspace header");
#endif
// Workspace header mode0=zero-live,1=literal tiny/full/zero-k fallback,3=multiCTA.
// Its state is readable only AFTER same-stream completion. status0 success;
// no CPU mode/counter or model use may be inferred before actual completion.
