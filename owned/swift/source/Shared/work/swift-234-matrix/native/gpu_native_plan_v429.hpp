#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace swift_plan_v429 {
constexpr int MaxRows=80,MaxBanks=7;
struct Plan {
 int32_t counts[4],start[81],dst[80],tok[80],padding;
 unsigned long long ptr[80];
};
static_assert(sizeof(Plan)==1624&&offsetof(Plan,ptr)==984);
struct Physical {unsigned long long ptr;int bank,padding;};
static_assert(sizeof(Physical)==16);
// Original residency controls arithmetic classification; physical storage does
// not promote CPU-class experts to the native arithmetic path. The optional
// expert-indexed table serves immutable all-expert residency, not a new role.
cudaError_t build(const int32_t*ids,const int32_t*residency,
 const Physical*physical,int physical_count,bool by_expert,
 unsigned long long logical_base,const uint64_t*logical_offsets,int logical_slots,
 int T,int banks,Plan*bank_plans,Plan*union_plan,uint32_t*sticky_error,cudaStream_t);
}
