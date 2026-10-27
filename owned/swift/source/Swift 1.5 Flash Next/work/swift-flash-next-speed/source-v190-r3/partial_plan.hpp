#pragma once
#include <cuda_runtime.h>
#include <cstdint>
// A private component, not yet integrated into Verifier or public launchers.
namespace swift190 {
constexpr int MaxEntries=128;
cudaError_t prepare();
constexpr int plan_ints(int cap){return ((4+(cap+1)+2*cap+1)&~1)+5*cap+1;}
cudaError_t wait_ready(const uint64_t* mapped_ready,uint32_t ring,uint32_t* error,cudaStream_t stream);
cudaError_t plan(const int32_t* ids,int n,int k,const int32_t* residency,int experts,
 const uint8_t* cache,const uint64_t* offsets,int slots,int64_t blob,
 int32_t* output,int cap,const uint64_t* mapped_ready,uint32_t ring,
 uint32_t* error,cudaStream_t stream);
}
