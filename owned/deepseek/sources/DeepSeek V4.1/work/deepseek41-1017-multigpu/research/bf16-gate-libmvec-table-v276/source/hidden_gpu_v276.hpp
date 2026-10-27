#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
/* Launch only: caller already admitted current device0/1 and immutable device
 table/domain uploaded from qualified Ds276ExpTable. No CUDA host API queries,
 allocations, synchronization or device hops inside this function. Active route
 widths2304, routes1..36. error must be zeroed before launch; nonzero forbids use
 of ANY outputs/no retry fallback. All pointers stay alive through completion. */
void ds276_launch_hidden(const uint32_t*exp_bits,const uint8_t*domain,
 const float*gate,const float*up,const float*alpha,int routes,
 uint16_t*hidden_bf16,float*hidden_fp8,int8_t*a8,float*a8_scales,
 int32_t*per4_sums,int*error,cudaStream_t stream);
