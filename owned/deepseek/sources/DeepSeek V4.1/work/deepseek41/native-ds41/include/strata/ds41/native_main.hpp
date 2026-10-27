#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace strata::ds41 {
// Capture the mean of four HC copies, in BF16, for DSpark target layers37..39.
// ring stores128 chronological positions modulo128, each [3,5120].
void native_main_capture(const uint16_t* hc,int rows,int position,int target,uint16_t* ring,cudaStream_t stream=0);
void native_main_commit(const uint16_t* tentative,int keep,int position,uint16_t* ring,cudaStream_t stream=0);
}
