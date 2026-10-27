#pragma once
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
namespace strata::kernels::actstage_v264 {
void component_gu(const NativeExpertLayout&,const unsigned long long*,const int*,const int*,const int*,const void*,float*,float*,dim3,cudaStream_t);
}
