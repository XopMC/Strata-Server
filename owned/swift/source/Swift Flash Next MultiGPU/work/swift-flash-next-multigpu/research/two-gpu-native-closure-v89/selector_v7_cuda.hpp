#pragma once
#include "selector16.hpp"
#include "strata/kernels/qsa_select.hpp"
namespace swift_selector_v7_leaf {
cudaError_t final_select(const float*,const int32_t*,int,int64_t,int64_t,int32_t*,cudaStream_t);
void leaf_kernels(void**); // local17,local33,pack,translate; cold proof only
void* final_kernel();
}
