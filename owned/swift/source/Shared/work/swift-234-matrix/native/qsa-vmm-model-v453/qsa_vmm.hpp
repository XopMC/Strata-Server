#pragma once
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>
namespace qsa_vmm453 {
struct Layer{int global_layer;float* pooled;const float* dead;int64_t cells;};
float* allocate_pool(const void* state,int64_t rows,int dim);
struct Lifetime{~Lifetime();};
bool before_prompt(std::string&) noexcept;
bool activate_request(int64_t prefix,std::string&) noexcept;
bool prepare(const void*,int,int,int64_t,cudaStream_t,const std::vector<Layer>&,std::string&) noexcept;
bool select_graph_bank(const void*,cudaGraphExec_t*,cudaGraphExec_t*,bool*,bool*,int,std::string&) noexcept;
void enqueue(const void*,int,int,int,const float*,const float*,const float*,const int32_t*,float*,int32_t*,const strata::kernels::QsaShapes&,cudaStream_t);
void release(const void*) noexcept;
}
