#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace swift_selector_v7_leaf {
constexpr int Parts=16,Keep=514,Packed=Parts*Keep+1;
struct Workspace {
 int32_t *local_ids,*counts,*mapping,*packed_steps,*packed_ids;
 float* packed_scores;
};
cudaError_t chunks(const float*,const int32_t*,int,int64_t,Workspace,int,cudaStream_t);
cudaError_t pack(const float*,const int32_t*,int,int64_t,Workspace,cudaStream_t);
cudaError_t translate(const int32_t*,int,int64_t,Workspace,int32_t*,cudaStream_t);
}
