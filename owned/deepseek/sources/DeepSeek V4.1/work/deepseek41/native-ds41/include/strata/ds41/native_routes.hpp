#pragma once
#include <cstdint>
#include <vector>
namespace strata::ds41 {
struct NativeRoutePlan {std::vector<int> offsets,tokens;std::vector<float>weights;};
// Stable token order within ascending expert IDs. Each expert owns a token
// at most once, permitting deterministic non-atomic GPU scatter per expert.
NativeRoutePlan native_route_plan(const int32_t*ids,const float*weights,int rows,int topk,int experts);
}
