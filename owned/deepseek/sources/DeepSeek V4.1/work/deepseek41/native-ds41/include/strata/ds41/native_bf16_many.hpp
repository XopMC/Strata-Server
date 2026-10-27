#pragma once
#include "strata/ds41/ops.hpp"
namespace strata::ds41::ops {
// Preserve each row's lane ownership, FMA order and warp reduction. Weight
// values are shared across a maximum of eight independent input rows.
void bf16_linear_many(const bf16* x, const bf16* w, int m, int64_t k,
                      int64_t n, float* y, cudaStream_t stream=0);
}
