// Isolated K3 causal-prefix component; no Engine or model integration.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ds41::kernels::causal_v309 {
// Caller guarantees sequential params rows [token,pos,pos+1,(pos+1)/2],
// immutable pre-window[128,512], prefix_rows[m,512], valid causal indices
// with stride640. Each query uses only prefix_rows[0..query]. No state writes.
void sparse_attn_causal_v309(const __nv_bfloat16* q,
                            const __nv_bfloat16* window,
                            const __nv_bfloat16* prefix_rows,
                            const __nv_bfloat16* comp,
                            const int32_t* idx, const int* params, int m,
                            int index_stride, int ratio, const float* sink,
                            float scale, __nv_bfloat16* out,
                            cudaStream_t stream);
}
