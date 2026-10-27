#pragma once
#include <cstdint>
namespace strata::core {
struct ExpertDispatch;
// Private consumer contract, not a generic dispatch alternative. Call ONLY
// while the sealed V65 EP bridge is activated, from drive_pool_multi after
// assigning d.layers. Its captured UnionRows copy_rows_from_mapped must zero
// every logical GPU-hit destination and never read those mapped host rows.
// This free function owns a same-thread, one-dispatch TLS lease. It validates
// exact arguments/modes and permanently fails d on a mismatch; no fallback.
// No existing struct layout, CPU job, GPU kernel or arithmetic is changed.
void expert_pool_dispatch_multi_gpu_zeroed_v66(ExpertDispatch& d,const float* x_f,
    const int32_t* ids,int64_t n_tok,int64_t k,float* out);
}
