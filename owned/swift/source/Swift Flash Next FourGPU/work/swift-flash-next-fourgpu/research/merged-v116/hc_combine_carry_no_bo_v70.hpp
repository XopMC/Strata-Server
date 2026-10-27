#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Private SOURCE-ONLY experiment. No public Strata structure or numerical ABI changes.
// Exactly N2560/HC4/K10, BF16 mode0, ldx10240, no low parts/pad/FP16 image.
namespace swift_hc_carry_no_bo_v70 {
struct Args {
    const float* dm = nullptr;
    const int32_t* slot = nullptr;
    const float* weight = nullptr;
    const float* shared = nullptr;
    const float* shared_gate = nullptr;
    const float* row_sd = nullptr;
    const float* inject = nullptr;
    const float* next_norm = nullptr;
    float* bo = nullptr;                 // Still valid, allocated and disjoint even when not stored.
    bool store_bo = true;               // Default retains diagnostics; false needs caller dead-store proof.
    float* residual = nullptr;           // In-place R, outer allocation.
    float* rs = nullptr;                 // Outer allocation.
    uint16_t* xn16 = nullptr;            // Must equal retired Xq's start.
    const uint8_t* retired_xq = nullptr;  // Actual carved m.Xq; checked independently.
    uint64_t xq_payload_bytes = 0;       // Payload only: never the allocator guard.
    const uint8_t* region = nullptr;
    uint64_t region_bytes = 0;
    uint64_t hc_view_bytes = 0;          // Full counted HC view, including its guards.
    int64_t tokens = 0;                  // 1..49152; valid slot rows in [0,tokens*10).
    float eps = 1e-6f;
};
// Host interval/admission checks only; no allocations, CUDA queries or synchronization.
// Caller additionally owns the same-stream completion and immutable-input contract.
cudaError_t validate(const Args&);
cudaError_t launch(const Args&, cudaStream_t);
}
