#pragma once
#include "tail_plan.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <array>

namespace swift_tail_r027 {
// Root owns current-index admission, typed allocation leases, drains and upload.
// Both phases use the SAME owner/stream/epoch. No allocation or old-path call here.
struct Shape {
    Geometry geometry;
    size_t shared_bytes = 0, partial_bytes = 0;
};
struct Binding {
    int owner = -1;
    cudaStream_t stream = nullptr;
    uint64_t epoch = 0;
    const int32_t* identity_ids = nullptr;
    float* partial = nullptr;
    size_t partial_capacity = 0;
    const TileSpan* device_spans = nullptr;
    const int32_t* device_previous_blocks = nullptr;
    size_t span_count = 0, previous_count = 0;
    const float* tails = nullptr;
    int64_t row0 = 0, nr = 0;
    void* hq = nullptr;
    size_t hq_capacity = 0;
    float* row_sd = nullptr;
    size_t row_sd_capacity = 0;
    float* diagnostic_completed_gu = nullptr;
    int64_t diagnostic_ld = 0;
    size_t diagnostic_capacity = 0;
};
// PREPARE ONLY: original J/grid selection and dynamic-shared setup before capture.
Shape prepare_shape(const strata::prefill::mmq::Product&);
// RAW PHASE: original unpaired GU row order; final old fixup is NOT launched.
void enqueue_raw(const strata::prefill::mmq::Product&, const Shape&, const Binding&);
// FINAL PHASE: original fixup FP32/FTZ then original R56 scalar/D4 semantics.
void enqueue_completion(const strata::prefill::mmq::Product&, const Shape&, const Binding&);
// V16 host preparation only. Numeric raw/completion bodies remain V12 bytes.
struct LaunchProfile {
    int owner = -1, sm_count = 0, largest_J = 0;
    std::array<size_t,16> shared{}; // J/8-1; zero means original unsupported config
};
LaunchProfile prepare_launch_profile(); // init only: validates hardware, sets shared attributes
Shape describe_shape(const strata::prefill::mmq::Product&, const LaunchProfile&); // no allocation/attribute call
} // namespace swift_tail_r027
