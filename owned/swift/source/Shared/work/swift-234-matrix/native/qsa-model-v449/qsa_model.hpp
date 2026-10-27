#pragma once
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace qsa_model_v449 {
struct Layer {
    int global_layer;
    const float* pooled;
    const float* dead;
    int64_t cells;
};
// Immutable owning pointers; allocation precedes every graph capture.
bool prepare(const void* owner, int device, int cards, int queries, int64_t blocks,
             cudaStream_t stream, const std::vector<Layer>& layers, std::string& error) noexcept;
// Called after the complete new/restored prompt and before its timing closes.
bool seed_request(std::string& error) noexcept;
// Exactly one outstanding window per owner. completion follows the parent drain.
bool begin_window(const void* owner, int64_t position, int queries, std::string& error) noexcept;
bool completed(const void* owner, std::string& error) noexcept;
void enqueue(const void* owner, int layer, int queries, int token_base,
             const float* pooled, const float* dead, const float* query, const int32_t* steps,
             float* scores, int32_t* ids, const strata::kernels::QsaShapes& shape, cudaStream_t stream);
// All parent graphs must already be destroyed; this positively retires helpers.
void release(const void* owner) noexcept;

struct Control { uint64_t epoch, sequence; int64_t position; int32_t queries, reserved; };
cudaError_t peer_inputs(const float*, const float*, const int32_t*, float*, float*, int32_t*,
                        const float*, float*, int64_t, const Control*, const uint64_t*, uint32_t*,
                        int, cudaStream_t);
cudaError_t return_packet(const uint32_t*, uint32_t*, int, int, cudaStream_t);
cudaError_t return_ids(const int32_t*, int32_t*, const int32_t*, int, int, cudaStream_t);
cudaError_t bounded_error_ids(int32_t*, const int32_t*, const uint32_t*, int, cudaStream_t);
}
