#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace strata::ds41::pair_indexer_v1 {

// One owner per ordinary prefill_batched invocation, including every pass.
// Default OFF. No state is retained by decode, verify or an MTP graph.
class Owner {
public:
    Owner(int max_sequence, int batch_capacity, int initial_position, cudaStream_t producer);
    ~Owner() noexcept;
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;

    // Metadata publication only. Called AFTER the original GPU0 key copy.
    // No DMA is issued here, including when the subsequent small call declines.
    void publish(int layer, const __nv_bfloat16* source, int first, int count);

    // False means decline BEFORE this owner's first enqueue for this call.
    // True means both complete query slices and the GPU0 output suffix joined.
    bool run(int layer, int ratio, const __nv_bfloat16* q,
             const __nv_bfloat16* source, const __nv_bfloat16* w,
             int rows, int position, int32_t* output, void* local_workspace,
             size_t local_workspace_bytes);

    // Explicitly drain before the Engine releases prefill scratch. The
    // destructor also attempts every owning stream on every exit path.
    void finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace strata::ds41::pair_indexer_v1

// Read-only cumulative process counters, n must equal 20; layout in README.
extern "C" int ds1017_pair_indexer_info_v1(uint64_t* output, int n);
