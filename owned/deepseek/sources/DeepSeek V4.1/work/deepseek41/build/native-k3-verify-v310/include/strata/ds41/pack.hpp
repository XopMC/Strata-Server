// include/strata/ds41/pack.hpp - the ds41 pack as the engine sees it (written by tools/ds41/pack.py).
//
// The loader reads only flat text indexes; it never re-derives a layout (same rule as upstream's weights.hpp).
//   dense:   every non-expert tensor copied into one device arena, looked up by name
//   experts: experts.bin mapped read-only; each expert is a 4 KiB aligned slot with 12 components
//   engram:  the two n-gram tables stay in their original files; rows are read on demand
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>
#include "strata/ds41/native_checkpoint.hpp"

namespace strata::ds41 {

enum class DType { F8E4M3, E8M0, BF16, F32, F16, I32, I16, I8, U8 };

struct DenseTensor {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
    void* device = nullptr;          ///< inside the arena, after upload
};

/// One routed expert: offsets are inside its slot.
struct ExpertSlot {
    uint64_t offset = 0;             ///< in experts.bin
    uint64_t bytes = 0;
    float bits[3] = {0, 0, 0};       ///< w1, w3, w2
    uint64_t comp_off[12] = {};      ///< w1.trellis w1.suh w1.svh w1.mul1, then w3, then w2
    uint64_t comp_bytes[12] = {};
};

struct EngramTable {
    int layer = 0;
    int64_t rows = 0;
    int dim = 0;
    uint64_t weight_offset = 0;
    uint64_t scale_offset = 0;
    std::string path;
};

struct EngramHash {
    int max_ngram = 0;
    int n_heads = 0;
    int64_t pad = 0;
    std::vector<int> layers;                        ///< backbone layer of each engram table
    std::vector<std::vector<int64_t>> multipliers;  ///< [table][max_ngram]
    std::vector<std::vector<int64_t>> primes;       ///< [table][(max_ngram-1)*n_heads]
    std::vector<std::vector<int64_t>> offsets;      ///< [table][(max_ngram-1)*n_heads]
    std::vector<int32_t> token_map;                 ///< vocab -> compressed id
};

struct DenseStreamStateV190;

class Pack {
public:
    /// Parses the indexes and checks that the pack is finished. Throws on any inconsistency.
    explicit Pack(const std::string& dir);
    Pack(std::shared_ptr<NativeCheckpoint> checkpoint,EngramHash hash);
    ~Pack();
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;

    /// Copies dense.bin into one device arena. Returns the arena size in bytes.
    uint64_t upload_dense();
    // Experimental190: default OFF. Enable only before the first upload_dense().
    // Stable base contains non-layer tensors plus every layer .engram. tensor.
    // Only ordinary layer tensors use the two bounded banks during prefill.
    void configure_dense_stream_v190(bool enabled);
    bool dense_stream_enabled_v190() const noexcept;
    bool dense_stream_active_v190() const noexcept;
    void begin_dense_stream_v190();
    void prefetch_dense_layer_v190(int layer);
    void activate_dense_layer_v190(int layer, cudaStream_t consumer = nullptr);
    void retire_dense_layer_v190(int layer, cudaStream_t consumer = nullptr);
    // Caller must first retire every active layer and free its prefill scratch.
    // Afterwards all Engine layer pointers must be rebound before graph capture.
    void end_dense_stream_v190();
    // [schema,enabled,active,begin_count,restore_count,base_alloc,layer_alloc,
    // bank_capacity,current_GPU_banks,current_pinned_banks,current_dense_GPU,
    // last_reclaimed_GPU,layer_upload_bytes,layer_upload_count,prefetch_calls,
    // activate_calls,retire_calls,restore_bytes,restore_tensor_copies,
    // free_before_begin,free_after_begin,free_after_restore,scratch_peak,failed].
    std::array<uint64_t, 24> dense_stream_info_v190() const noexcept;

    /// Maps experts.bin read-only (MAP_SHARED: pages come from the OS page cache, as upstream's mmap tier).
    void map_experts();

    const DenseTensor& dense(const std::string& name) const;
    bool has_dense(const std::string& name) const { return dense_.count(name) != 0; }
    const ExpertSlot& expert(int layer, int e) const { return experts_[(size_t) layer * n_experts_ + e]; }
    const uint8_t* expert_base() const { return experts_map_; }
    const std::vector<EngramTable>& engram_tables() const { return engram_; }
    const EngramHash& engram_hash() const { return hash_; }
    int n_layers() const { return n_layers_; }
    int n_experts() const { return n_experts_; }
    const std::string& dir() const { return dir_; }
    const std::shared_ptr<NativeCheckpoint>& native()const{return native_;}

private:
    std::string dir_;
    int n_layers_ = 0;
    int n_experts_ = 0;
    std::map<std::string, DenseTensor> dense_;
    std::vector<ExpertSlot> experts_;
    std::vector<EngramTable> engram_;
    EngramHash hash_;
    void* arena_ = nullptr;
    const uint8_t* experts_map_ = nullptr;
    uint64_t experts_map_bytes_ = 0;
    std::shared_ptr<NativeCheckpoint> native_;
    // Appended: preserve original field offsets; all Pack sizeof consumers rebuild.
    std::unique_ptr<DenseStreamStateV190> dense_stream_v190_;
};

size_t dtype_size(DType t);

}  // namespace strata::ds41
