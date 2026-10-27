#pragma once

#include "../runtime-nvfp4-reader-v1/nvfp4_projection_v1.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::ds41 {

// Uniform is exactly deepseek41-signed-uniform-bf16-group-v1, group32:
// each row contains all signed-uniform packed codes followed by BF16 scales.
// Native MXFP4 stores two E2M1 values per byte and has separate E8M0 scales.
enum class TensorEncoding { Bf16, F32, F16, F8e4m3, E8m0, I8, U8, I16, I32, I64, Mxfp4, Uniform, Output16Compact, Nvfp4 };

struct MemoryRegion {
    const uint8_t* data;
    size_t bytes;
};

struct BorrowedTensor {
    TensorEncoding encoding;
    std::vector<int64_t> shape;
    const uint8_t* data;
    size_t bytes;
    int bits = 0;
    bool protected_native_rows = false;
};

struct PackedProjection {
    const uint8_t* weights;
    const uint8_t* scales; // nullptr: BF16 scales are embedded in each Uniform row
    size_t weight_row_bytes;
    size_t scale_row_bytes;
    int rows;
    int columns;
    int bits; // 0 is the existing native MXFP4 CPU-kernel convention
};

// A descriptor for the exact borrowed CPU output16-compact-v3 allocation.
// data is never a raw row-packed weight pointer. Native E8M0 bytes are stored
// zero-extended uint16 in each tile footer; own BF16 scale bits are unchanged.
struct CompactProjectionV248 {
    const uint8_t* data;
    size_t bytes;
    int rows, columns, bits; // bits0 native MXFP4, bits3 signed Uniform
    const uint8_t* first_scale_footer() const { return data + 64 * size_t(bits ? bits : 4); }
};

struct StagedProjection {
    PackedProjection projection{};
    std::shared_ptr<std::vector<uint8_t>> weights,scales;
    size_t owned_bytes()const{return(weights?weights->size():0)+(scales?scales->size():0);}
};

struct Nvfp4Binding1017 { ds1017_nvfp4_v1::ProjectionView projection; ds1017_nvfp4_v1::OwnerLease owner; };

class BorrowedWeights {
public:
    // owner must retain every whitelisted allocation until this registry and
    // the engine's users are destroyed. Payloads are never owned/copied here.
    BorrowedWeights(std::shared_ptr<const void> owner, std::vector<MemoryRegion> regions);
    void add(const std::string& name, BorrowedTensor tensor);
    void add_segmented(const std::string& name, BorrowedTensor tensor, std::vector<MemoryRegion> segments);
    const BorrowedTensor& at(const std::string& name) const;
    // Raw access rejects compact descriptors; use this explicit metadata/data API.
    bool is_nvfp4(const std::string& prefix) const;
    Nvfp4Binding1017 nvfp4_projection(const std::string& prefix,int rows,int columns) const;
    bool is_output16_compact(const std::string& prefix) const;
    CompactProjectionV248 compact_projection(const std::string& prefix,int rows,int columns) const;
    std::vector<uint64_t> output16_info_v248() const; // schema1,count,bytes

    const uint8_t* read_range(const std::string& name,size_t offset,size_t bytes,std::vector<uint8_t>& scratch) const;
    PackedProjection projection(const std::string& prefix, int rows, int columns) const;
    // Metadata-only admission permits null pointers for segmented matrices.
    PackedProjection inspect_projection(const std::string& prefix,int rows,int columns)const;
    // Retains any bounded copies across shard boundaries; direct views stay zero-copy.
    StagedProjection stage_projection(const std::string& prefix,int rows,int columns,size_t copy_budget=64ull<<20)const;

private:
    std::shared_ptr<const void> owner_;
    std::vector<MemoryRegion> regions_;
    std::unordered_map<std::string, BorrowedTensor> tensors_;
    std::unordered_map<std::string, std::vector<MemoryRegion>> segments_;
    void check_region(MemoryRegion) const;
    const BorrowedTensor& metadata(const std::string& name) const;
};

} // namespace strata::ds41
