#pragma once
#include "strata/ds41/borrowed_weights.hpp"
#include <map>
namespace strata::ds41 {
// Bounded CPU row gather. Engram tables stay packed in their original RAM
// allocations; only requested rows are decoded. No table-sized float copy.
class BorrowedRows {
public:
    explicit BorrowedRows(std::shared_ptr<const BorrowedWeights> weights);
    void protect(const std::string& tensor,const std::string& ids,
                 const std::string& native_weight,const std::string& native_scale);
    std::vector<uint16_t> gather_bf16(const std::string& tensor,const std::vector<int64_t>& ids,
                                    size_t maximum_bytes=64ull<<20) const;
private:
    struct Overlay {std::string ids,weight,scale;};
    std::shared_ptr<const BorrowedWeights> weights_;
    std::map<std::string,Overlay> overlays_;
};
}
