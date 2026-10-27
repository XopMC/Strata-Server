#pragma once
#include "strata/ds41/borrowed_rows.hpp"
namespace strata::ds41 {
// CPU-only full backbone admission. Dense payload remains borrowed until an
// explicitly GPU-bound Pack uploads it. This is not a vision/MTP admission.
class NativeCheckpoint {
public:
    NativeCheckpoint(std::shared_ptr<const BorrowedWeights>,const std::string& geometry);
    static void validate_geometry(const std::string&);
    const std::vector<std::string>& dense_names()const{return dense_;}
    size_t dense_arena_bytes()const{return dense_bytes_;}
    std::shared_ptr<const BorrowedWeights> weights()const{return weights_;}
    std::shared_ptr<BorrowedRows> rows()const{return rows_;}
private:
    std::shared_ptr<const BorrowedWeights> weights_;
    std::shared_ptr<BorrowedRows> rows_;
    std::vector<std::string> dense_;
    size_t dense_bytes_=0;
};
}
