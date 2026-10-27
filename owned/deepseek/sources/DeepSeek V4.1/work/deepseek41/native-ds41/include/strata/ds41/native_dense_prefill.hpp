#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
namespace strata::ds41 {
class NativeDensePrefill {
    struct Impl;std::unique_ptr<Impl>impl_;
public:
    explicit NativeDensePrefill(size_t weight_bytes,int max_rows=256);
    ~NativeDensePrefill();
    void next_layer();
    bool run(const uint16_t*x,int rows,int64_t k,const uint8_t*w,const uint8_t*s,int64_t n,uint16_t*y);
};
}
