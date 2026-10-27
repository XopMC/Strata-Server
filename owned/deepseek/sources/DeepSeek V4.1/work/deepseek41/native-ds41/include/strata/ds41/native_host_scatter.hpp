#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace strata::ds41 {
// Persistent explicitly pinned workers. The caller owns the FP32 accumulator
// for the object's lifetime, and completed GPU-copy packets until add returns.
// Expert order and unique ascending row IDs enforce deterministic accumulation.
class NativeHostScatter {
public:
    NativeHostScatter(float*out,size_t elements,int rows,int dim,std::vector<int>cpus);
    ~NativeHostScatter();
    NativeHostScatter(const NativeHostScatter&)=delete;
    NativeHostScatter&operator=(const NativeHostScatter&)=delete;
    void begin_layer();
    void add(int expert,const std::vector<int32_t>&rows,const std::vector<uint16_t>&values);
    void add_view(int expert,const int32_t*rows,size_t count,const uint16_t*values,size_t elements);
    std::vector<uint64_t> worker_rows()const;
private:
    struct Impl;std::unique_ptr<Impl>impl_;
};
}
