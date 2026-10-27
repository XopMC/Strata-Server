#pragma once
#include "strata/ds41/borrowed_weights.hpp"
#include <array>

namespace strata::ds41 {
// Serial complete MoE evaluator. Outputs are BF16 [rows,topk,dim], sorted by
// expert ID within each row, matching source accumulation order. A8 is an
// explicit arithmetic choice; native F32 never silently accepts own quant.
class NativeExperts {
public:
    NativeExperts(std::shared_ptr<const BorrowedWeights> weights,
                  const std::string& task_library, const std::string& pool_library,
                  bool a8, std::vector<int> cpus, int coordinator_cpu,bool mixed_f32=false);
    ~NativeExperts();
    NativeExperts(const NativeExperts&) = delete;
    NativeExperts& operator=(const NativeExperts&) = delete;
    void add_layer(int layer, const std::string& prefix, int experts,
                   int dim, int inter, int topk, float swiglu_limit);
    std::vector<uint16_t> run(int layer, int rows, const std::vector<uint16_t>& input,
                             const std::vector<int64_t>& ids, const std::vector<float>& routing);
    std::vector<uint16_t> run_selected(int layer,int rows,const std::vector<uint16_t>&input,const std::vector<int64_t>&ids,const std::vector<float>&routing,int topk);
    std::vector<uint64_t> worker_rows() const;
    void build_gu_cache_v19(size_t bytes,const std::vector<std::array<uint64_t,384>>&frequency);
    std::array<uint64_t,32> gu_statistics_v19() const;
    void prepare_batch(int rows);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
