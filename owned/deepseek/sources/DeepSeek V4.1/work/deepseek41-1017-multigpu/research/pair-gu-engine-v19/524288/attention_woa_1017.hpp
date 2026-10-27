#pragma once
#include "strata/ds41/ops.hpp"
#include "strata/ds41/config.hpp"
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::ds41::attention_woa_1017 {
// Source-only opt-in, NOT a model or operator admission receipt.
inline void require_opt_in() {
    static const bool enabled=[] {
        const char* s=std::getenv("DS41_ATTENTION_WOA_1017_SOURCE_PORT");
        return s && !std::strcmp(s,"1");
    }();
    if(!enabled)throw std::runtime_error("Private1017 attention boundary defaultOFF; Root proof required");
}
inline void quantize(ops::bf16* output,int64_t tokens,cudaStream_t stream) {
    require_opt_in();
    static_assert(kHeads*kHeadDim==8*4096);
    static_assert((4096%32)==0);
    if(!output || tokens<1 || tokens>INT_MAX/(kHeads*kHeadDim))
        throw std::invalid_argument("Invalid bounded1017 output rows");
    // Flattening token/group axes preserves every original4096 row's K32 groups.
    ops::act_quant_inplace(output,int(tokens*kHeads*kHeadDim),stream);
}
}
