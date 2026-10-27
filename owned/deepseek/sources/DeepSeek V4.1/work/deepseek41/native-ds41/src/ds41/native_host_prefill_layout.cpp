#include "strata/ds41/native_host_prefill_layout.hpp"
#include <stdexcept>
namespace strata::ds41 {
NativeHostPrefillLayout native_host_prefill_layout(int rows,size_t budget,size_t available,size_t reserve){
    if(rows<1||rows>1048576)throw std::invalid_argument("host prefill rows outside1..1M");
    const size_t n=rows;
    NativeHostPrefillLayout p{n*4*5120*2,n*5120*4,n*512*4,n*2048*4,0,0};
    p.core_bytes=p.hc_bytes+p.routed_bytes+p.topk_bytes+p.candidate_bytes;
    p.admitted_bytes=p.core_bytes+(size_t(2)<<30);
    if(p.admitted_bytes>budget||reserve>available||p.admitted_bytes>available-reserve)
        throw std::invalid_argument("host prefill would exceed explicit budget or RAM reserve");
    return p;
}
}
