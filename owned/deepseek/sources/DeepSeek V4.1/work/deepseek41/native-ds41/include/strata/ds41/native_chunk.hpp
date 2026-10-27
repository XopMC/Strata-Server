#pragma once
#include <cstddef>
#include <stdexcept>
namespace strata::ds41 {
// Size(cap) must be a monotone scratch-layout function. Preserve a completely
// fitting request; otherwise use the largest fitting whole attention batch.
// Avoid halving down just below n/2 and paying for an almost-empty third pass.
template<class Size>
int native_chunk_capacity(int requested,int sub,size_t free,size_t reserve,Size bytes){
    if(requested<1||requested>1048576||sub<1||sub>requested||free<reserve)
        throw std::invalid_argument("invalid native prefill memory capacity");
    const size_t budget=free-reserve;
    auto fits=[&](int cap){return bytes(cap)<=budget;};
    if(fits(requested))return requested;
    if(!fits(sub))throw std::runtime_error("native prefill cannot fit one attention batch with reserve");
    int lo=1,hi=requested/sub;
    while(lo<hi){
        const int mid=lo+(hi-lo+1)/2;
        if(fits(mid*sub))lo=mid;else hi=mid-1;
    }
    return lo*sub;
}
}
