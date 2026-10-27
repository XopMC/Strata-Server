#include "strata/ds41/native_routes.hpp"
#include <cmath>
#include <stdexcept>
namespace strata::ds41 {
NativeRoutePlan native_route_plan(const int32_t*ids,const float*weights,int rows,int topk,int experts){
    if(!ids||!weights||rows<1||rows>1048576||topk<1||topk>64||experts<topk||experts>4096)
        throw std::invalid_argument("invalid native route geometry");
    NativeRoutePlan result;result.offsets.assign(experts+1,0);
    const size_t n=size_t(rows)*topk;result.tokens.resize(n);result.weights.resize(n);
    for(int t=0;t<rows;t++)for(int j=0;j<topk;j++){
        const size_t at=size_t(t)*topk+j;const int e=ids[at];
        if(e<0||e>=experts||!std::isfinite(weights[at])||weights[at]<0)
            throw std::invalid_argument("invalid native expert ID or routing weight");
        for(int k=0;k<j;k++)if(ids[size_t(t)*topk+k]==e)throw std::invalid_argument("duplicate native expert in one token");
        result.offsets[e+1]++;
    }
    for(int e=1;e<=experts;e++)result.offsets[e]+=result.offsets[e-1];
    auto fill=result.offsets;
    for(int t=0;t<rows;t++)for(int j=0;j<topk;j++){
        const size_t at=size_t(t)*topk+j;const int dest=fill[ids[at]]++;
        result.tokens[dest]=t;result.weights[dest]=weights[at];
    }
    return result;
}
}
