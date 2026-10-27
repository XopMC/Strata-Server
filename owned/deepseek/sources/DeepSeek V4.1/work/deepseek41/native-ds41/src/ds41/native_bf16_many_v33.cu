#include "../../../../../../../include/strata_ds/fresh_diagnostic_visibility.hpp"
#include "strata/ds41/native_bf16_many.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::ds41::ops {
namespace {
template<int M>
__global__ void bf16_many(const bf16* __restrict__ x, const bf16* __restrict__ w,
                         int64_t k, int64_t n, float* __restrict__ y) {
    const int64_t row = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n) return;
    const bf16* wr = w + row*k;
    float acc[M] = {};
    for (int64_t i=lane; i<k; i+=32) {
        const float weight = __bfloat162float(wr[i]);
#pragma unroll
        for (int t=0; t<M; ++t)
            acc[t] = __fmaf_rn(__bfloat162float(x[int64_t(t)*k+i]), weight, acc[t]);
    }
#pragma unroll
    for (int t=0; t<M; ++t) {
        for (int off=16; off>0; off>>=1)
            acc[t] = __fadd_rn(acc[t], __shfl_down_sync(0xffffffffu, acc[t], off));
        if (lane==0) y[int64_t(t)*n+row] = acc[t];
    }
}
}
void bf16_linear_many(const bf16* x, const bf16* w, int m, int64_t k,
                      int64_t n, float* y, cudaStream_t stream) {
    if (!x || !w || !y || m<1 || m>8 || k<32 || k>65536 || k%32 || n<1 || n>129280)
        throw std::invalid_argument("bounded BF16 multirow geometry required");
    if (m==1) {bf16_linear(x, nullptr, w, k, n, nullptr, y, stream); return;}
    const unsigned grid=unsigned((n+7)/8);
#define RUN(M) case M: bf16_many<M><<<grid,256,0,stream>>>(x,w,k,n,y); break
    switch (m) {RUN(2);RUN(3);RUN(4);RUN(5);RUN(6);RUN(7);RUN(8);}
#undef RUN
    if (cudaGetLastError()!=cudaSuccess) throw std::runtime_error("BF16 multirow launch failed");
}
}

// Focused component probe; never selects or opens another physical card.
extern "C" int ds41_native_bf16_many_probe(const uint16_t* x,const uint16_t* w,
        int m,int k,int n,float* y,int serial) {
    if(!strata::ds41::publication::metadata_snapshot(1) ||
       !x || !w || !y || m<1 || m>8 || k<32 || k>65536 || k%32 || n<1 || n>129280 || (serial!=0&&serial!=1)) return 1;
    try {
        int device=-1;
        if(cudaGetDevice(&device)!=cudaSuccess || device!=0 || !strata::ds41::publication::runtime_snapshot(1)) return 2;
        const auto* xb=reinterpret_cast<const strata::ds41::ops::bf16*>(x);
        const auto* wb=reinterpret_cast<const strata::ds41::ops::bf16*>(w);
        if(serial) for(int t=0;t<m;t++)
            strata::ds41::ops::bf16_linear(xb+size_t(t)*k,nullptr,wb,k,n,nullptr,y+size_t(t)*n);
        else strata::ds41::ops::bf16_linear_many(xb,wb,m,k,n,y);
        return 0;
    } catch (...) {return 3;}
}
