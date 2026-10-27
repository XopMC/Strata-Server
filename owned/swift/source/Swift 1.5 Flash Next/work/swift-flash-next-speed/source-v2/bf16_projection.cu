// Retain resident BF16 weights and exactly represent each finite FP16 input
// as two BF16 components. Reuse bounded native dequant scratch, no allocation.
#include "strata/prefill/gemm.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>

namespace strata::prefill {
__global__ void split_half_bf16(const uint16_t* input, uint16_t* hi, uint16_t* lo, int64_t n) {
    const int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const float x=__half2float(reinterpret_cast<const __half*>(input)[i]);
    const __nv_bfloat16 h=__float2bfloat16_rn(x);
    reinterpret_cast<__nv_bfloat16*>(hi)[i]=h;
    reinterpret_cast<__nv_bfloat16*>(lo)[i]=__float2bfloat16_rn(x-__bfloat162float(h));
}

bool swift_bf16_projection(Gemm& gm,const uint16_t* X,const uint16_t* W,float* Y,
                           int64_t T,int64_t N,int64_t K,int64_t ldy,std::string& err) {
    if(T<1||N<1||K<1||!gm.scratch()||gm.scratch_elems()<2*K) {
        err="Swift BF16 prefill: invalid shape or insufficient bounded scratch";return false;
    }
    if(ldy==0)ldy=N;
    if(ldy<N){err="Swift BF16 prefill: invalid output stride";return false;}
    const int64_t rows=std::min(T,gm.scratch_elems()/(2*K));
    uint16_t* hi=gm.scratch();uint16_t* lo=hi+rows*K;
    for(int64_t t=0;t<T;t+=rows) {
        const int64_t batch=std::min(rows,T-t),n=batch*K;
        split_half_bf16<<<(n+255)/256,256,0,(cudaStream_t)gm.stream()>>>(X+t*K,hi,lo,n);
        if(cudaPeekAtLastError()!=cudaSuccess){err="Swift BF16 prefill: split kernel failed";return false;}
        gm.bf16(hi,W,Y+t*ldy,batch,N,K,ldy,0.f);
        gm.bf16(lo,W,Y+t*ldy,batch,N,K,ldy,1.f);
    }
    return true;
}
}
