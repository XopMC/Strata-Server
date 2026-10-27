#pragma once
// Exact private replacement for the score declaration formerly supplied by
// partition.hpp. No env switch, allocation, original scorer call, or retry.
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" void* swift_score_kernel112(int);

namespace swift_score_v14 {
constexpr int MaxQueries=8;
constexpr int64_t MaxBlocks=262146;
inline void require(bool b,const char* message) {
    if(!b) throw std::runtime_error(std::string("Swift strict score V14: ")+message);
}
inline void check(cudaError_t e,const char* message) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string("Swift strict score V14: ")+message+": "+cudaGetErrorString(e));
}
inline bool overlap(const void* a,size_t na,const void* b,size_t nb) {
    const uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    require(na<=UINTPTR_MAX-x&&nb<=UINTPTR_MAX-y,"live interval overflow");
    return x<y+nb&&y<x+na;
}
} // namespace swift_score_v14

namespace swift_v59 {
// Root has already prepared typed current-run owners/streams before capture and
// enters its SelectorV7Scope. This exact-signature adapter verifies the CURRENT
// device against the actual CUDA stream owner; it does not store UUIDs, create a
// second owner registry, switch contexts, or query/inspect device step contents.
// Steps remain trusted coupled QSA records: n_kv>=0, n_bid=n_kv/4<blocks.
// Pointers/capacities are typed caller-owned leases live through graph retirement.
inline void scores(const float* pooled,const float* dead,const float* query,const int32_t* steps,
        int64_t nq,int64_t blocks,const strata::kernels::QsaShapes& shape,float* out,
        void* stream,int64_t active=-1) {
    namespace S=swift_score_v14;
    S::require(nq>=1&&nq<=S::MaxQueries&&blocks>=1&&blocks<=S::MaxBlocks,
        "nq1..8 and score stride1..262146 required");
    S::require(shape.idx_dim==128&&shape.idx_n_head==4&&shape.idx_block==4,
        "original 4-head/128-dimension/block4 geometry required");
    S::require(pooled&&dead&&query&&steps&&out&&stream,"non-null explicit owning buffers/stream required");
    S::require((reinterpret_cast<uintptr_t>(pooled)&15)==0&&
        (reinterpret_cast<uintptr_t>(dead)&15)==0&&(reinterpret_cast<uintptr_t>(query)&15)==0&&
        (reinterpret_cast<uintptr_t>(steps)&3)==0&&(reinterpret_cast<uintptr_t>(out)&3)==0,
        "original float4 inputs / scalar score alignment required");
    S::require(active<=0||active<=blocks,"active scheduling bound exceeds score stride");
    const size_t output_bytes=size_t(nq)*size_t(blocks)*sizeof(float);
    S::require(!S::overlap(out,output_bytes,pooled,size_t(blocks)*128*sizeof(float))&&
        !S::overlap(out,output_bytes,dead,128*sizeof(float))&&
        !S::overlap(out,output_bytes,query,size_t(nq)*512*sizeof(float))&&
        !S::overlap(out,output_bytes,steps,size_t(nq)*4*sizeof(int32_t)),"score output overlaps live input");
    int device=-1,stream_device=-1;
    S::check(cudaGetDevice(&device),"current owner");
    S::check(cudaStreamGetDevice(static_cast<cudaStream_t>(stream),&stream_device),"stream owner");
    S::require(device>=0&&device<2&&stream_device==device,"current admitted GPU0/1 and matching owning stream required");
    int n=int(nq);
    void* fn=swift_score_kernel112(n);
    S::require(fn!=nullptr,"frozen register leaf specialization absent");
    // This frozen leaf uses the original score expression and warp reduction.
    // active is only a trusted scheduling upper bound (when positive). The fixed
    // register grid reads live steps on EVERY replay and preserves inactive bytes.
    void* args[]={&pooled,&dead,&query,&steps,&n,&blocks,&out};
    S::check(cudaLaunchKernel(fn,dim3(256),dim3(256),args,0,static_cast<cudaStream_t>(stream)),"frozen register launch");
}
} // namespace swift_v59
