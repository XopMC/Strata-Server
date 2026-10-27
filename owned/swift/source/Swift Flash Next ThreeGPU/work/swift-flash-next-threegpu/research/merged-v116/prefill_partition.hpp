#pragma once
#include "selector16.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Keep the V77 call-site namespace/signature. Only host admission is generalized;
// selector16.cu is unchanged and remains constrained to max_blocks<=262146.
namespace swift_pp_v77 {
namespace P=swift167p16;
namespace K=strata::kernels;
constexpr uint64_t workspace_words_v88=P::Parts*P::Keep+P::Parts+2*P::Packed+4+2051;
inline int64_t max_queries_v88(uint64_t scratch_floats) {
    return std::min<int64_t>(256,scratch_floats/workspace_words_v88);
}
inline bool eligible_v88(int64_t nq,int64_t blocks,int64_t cap,const K::QsaShapes& shape,
                          int64_t active,const float* scratch,uint64_t scratch_floats,
                          int64_t minimum_active=32768) {
    return nq>16&&nq<=256&&(blocks==131074||blocks==262146)&&active>=minimum_active&&active<=blocks&&cap==2051&&
           shape.idx_block==4&&shape.idx_top_k==2048&&shape.idx_dim==128&&shape.idx_n_head==4&&
           scratch&&max_queries_v88(scratch_floats)>0;
}
// Explicit candidate entry for an independent captured component: no getenv,
// device queries, logging, host readback or allocation on this path.
inline cudaError_t select16_v88(const float* scores,const int32_t* steps,int64_t nq,int64_t blocks,int64_t cap,
    const K::QsaShapes& shape,int32_t* ids,cudaStream_t cs,int64_t active,float* scratch,uint64_t scratch_floats) {
    if(!scores||!steps||!ids||!eligible_v88(nq,blocks,cap,shape,active,scratch,scratch_floats))return cudaErrorInvalidValue;
    const int64_t maxq=max_queries_v88(scratch_floats);
    for(int64_t at=0;at<nq;at+=maxq) {
        const int n=int(std::min<int64_t>(maxq,nq-at));auto* p=reinterpret_cast<int32_t*>(scratch);
        auto take=[&](uint64_t z){auto* r=p;p+=z;return r;};P::Workspace v{};
        v.local_ids=take(n*P::Parts*P::Keep);v.counts=take(n*P::Parts);v.mapping=take(n*P::Packed);
        v.packed_steps=take(n*4);v.packed_ids=take(n*cap);v.packed_scores=reinterpret_cast<float*>(take(n*P::Packed));
        if(uint64_t(p-reinterpret_cast<int32_t*>(scratch))>scratch_floats)return cudaErrorInvalidValue;
        const auto* st=steps+at*4;const auto* sc=scores+at*blocks;auto* dst=ids+at*cap;
        cudaError_t e=P::chunks(sc,st,n,blocks,v,0,cs);if(e!=cudaSuccess)return e;
        e=P::pack(sc,st,n,blocks,v,cs);if(e!=cudaSuccess)return e;
        K::qsa_block_topk(v.packed_scores,v.packed_steps,n,P::Packed,cap,shape,v.packed_ids,cs,P::Packed);
        e=cudaGetLastError();if(e!=cudaSuccess)return e;
        e=P::translate(st,n,cap,v,dst,cs);if(e!=cudaSuccess)return e;
    }
    return cudaSuccess;
}
inline void topk(const float* scores,const int32_t* steps,int64_t nq,int64_t blocks,int64_t cap,
 const K::QsaShapes& shape,int32_t* ids,void* stream,int64_t active,float* scratch,uint64_t scratch_floats) {
    static const bool enabled=[] {const char* v=std::getenv("STRATA_SWIFT_PP_P16_V77");return v&&v[0]=='1'&&v[1]=='\0';}();
    static const bool checked=[] {const char* v=std::getenv("STRATA_SWIFT_PP_P16_CHECK_V77");return v&&v[0]=='1'&&v[1]=='\0';}();
    static const int64_t minimum_active=[] {
        const char* v=std::getenv("STRATA_SWIFT_PP_P16_MIN_ACTIVE_V88");
        if(!v)return int64_t(32768);
        char* end=nullptr;const long long z=std::strtoll(v,&end,10);
        if(end==v||!end||*end!='\0'||z<32768||z>262146){std::fprintf(stderr,"PP P16 V88: invalid active threshold\n");std::abort();}
        return int64_t(z);
    }();
    if(!enabled||!eligible_v88(nq,blocks,cap,shape,active,scratch,scratch_floats,minimum_active)) {
        K::qsa_block_topk(scores,steps,nq,blocks,cap,shape,ids,stream,active);return;
    }
    auto ck=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"PP P16 V88: %s\n",cudaGetErrorString(e));std::abort();}};
    int device=-1;ck(cudaGetDevice(&device));if(device<0||device>=2)std::abort();
    static std::atomic<unsigned> calls[2]{};const unsigned call=calls[device].fetch_add(1,std::memory_order_relaxed);
    if(call==0)std::fprintf(stderr,"PP P16 V88 CUDA%d: exact16part514, active>=%lld, maxq=%lld, existing scratch=%llu bytes, no allocation\n",
        device,(long long)minimum_active,(long long)max_queries_v88(scratch_floats),(unsigned long long)(scratch_floats*4));
    auto cs=static_cast<cudaStream_t>(stream);
    ck(select16_v88(scores,steps,nq,blocks,cap,shape,ids,cs,active,scratch,scratch_floats));
    if(checked&&call<2) {
        // Scratch is idle after translate. Reuse its first maxq*cap words as the
        // original selector's output; the final IDs live in a disjoint buffer.
        auto* reference=reinterpret_cast<int32_t*>(scratch);
        std::vector<int32_t>a(size_t(nq)*cap),b(a.size()),hs(size_t(nq)*4);
        ck(cudaMemcpyAsync(a.data(),ids,a.size()*4,cudaMemcpyDeviceToHost,cs));
        ck(cudaMemcpyAsync(hs.data(),steps,hs.size()*4,cudaMemcpyDeviceToHost,cs));
        const int64_t maxq=max_queries_v88(scratch_floats);
        for(int64_t at=0;at<nq;at+=maxq) {
            const int64_t n=std::min<int64_t>(maxq,nq-at);
            K::qsa_block_topk(scores+at*blocks,steps+at*4,n,blocks,cap,shape,reference,cs,active);
            ck(cudaMemcpyAsync(b.data()+at*cap,reference,size_t(n)*cap*4,cudaMemcpyDeviceToHost,cs));
        }
        ck(cudaStreamSynchronize(cs));
        for(int64_t q=0;q<nq;++q)for(int j=0;j<hs[q*4+3];++j)if(a[size_t(q)*cap+j]!=b[size_t(q)*cap+j]) {
            std::fprintf(stderr,"PP P16 V88 mismatch GPU%d query%lld cell%d\n",device,(long long)q,j);std::abort();
        }
        std::fprintf(stderr,"PP P16 V88 CUDA%d: actual model IDs exact, q=%lld blocks=%lld active=%lld\n",device,(long long)nq,(long long)blocks,(long long)active);
    }
}
}
