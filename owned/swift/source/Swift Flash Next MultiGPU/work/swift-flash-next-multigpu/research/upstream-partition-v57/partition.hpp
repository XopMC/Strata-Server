#pragma once
#include "selector16.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cstdio>
#include <cstdlib>
namespace swift_v57 {
inline void topk(const float* scores,const int32_t* steps,int64_t nq,int64_t blocks,int64_t cap,
 const strata::kernels::QsaShapes& shape,int32_t* ids,void* stream,float* scratch,uint64_t scratch_floats){
 namespace K=strata::kernels;namespace P=swift167p16;
 static const bool enabled=[](){const char* v=std::getenv("STRATA_SWIFT_PARTITION_V57");return v&&v[0]=='1'&&v[1]=='\0';}();
 const uint64_t words=uint64_t(nq)*(P::Parts*P::Keep+P::Parts+2*P::Packed+4+cap);
 if(!enabled||nq<1||nq>8||blocks<16384||blocks>262146||cap!=2051||shape.idx_block!=4||shape.idx_top_k!=2048||shape.idx_dim!=128||shape.idx_n_head!=4||!scratch||scratch_floats<words){K::qsa_block_topk(scores,steps,nq,blocks,cap,shape,ids,stream);return;}
 // Each Verifier owns its arena. Index selection finishes before attention uses
 // these same rows, including the event-joined side-indexer stream. No globals,
 // allocations or extra GPU contexts; simultaneous verifiers use disjoint arenas.
 auto* p=reinterpret_cast<int32_t*>(scratch);auto take=[&](uint64_t n){auto* r=p;p+=n;return r;};P::Workspace v{};
 v.local_ids=take(nq*P::Parts*P::Keep);v.counts=take(nq*P::Parts);v.mapping=take(nq*P::Packed);v.packed_steps=take(nq*4);v.packed_ids=take(nq*cap);v.packed_scores=reinterpret_cast<float*>(take(nq*P::Packed));
 auto ck=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"Swift partition V57: %s\n",cudaGetErrorString(e));std::abort();}};auto cs=static_cast<cudaStream_t>(stream);
 ck(P::chunks(scores,steps,int(nq),blocks,v,0,cs));ck(P::pack(scores,steps,int(nq),blocks,v,cs));
 K::qsa_block_topk(v.packed_scores,v.packed_steps,nq,P::Packed,cap,shape,v.packed_ids,stream,P::Packed);
 ck(P::translate(steps,int(nq),cap,v,ids,cs));
}
}
