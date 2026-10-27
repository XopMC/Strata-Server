#pragma once
#include "selector16.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

extern "C" void* swift_score_kernel112(int);
namespace swift_v59 {
inline void scores(const float* pooled,const float* dead,const float* query,const int32_t* steps,int64_t nq,int64_t blocks,
 const strata::kernels::QsaShapes& shape,float* out,void* stream,int64_t active=-1){
 static const bool enabled=[](){const char* v=std::getenv("STRATA_SWIFT_REGISTER_V59");return v&&v[0]=='1'&&v[1]=='\0';}();
 // New immutable default-off scope: split uses only existing NT4/NT3 kernels;
 // direct uses already-compiled NT7/NT8. All numerical objects remain unchanged.
 static const int wide=[](){const char* v=std::getenv("STRATA_SWIFT_REGISTER_78_V70");if(!v||!std::strcmp(v,"off"))return 0;if(!std::strcmp(v,"split"))return 1;if(!std::strcmp(v,"direct"))return 2;if(!std::strcmp(v,"auto"))return 3;std::fprintf(stderr,"Swift register78: expected off/split/direct/auto\n");std::abort();}();
 auto original=[&](){strata::kernels::qsa_block_scores(pooled,dead,query,steps,nq,blocks,shape,out,stream,active);};
 if(!enabled||nq<3||nq>8||active>0||shape.idx_dim!=128||shape.idx_n_head!=4||shape.idx_block!=4||(nq>6&&(!wide||(wide==3&&blocks>65538)))){original();return;}
 auto launch=[&](int n,const float* q,const int32_t* st,float* dst){
  void* fn=swift_score_kernel112(n);if(!fn){std::fprintf(stderr,"Swift register78: absent existing NT%d kernel\n",n);std::abort();}
  void* args[]={&pooled,&dead,&q,&st,&n,&blocks,&dst};
  const cudaError_t e=cudaLaunchKernel(fn,dim3(256),dim3(256),args,0,static_cast<cudaStream_t>(stream));
  if(e!=cudaSuccess){std::fprintf(stderr,"Swift register78: %s\n",cudaGetErrorString(e));std::abort();}
 };
 if(nq>6&&wide){
  static thread_local unsigned seen=0;int dev=-1;
  if(cudaGetDevice(&dev)!=cudaSuccess||dev<0||dev>1){std::fprintf(stderr,"Swift register78: unauthorized device\n");std::abort();}
  const unsigned bit=1u<<(dev*8+int(nq)-1);
  if(!(seen&bit)){seen|=bit;std::fprintf(stderr,"Swift register78 actual CUDA%d: nq%lld blocks%lld, %s, original scoring/topk arithmetic\n",dev,(long long)nq,(long long)blocks,(wide==1||wide==3)?"split4+tail":"direct");}
 }
 if(nq>6&&(wide==1||wide==3)){launch(4,query,steps,out);launch(int(nq)-4,query+4*512,steps+4*4,out+4*blocks);}
 else launch(int(nq),query,steps,out);
}
}
