#pragma once
#include "selector16.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace swift_pp_v77 {
inline void topk(const float* scores,const int32_t* steps,int64_t nq,int64_t blocks,int64_t cap,
 const strata::kernels::QsaShapes& shape,int32_t* ids,void* stream,int64_t active,
 float* scratch,uint64_t scratch_floats){
 namespace K=strata::kernels;namespace P=swift167p16;
 static const bool enabled=[](){const char* v=std::getenv("STRATA_SWIFT_PP_P16_V77");return v&&v[0]=='1'&&v[1]=='\0';}();
 static const bool checked=[](){const char* v=std::getenv("STRATA_SWIFT_PP_P16_CHECK_V77");return v&&v[0]=='1'&&v[1]=='\0';}();
 constexpr uint64_t words=P::Parts*P::Keep+P::Parts+2*P::Packed+4+2051;
 const int64_t maxq=std::min<int64_t>(256,scratch_floats/words);
 if(!enabled||nq<=16||nq>256||blocks!=262146||active<147456||active>blocks||cap!=2051||
    shape.idx_block!=4||shape.idx_top_k!=2048||shape.idx_dim!=128||shape.idx_n_head!=4||!scratch||maxq<1){
  K::qsa_block_topk(scores,steps,nq,blocks,cap,shape,ids,stream,active);return;
 }
 // Existing per-Prefill attention workspace is idle until ALL index selections finish on this stream.
 // No allocation and no global device pointers; the two GPU stages use disjoint arenas.
 auto ck=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"PP P16 V77: %s\n",cudaGetErrorString(e));std::abort();}};
 int device=-1;ck(cudaGetDevice(&device));if(device<0||device>=2)std::abort();
 static std::atomic<unsigned> calls[2]{};const unsigned call=calls[device].fetch_add(1,std::memory_order_relaxed);
 if(call==0)std::fprintf(stderr,"PP P16 V77 CUDA%d: exact16part514, active>=147456, maxq=%lld, borrowed existing scratch=%llu bytes, no allocation\n",device,(long long)maxq,(unsigned long long)(scratch_floats*4));
 auto cs=static_cast<cudaStream_t>(stream);
 for(int64_t at=0;at<nq;at+=maxq){
  const int n=int(std::min<int64_t>(maxq,nq-at));auto* p=reinterpret_cast<int32_t*>(scratch);
  auto take=[&](uint64_t z){auto* r=p;p+=z;return r;};P::Workspace v{};
  v.local_ids=take(n*P::Parts*P::Keep);v.counts=take(n*P::Parts);v.mapping=take(n*P::Packed);
  v.packed_steps=take(n*4);v.packed_ids=take(n*cap);v.packed_scores=reinterpret_cast<float*>(take(n*P::Packed));
  const auto* st=steps+at*4;const auto* sc=scores+at*blocks;auto* dst=ids+at*cap;
  ck(P::chunks(sc,st,n,blocks,v,0,cs));ck(P::pack(sc,st,n,blocks,v,cs));
  K::qsa_block_topk(v.packed_scores,v.packed_steps,n,P::Packed,cap,shape,v.packed_ids,stream,P::Packed);
  ck(P::translate(st,n,cap,v,dst,cs));
  if(checked&&call<2){
   K::qsa_block_topk(sc,st,n,blocks,cap,shape,v.packed_ids,stream,active);
   std::vector<int32_t>a(size_t(n)*cap),b(a.size());ck(cudaMemcpyAsync(a.data(),dst,a.size()*4,cudaMemcpyDeviceToHost,cs));
   ck(cudaMemcpyAsync(b.data(),v.packed_ids,b.size()*4,cudaMemcpyDeviceToHost,cs));ck(cudaStreamSynchronize(cs));
   // Compare active outputs only; each selector deliberately leaves inactive rows untouched.
   std::vector<int32_t>hs(size_t(n)*4);ck(cudaMemcpy(hs.data(),st,hs.size()*4,cudaMemcpyDeviceToHost));
   for(int q=0;q<n;++q)for(int j=0;j<hs[q*4+3];++j)if(a[size_t(q)*cap+j]!=b[size_t(q)*cap+j]){
    std::fprintf(stderr,"PP P16 V77 mismatch GPU%d query%d cell%d\n",device,q,j);std::abort();
   }
   std::fprintf(stderr,"PP P16 V77 CUDA%d: actual model scores/IDs exact, q=%d\n",device,n);
  }
 }
}
}
