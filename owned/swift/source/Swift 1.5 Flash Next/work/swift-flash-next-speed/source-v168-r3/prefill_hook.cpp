// Private executable-local hooks. No frozen class layout/archive changes.
#include "../source-v142/selector.hpp"
#include "../source-v167/selector16.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
namespace K=strata::kernels;
extern "C" void real_topk(const float*,const int32_t*,int64_t,int64_t,int64_t,const K::QsaShapes&,int32_t*,void*,int64_t) asm("__real__ZN6strata7kernels14qsa_block_topkEPKfPKilllRKNS0_9QsaShapesEPiPvl");
extern "C" void wrap_topk(const float*,const int32_t*,int64_t,int64_t,int64_t,const K::QsaShapes&,int32_t*,void*,int64_t) asm("__wrap__ZN6strata7kernels14qsa_block_topkEPKfPKilllRKNS0_9QsaShapesEPiPvl");
extern "C" bool real_run(void*,const int64_t*,int64_t,int64_t,std::string&) asm("__real__ZN6strata7prefill7Prefill3runEPKlllRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
extern "C" bool wrap_run(void*,const int64_t*,int64_t,int64_t,std::string&) asm("__wrap__ZN6strata7prefill7Prefill3runEPKlllRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
namespace {
constexpr int Q=256,Cap=2051;constexpr int64_t Gate=147456;
constexpr size_t Bytes=4ull*Q*(swift142::Parts*swift142::Keep+swift142::Parts+2*swift142::Packed+4+Cap);
[[noreturn]] void fail(const char* why,cudaError_t e=cudaSuccess){std::fprintf(stderr,"Swift PP143 FATAL: %s (%s)\n",why,cudaGetErrorString(e));std::abort();}
void ck(cudaError_t e,const char* s){if(e!=cudaSuccess)fail(s,e);}
bool flag(const char* key){const char* p=std::getenv(key);if(!p||!p[0]||!std::strcmp(p,"0"))return false;if(std::strcmp(p,"1"))fail("flags must be0or1");return true;}
bool enabled(){static const bool b=flag("STRATA_SWIFT_PP_PARTITION143");return b;}
bool checked(){static const bool b=flag("STRATA_SWIFT_PP_PARTITION143_CHECK");return b;}
struct State {bool active=false,fallback=false;void* raw=nullptr;cudaStream_t stream=nullptr;swift142::Workspace w{};unsigned calls=0,comparisons=0;};
thread_local State state;
void device(){static thread_local bool done=false;if(done)return;int n=0;ck(cudaGetDeviceCount(&n),"device count");if(n!=1)fail("one visible device required");cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0),"properties");const unsigned char u[]={0x7c,0x89,0xb7,0x99,0x91,0x49,0x74,0x94,0x39,0x90,0x02,0x81,0x3f,0x9a,0x57,0x75};if(std::memcmp(u,p.uuid.bytes,16)||p.major!=8||p.minor!=9)fail("exact GPU0 SM89 required");done=true;}
bool allocate(){
 if(state.raw)return true;if(state.fallback)return false;device();size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"memory budget");
 if(free<Bytes+(64ull<<20)){state.fallback=true;std::fprintf(stderr,"Swift PP143: fallback original selector, free=%zu required=%zu\n",free,Bytes);return false;}
 ck(cudaMalloc(&state.raw,Bytes),"bounded prefill-only scratch");auto* p=static_cast<int32_t*>(state.raw);
 auto take=[&](size_t n){auto* out=p;p+=n;return out;};
 state.w.local_ids=take(Q*swift142::Parts*swift142::Keep);state.w.counts=take(Q*swift142::Parts);state.w.mapping=take(Q*swift142::Packed);state.w.packed_steps=take(Q*4);state.w.packed_ids=take(Q*Cap);state.w.packed_scores=reinterpret_cast<float*>(take(Q*swift142::Packed));
 if(reinterpret_cast<char*>(p)-static_cast<char*>(state.raw)!=Bytes)fail("scratch layout");
 std::fprintf(stderr,"Swift PP143: exact16part514block selector active, gate=%lld blocks scratch=%zu, prefill-only\n",(long long)Gate,Bytes);return true;
}
void cleanup(){
 const unsigned calls=state.calls,comparisons=state.comparisons;const bool fallback=state.fallback;
 if(state.raw){ck(cudaStreamSynchronize(state.stream),"drain prefill scratch users");ck(cudaFree(state.raw),"free prefill-only scratch");}
 state=State{};std::fprintf(stderr,"Swift PP143: prefill scope complete calls=%u checked=%u fallback=%d scratch_released=1\n",calls,comparisons,int(fallback));
}
struct Guard{~Guard(){cleanup();}};
}

namespace decode168 {
constexpr int Q=8,Cap=2051;
constexpr size_t Bytes=4ull*Q*(swift167p16::Parts*swift167p16::Keep+swift167p16::Parts+2*swift167p16::Packed+4+Cap);
struct State {void* raw=nullptr;swift167p16::Workspace w{};uint64_t *host=nullptr,*gpu=nullptr;unsigned captured=0,mask=0;bool ready=false;};
// N1 is serialized by the native main loop. Every graph shares this process-life
// scratch; no allocation, host synchronization or object-layout change in capture.
thread_local State d;
bool on(){static const bool b=flag("STRATA_SWIFT_DK_PART168");return b;}
bool check(){static const bool b=flag("STRATA_SWIFT_DK_PART168_CHECK");return b;}
}
extern "C" cudaError_t swift168_check_ids(const int32_t*,const int32_t*,const int32_t*,int,int64_t,uint64_t*,cudaStream_t);
extern "C" bool swift168_prepare(bool allowed){
 using namespace decode168;if(!on())return true;
 if(!allowed){std::fprintf(stderr,"Swift DK168: N1/singleGPU only\n");return false;}
 if(d.ready)return true;device();size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"DK168 memory budget");
 if(free<decode168::Bytes+(64ull<<20)){std::fprintf(stderr,"Swift DK168: budget fallback original selector free=%zu required=%zu\n",free,decode168::Bytes);return true;}
 ck(cudaMalloc(&d.raw,decode168::Bytes),"DK168 process-life scratch");auto* p=static_cast<int32_t*>(d.raw);auto take=[&](size_t n){auto* r=p;p+=n;return r;};
 d.w.local_ids=take(decode168::Q*swift167p16::Parts*swift167p16::Keep);d.w.counts=take(decode168::Q*swift167p16::Parts);d.w.mapping=take(decode168::Q*swift167p16::Packed);d.w.packed_steps=take(decode168::Q*4);d.w.packed_ids=take(decode168::Q*decode168::Cap);d.w.packed_scores=reinterpret_cast<float*>(take(decode168::Q*swift167p16::Packed));
 if(reinterpret_cast<char*>(p)-static_cast<char*>(d.raw)!=decode168::Bytes)fail("DK168 scratch layout");
 if(check()){ck(cudaHostAlloc(reinterpret_cast<void**>(&d.host),4*sizeof(uint64_t),cudaHostAllocMapped),"DK168 mapped check receipt");std::memset(d.host,0,4*sizeof(uint64_t));ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d.gpu),d.host,0),"DK168 check device address");}
 d.ready=true;std::fprintf(stderr,"Swift DK168: prepared exact16part514 selector scratch=%zu check=%d after expert cache fill; process-life N1 pool\n",decode168::Bytes,int(check()));return true;
}
extern "C" void swift168_report(){
 using namespace decode168;if(!on())return;
 if(check()&&d.ready){ck(cudaDeviceSynchronize(),"DK168 diagnostic receipt completion");auto* h=static_cast<volatile uint64_t*>(d.host);
  std::fprintf(stderr,"Swift DK168: actual_checked_calls=%llu actual_checked_IDs=%llu mismatched_IDs=%llu invalid_widths=%llu\n",(unsigned long long)h[0],(unsigned long long)h[1],(unsigned long long)h[2],(unsigned long long)h[3]);
  if(h[2]||h[3])fail("DK168 actual selected IDs differ from original");
 }
 std::fprintf(stderr,"Swift DK168: ready=%d captured_calls=%u captured_T_mask=%u scratch=%zu process_life=1\n",int(d.ready),d.captured,d.mask,d.ready?decode168::Bytes:0);
}
extern "C" bool wrap_run(void* self,const int64_t* tokens,int64_t n,int64_t pos,std::string& err){
 if(!enabled())return real_run(self,tokens,n,pos,err);
 if(state.active)fail("nested prefill not admitted");state.active=true;Guard g;return real_run(self,tokens,n,pos,err);
}
extern "C" void wrap_topk(const float* scores,const int32_t* steps,int64_t nq,int64_t max_blocks,int64_t cap,const K::QsaShapes& shape,int32_t* ids,void* stream,int64_t active_blocks){
 if(decode168::on()&&decode168::d.ready&&!state.active&&nq>=1&&nq<=8&&active_blocks<0&&
    (max_blocks==65538||max_blocks==131074||max_blocks==262146)&&cap==Cap&&shape.idx_block==4&&shape.idx_top_k==2048&&shape.idx_dim==128&&shape.idx_n_head==4){
  auto cs=static_cast<cudaStream_t>(stream);auto& d=decode168::d;
  ck(swift167p16::chunks(scores,steps,int(nq),max_blocks,d.w,0,cs),"DK168 local exact candidates");
  ck(swift167p16::pack(scores,steps,int(nq),max_blocks,d.w,cs),"DK168 compact");
  real_topk(d.w.packed_scores,d.w.packed_steps,nq,swift167p16::Packed,cap,shape,d.w.packed_ids,stream,swift167p16::Packed);
  ck(swift167p16::translate(steps,int(nq),cap,d.w,ids,cs),"DK168 original global cell IDs");
  if(decode168::check()){
   real_topk(scores,steps,nq,max_blocks,cap,shape,d.w.packed_ids,stream,active_blocks);
   ck(swift168_check_ids(ids,d.w.packed_ids,steps,int(nq),cap,d.gpu,cs),"DK168 actual-ID check");
  }
  ++d.captured;d.mask|=1u<<unsigned(nq);return;
 }
 if(!enabled()||!state.active||nq<=16||nq>Q||max_blocks!=262146||cap!=Cap||shape.idx_block!=4||shape.idx_top_k!=2048||active_blocks<Gate||active_blocks>max_blocks){real_topk(scores,steps,nq,max_blocks,cap,shape,ids,stream,active_blocks);return;}
 auto cs=static_cast<cudaStream_t>(stream);cudaStreamCaptureStatus capture;ck(cudaStreamIsCapturing(cs,&capture),"capture check");
 if(capture!=cudaStreamCaptureStatusNone||!allocate()){real_topk(scores,steps,nq,max_blocks,cap,shape,ids,stream,active_blocks);return;}
 if(state.stream&&state.stream!=cs)fail("prefill stream changed");state.stream=cs;
 ck(swift142::chunks(scores,steps,int(nq),max_blocks,state.w,0,cs),"local exact candidates");ck(swift142::pack(scores,steps,int(nq),max_blocks,state.w,cs),"stable compact");
 real_topk(state.w.packed_scores,state.w.packed_steps,nq,swift142::Packed,cap,shape,state.w.packed_ids,stream,swift142::Packed);
 ck(swift142::translate(steps,int(nq),cap,state.w,ids,cs),"global cell IDs");++state.calls;
 if(checked()&&state.comparisons<8){
  // The packed IDs are no longer needed after translation. Reuse that storage
  // for an independent original-selector comparison on actual model scores.
  real_topk(scores,steps,nq,max_blocks,cap,shape,state.w.packed_ids,stream,active_blocks);
  std::vector<int32_t>a(size_t(nq)*cap),b(a.size());
  ck(cudaMemcpyAsync(a.data(),ids,a.size()*4,cudaMemcpyDeviceToHost,cs),"candidate IDs");ck(cudaMemcpyAsync(b.data(),state.w.packed_ids,b.size()*4,cudaMemcpyDeviceToHost,cs),"original IDs");ck(cudaStreamSynchronize(cs),"checked original/candidate IDs");
  if(a!=b)fail("actual model selected IDs differ from original");++state.comparisons;
 }
}

// Frozen verify142 uses this fastpath before its single-CTA fallback. Preserve
// it unchanged when OFF; substitute exact partition only for admitted N1 shapes.
extern "C" bool real_parallel(const float*,const int32_t*,int64_t,int64_t,int64_t,const K::QsaShapes&,int32_t*,void*,uint64_t,void*) asm("__real__ZN6strata7kernels23qsa_block_topk_parallelEPKfPKilllRKNS0_9QsaShapesEPiPvmS9_");
extern "C" bool wrap_parallel(const float*,const int32_t*,int64_t,int64_t,int64_t,const K::QsaShapes&,int32_t*,void*,uint64_t,void*) asm("__wrap__ZN6strata7kernels23qsa_block_topk_parallelEPKfPKilllRKNS0_9QsaShapesEPiPvmS9_");
extern "C" bool wrap_parallel(const float* sc,const int32_t* st,int64_t nq,int64_t blocks,int64_t cap,const K::QsaShapes& shape,int32_t* ids,void* scratch,uint64_t bytes,void* stream){
 if(decode168::on()&&decode168::d.ready&&!state.active&&nq>=1&&nq<=8&&(blocks==65538||blocks==131074||blocks==262146)&&cap==Cap&&shape.idx_dim==128&&shape.idx_n_head==4&&shape.idx_block==4&&shape.idx_top_k==2048){
  wrap_topk(sc,st,nq,blocks,cap,shape,ids,stream,-1);return true;
 }
 return real_parallel(sc,st,nq,blocks,cap,shape,ids,scratch,bytes,stream);
}
