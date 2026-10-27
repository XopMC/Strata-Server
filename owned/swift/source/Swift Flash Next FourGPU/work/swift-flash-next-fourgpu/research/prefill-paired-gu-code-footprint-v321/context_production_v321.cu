#include "common.cuh"
#include "paired_v307.hpp"
#include <stdexcept>
namespace paired_v307 {
void dispatch_control(ggml_backend_cuda_context&,const strata::prefill::mmq::Product&,cudaStream_t);
bool dispatch_pair(ggml_backend_cuda_context&,const strata::prefill::mmq::Product&,const Epilogue&,cudaStream_t);
namespace {
struct BorrowedPool final:ggml_cuda_pool {
 void*base=nullptr;size_t capacity=0;bool live=false;void*stream=nullptr;
 void bind(void*p,size_t n,void*s){if(live||!p||n<(64ull<<20)||((uintptr_t)p&255)||!s)throw std::runtime_error("V307 nonowning64MiB pool bind");base=p;capacity=n;stream=s;}
 void*alloc(size_t n,size_t*actual)override{if(live||!base||n>capacity)throw std::runtime_error("V307 bounded scratch exhausted; no allocation fallback");live=true;*actual=n;return base;}
 void free(void*p,size_t n)override{if(!live||p!=base||n>capacity)std::abort();live=false;}
};
struct State {
 ggml_backend_cuda_context ctx;BorrowedPool*pool;
 explicit State(int d):ctx(d){auto p=std::make_unique<BorrowedPool>();pool=p.get();ctx.pools[d][0]=std::move(p);}
};
}
Context::Context(){int d=0;CUDA_CHECK(cudaGetDevice(&d));impl_=new State(d);}
Context::~Context(){delete static_cast<State*>(impl_);}
void Context::bind_after_drain(void*p,size_t n,void*s){static_cast<State*>(impl_)->pool->bind(p,n,s);}
bool Context::run_paired(const strata::prefill::mmq::Product&p,const Epilogue&e,void*stream){
 using namespace strata::prefill::mmq;
 if(p.n<1||p.n>16||p.type!=40||p.w_rows!=1280||p.w_cols!=2560||p.expert_bytes!=1843200||
    (p.ld_dst!=1280&&p.ld_dst!=2560)||!p.w||!p.xq||!p.bounds||!p.ids||!p.dst||p.tab||p.y_scale||
    p.total_rows<=0||p.total_rows>32768ll*10||p.max_rows<=0||p.max_rows>p.total_rows||
    !e.hq||!e.tails||e.nr<=0||e.row0<0||e.nr>p.total_rows||e.row0>p.total_rows-e.nr)
  throw std::runtime_error("V307 strict native40 GU/pair lifetime geometry");
 auto&st=*static_cast<State*>(impl_);int d=-1;CUDA_CHECK(cudaGetDevice(&d));
 if(d!=st.ctx.device||stream!=st.pool->stream||!st.pool->base||ggml_cuda_info().devices[d].cc!=890||nvfp4_mode()!=Nvfp4Mode::W4A8||fp4_activations(40))
  throw std::runtime_error("V307 original owning SM89 W4A8 stream/scratch");
 const bool admitted=dispatch_pair(st.ctx,p,e,static_cast<cudaStream_t>(stream));
 CUDA_CHECK(cudaGetLastError());return admitted;
}
}
