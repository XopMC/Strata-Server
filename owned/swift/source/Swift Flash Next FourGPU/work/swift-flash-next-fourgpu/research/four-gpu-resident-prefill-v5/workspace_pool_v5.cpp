#include "workspace_pool_v5.hpp"
#include "strata/prefill/prefill.hpp"
#include "peer_matrix.hpp"
#include <cstdio>
#include <cstdlib>
namespace resident_pp_v5 {
namespace {
constexpr uint64_t Reserve=768ull<<20;
void check(cudaError_t e,const char*why){if(e!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e));}
[[noreturn]]void fatal(const char*why,cudaError_t e=cudaErrorUnknown)noexcept{std::fprintf(stderr,"PP5 workspace retirement refused: %s: %s\n",why,cudaGetErrorString(e));std::fflush(stderr);std::_Exit(2);}
struct Restore{int previous=-1;Restore(){check(cudaGetDevice(&previous),"PP5 workspace caller");}~Restore(){if(previous>=0&&cudaSetDevice(previous)!=cudaSuccess)fatal("restore context");}};
void drain_all()noexcept{
 for(int d=0;d<4;++d){auto e=cudaSetDevice(d);if(e!=cudaSuccess)fatal("drain select",e);e=cudaDeviceSynchronize();if(e!=cudaSuccess)fatal("positive all-consumer drain",e);}
}
}
namespace workspace_detail {
struct State {
 void*allocation=nullptr;uint64_t byte_count=0;int owner=-1;Profile profile=Profile::Allocated16K;
 ~State(){if(allocation){int old=-1;if(cudaGetDevice(&old)!=cudaSuccess)fatal("save free context");drain_all();auto e=cudaSetDevice(owner);if(e!=cudaSuccess)fatal("free owner",e);e=cudaFree(allocation);if(e!=cudaSuccess)fatal("independent arena free",e);if(cudaSetDevice(old)!=cudaSuccess)fatal("restore after free");}}
};
}
WorkspaceLease::WorkspaceLease(std::shared_ptr<workspace_detail::State>s):p_(std::move(s)){}
void*WorkspaceLease::base()const{require(p_&&p_->allocation,"PP5 empty independent arena lease");return p_->allocation;}
uint64_t WorkspaceLease::bytes()const{base();return p_->byte_count;}
int WorkspaceLease::device()const{base();return p_->owner;}
int WorkspaceLease::rows()const{base();return int(p_->profile);}
std::shared_ptr<const void>WorkspaceLease::retain()const{base();return p_;}
bool WorkspaceLease::owns(const void*p,uint64_t n)const{uintptr_t b=uintptr_t(base()),q=uintptr_t(p);return p&&n&&q>=b&&n<=bytes()&&q-b<=bytes()-n;}
AllocatedShape WorkspaceLease::allocated_shape(int actual_T,int maximum,int configured,int next)const{
 require(actual_T==rows(),"PP5 actual Impl::T differs from independently allocated profile");AllocatedShape a{p_->profile,rows(),maximum,configured,next};a.validate();return a;
}
void WorkspaceLease::admit_R(const float*p)const{require(owns(p,uint64_t(rows())*4*N*4),"PP5 actual R outside independently owned fixed-capacity arena");}
GroupBuffers WorkspaceLease::group_buffers(unsigned char*g,unsigned char*d,float*t)const{
 require(owns(g,gu_workspace_bytes())&&owns(d,dn_workspace_bytes())&&owns(t,tail_workspace_bytes()),"PP5 carved group buffers outside independent arena");return {g,d,t,gu_workspace_bytes(),dn_workspace_bytes(),tail_workspace_bytes(),retain()};
}
WorkspaceLease WorkspacePool::allocate(int d,Profile profile,const strata::core::ModelGeometry&g,const strata::core::SessionState&s){
 require(d>=0&&d<2&&!owned_[size_t(d)],"PP5 duplicate/foreign dense allocation");const int c=int(profile);require(c==16384||c==8192,"PP5 explicit16k or separate8k allocation only");
 require(g.n_layers==48&&g.n_embd==N&&g.n_ff==F&&g.n_expert==NE&&g.n_expert_used==K&&g.hc==4&&s.k==K&&s.layer_lo==d*24&&s.layer_hi==(d+1)*24,"PP5 original two24-layer stage geometry");
 require(s.max_cells==262144||s.max_cells==524288||s.max_cells==1048576,"PP5 explicit supported context");Restore restore;std::string error;
 const bool peer_ok=strata_peer_default::require_all(4,error);require(peer_ok,error.c_str());check(cudaSetDevice(d),"PP5 allocation owner");
 // Reuse the exact original contiguous-borrow carve price. The storage is a
 // NEW independent allocation. Ring omission is deliberately a separate carve
 // change; this conservative price never underprices the current original init.
 const uint64_t need=strata::prefill::Prefill::bytes_needed(g,s,c,true);require(need&&need<UINT64_MAX-Reserve,"PP5 invalid exact source arena price");size_t free=0,total=0;check(cudaMemGetInfo(&free,&total),"PP5 independent arena budget");require(free>=need+Reserve,"PP5 full fixed profile+768MiB reserve does not fit; no cache loan/shrink or8k retry");
 auto state=std::make_shared<workspace_detail::State>();state->owner=d;state->profile=profile;state->byte_count=need;check(cudaMalloc(&state->allocation,size_t(need)),"PP5 independent full profile allocation");
 check(cudaMemGetInfo(&free,&total),"PP5 postallocation reserve");require(free>=Reserve,"PP5 measured postallocation reserve below768MiB");owned_[size_t(d)]=state;
 std::fprintf(stderr,"PP5_WORKSPACE CUDA%d allocated_rows=%d bytes=%llu independent_cudaMalloc=1 cache_loan_bytes=0 reserve_MiB=%zu\n",d,c,(unsigned long long)need,free>>20);return WorkspaceLease(std::move(state));
}
WorkspacePool::~WorkspacePool(){
 bool any=false;for(auto&p:owned_)if(p){any=true;if(p.use_count()!=1)fatal("workspace pool outlives outstanding PP/graph leases contract violated");}
 if(!any)return;int old=-1;if(cudaGetDevice(&old)!=cudaSuccess)fatal("save pool retirement owner");drain_all();for(auto&p:owned_)p.reset();if(cudaSetDevice(old)!=cudaSuccess)fatal("restore pool owner");
}
} // namespace resident_pp_v5
