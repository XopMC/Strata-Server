#include "index_pair_admission_v501.hpp"
#include "../../whole-fixed-plan-owner-v469/source/resources_v469.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
namespace ds370 {
namespace {
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
bool mode469(){const char*v=std::getenv("DS1017_FIXED_PLAN_REFRESH_V469");need(v&&(!std::strcmp(v,"0")||!std::strcmp(v,"1")),"469 explicit immutable refresh mode");return !std::strcmp(v,"1");}
void exact(const char*k,const char*v){const char*p=std::getenv(k);need(p&&!std::strcmp(p,v),k);}
}
struct Owner::Impl {
 std::shared_ptr<ds469::Resources>resources;
 DsWholeHooksV299 hooks{};PolicyHooks370 policy{};
 std::mutex ref_mutex;std::shared_ptr<Owner>hook_hold;uint64_t hook_refs=0;
};
Owner::Owner(std::shared_ptr<const strata::ds41::BorrowedWeights>w,std::vector<Descriptor>d,const Ds276ExpTable&t,Config370 c):p_(std::make_unique<Impl>()){
 exact("DS1017_FIXED_PLAN_UPLOAD_V465","1");exact("DS1017_WHOLE_ADAPTIVE_V370","0");const bool mode=mode469();
 ds501::require_environment();
 int devices=0;need(cudaGetDeviceCount(&devices)==cudaSuccess&&devices==2,"469 exactly two admitted devices");
 need(w&&!ds276_exp_table_validate(&t)&&d.size()==15360&&!c.adaptive&&!c.component_only&&c.budget==(20ull<<30)&&!c.component_ready_delay_tasks&&!c.component_fail_after_submit,"469 production catalog/config/table");
 const auto b=strata::ds41::pair145::info(*w);need(b[0]==145&&b[1]==139&&b[2]==46080&&b[3]==305764761600ull&&b[4]&&b[5]&&b[6]==1&&!b[7],"469 full retained registry binding");
 std::stable_sort(d.begin(),d.end(),[](const Descriptor&a,const Descriptor&b){return a.frequency!=b.frequency?a.frequency>b.frequency:a.layer!=b.layer?a.layer<b.layer:a.expert<b.expert;});
 std::vector<ds469::Triple>catalog(15360);std::vector<unsigned>ranked;ranked.reserve(15360);std::array<bool,15360>seen{};
 for(const auto&v:d){
  need(v.layer>=0&&v.layer<40&&v.expert>=0&&v.expert<384&&v.frequency<=1000000000000ull,"469 descriptor range");
  const unsigned id=unsigned(v.layer*384+v.expert);need(!seen[id],"469 unique full catalog");seen[id]=true;ranked.push_back(id);
  const auto prefix="layers."+std::to_string(v.layer)+".ffn.experts."+std::to_string(v.expert);
  for(unsigned r=0;r<3;r++)catalog[id][r]=strata::ds41::pair145::projection(*w,prefix+(r==0?".w1":r==1?".w3":".w2"),r==2?5120:2304,r==2?2304:5120);
 }
 p_->resources=std::make_shared<ds469::Resources>(std::move(w),std::move(catalog),std::move(ranked),t,mode);
 p_->hooks={sizeof(DsWholeHooksV299),299,this,retain,release,contains,begin,finish,release_view,abort};
 p_->policy={sizeof(PolicyHooks370),370,this,retain,release,before,after};
}
std::shared_ptr<Owner>Owner::create(std::shared_ptr<const strata::ds41::BorrowedWeights>w,std::vector<Descriptor>d,const Ds276ExpTable&t,Config370 c){
 auto owner=std::shared_ptr<Owner>(new Owner(std::move(w),std::move(d),t,c));
 ds469::register_diagnostic(owner->p_->resources);return owner;
}
Owner::~Owner()noexcept=default;
const DsWholeHooksV299&Owner::hooks()const noexcept{return p_->hooks;}
const PolicyHooks370&Owner::policy_hooks()const noexcept{return p_->policy;}
int Owner::arm(uint64_t g,uint64_t e)noexcept{try{p_->resources->arm(g,e);return 0;}catch(...){return 1;}}
// The old max-two synchronous gap service is deliberately unavailable. The
// fixed plan is signalled by the successful 32nd after callback, never Python.
int Owner::gap_service(uint64_t,uint64_t,uint64_t,unsigned)noexcept{return 1;}
Receipt370 Owner::receipt()const{return p_->resources->legacy();}
int Owner::close()noexcept{try{p_->resources->close();return 0;}catch(...){return 9;}}
void Owner::retain(void*raw)noexcept{try{auto&o=*static_cast<Owner*>(raw);std::lock_guard<std::mutex>l(o.p_->ref_mutex);if(!o.p_->hook_refs)o.p_->hook_hold=o.shared_from_this();++o.p_->hook_refs;o.p_->resources->hook_refs(o.p_->hook_refs);}catch(...){std::terminate();}}
void Owner::release(void*raw)noexcept{std::shared_ptr<Owner>last;try{auto&o=*static_cast<Owner*>(raw);{std::lock_guard<std::mutex>l(o.p_->ref_mutex);if(!o.p_->hook_refs)std::terminate();--o.p_->hook_refs;o.p_->resources->hook_refs(o.p_->hook_refs);if(!o.p_->hook_refs)last=std::move(o.p_->hook_hold);}}catch(...){std::terminate();}}
int Owner::contains(void*p,int e,uintptr_t g,uintptr_t u,uintptr_t d)noexcept{if(!p)return -1;auto&r=*static_cast<Owner*>(p)->p_->resources;try{return r.contains(e,g,u,d);}catch(...){r.poison();return -1;}}
int Owner::begin(void*p,const DsWholeCommandV299*c,void**t)noexcept{if(t)*t=nullptr;if(!p)return 1;auto&r=*static_cast<Owner*>(p)->p_->resources;try{r.begin(c,t);return 0;}catch(...){r.poison();return 9;}}
int Owner::finish(void*p,void*t,uint64_t e,const uint16_t**o,uint64_t n)noexcept{if(o)*o=nullptr;if(!p)return 1;auto&r=*static_cast<Owner*>(p)->p_->resources;try{r.finish(t,e,o,n);return 0;}catch(...){r.poison();return 9;}}
int Owner::release_view(void*p,void*t,uint64_t e)noexcept{if(!p)return 1;auto&r=*static_cast<Owner*>(p)->p_->resources;try{r.release(t,e);return 0;}catch(...){r.poison();return 9;}}
int Owner::abort(void*p,void*t,uint64_t e)noexcept{if(!p)return 1;auto&r=*static_cast<Owner*>(p)->p_->resources;try{r.abort(t,e);return 0;}catch(...){r.poison();return 9;}}
int Owner::before(void*p,uintptr_t k,unsigned m,const uint8_t*c,unsigned n)noexcept{if(!p)return 1;try{static_cast<Owner*>(p)->p_->resources->before(k,m,c,n);return 0;}catch(...){static_cast<Owner*>(p)->p_->resources->poison();return 9;}}
int Owner::after(void*p,int status)noexcept{if(!p)return 1;try{static_cast<Owner*>(p)->p_->resources->after(status);return 0;}catch(...){static_cast<Owner*>(p)->p_->resources->poison();return 9;}}
}
