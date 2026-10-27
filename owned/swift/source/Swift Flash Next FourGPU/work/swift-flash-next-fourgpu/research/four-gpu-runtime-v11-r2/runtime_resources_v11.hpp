#pragma once
#include "admission_v11.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/program/split_ring.hpp"
#include "include/resident_runtime_v11.hpp"
#include "../four-gpu-bank-loader-v9-r1/bank_loader_v9.hpp"
#include "strata/kernels/cpu/expert_layout.hpp" // declaration required by V9 arena adapter
#include "../four-gpu-bank-loader-v9-r1/arena_source_v9.hpp"
#include "../four-gpu-resident-prefill-v5/workspace_pool_v5.hpp"
#include "resident_reference_v12.hpp"
#include "../four-gpu-virtual-logical-cache-v8/bind_selected_v8.hpp"
#include <charconv>
namespace four_runtime_main_v11 {
// Original PP ring/pinning policy must be selected BEFORE bank preflight prices,
// even though the independent physical C is exhaustive and PP streams no RAM weights.
inline void configure_original_pp(strata::core::ExpertSource*source,const four_reference_v11::Metadata&m){
 four_reference_v11::need(source,"original arena metadata required");uint64_t pinned=0;
 for(int l=0;l<48;++l)for(int e=0;e<512;++e)pinned+=source->pinned(l,e);
 double share=double(pinned)/24576.0;if(const char*t=std::getenv("STRATA_TEST_PINNED_SHARE"))share=std::atof(t);
 strata::prefill::Prefill::set_pinned_share(share);
 int64_t n=0;for(auto r:m.map)n+=r>=0;
 int ring=strata::program::split_ring_slots(std::getenv("STRATA_SPLIT_RING"),double(n)/24576.0);
 strata::prefill::Prefill::set_ring_override(ring); // including0: clear inherited no-borrow96 override
 std::fprintf(stderr,"FOUR11_PP_POLICY logical_resident=%lld pinned_fraction=%.17g original_ring_override=%d physicalC_does_not_classify=1\n",(long long)n,share,ring);
}
inline uint64_t positive(const char*k){const char*s=std::getenv(k);four_reference_v11::need(s&&*s,k);uint64_t n=0;auto r=std::from_chars(s,s+std::strlen(s),n);four_reference_v11::need(r.ec==std::errc{}&&!*r.ptr&&n,k);return n;}
inline four_loader_v9::Options budget(const std::array<uint64_t,2>&pp){
 auto*s=std::getenv("STRATA_FOUR_FUTURE_BYTES_V11");four_reference_v11::need(s&&*s,"24 explicit future budget bytes required");std::vector<uint64_t>v;const char*p=s;for(;;){uint64_t x=0;const char*e=std::strchr(p,',');if(!e)e=p+std::strlen(p);auto r=std::from_chars(p,e,x);four_reference_v11::need(r.ec==std::errc{}&&r.ptr==e,"budget integer");v.push_back(x);if(!*e)break;p=e+1;}four_reference_v11::need(v.size()==24,"4x6 explicit future budget categories");
 four_loader_v9::Options o;o.physical_epoch=positive("STRATA_FOUR_PHYSICAL_EPOCH_V11");for(int d=0;d<4;++d){auto&f=o.future[size_t(d)];size_t q=size_t(d)*6;f={v[q],v[q+1],v[q+2],v[q+3],v[q+4],v[q+5],true};four_reference_v11::need(f.prefill_peak>=(d<2?pp[size_t(d)]+(128ULL<<20):0),"future PP budget below exact workspace+128MiB original external buffers");four_reference_v11::need(f.head>=(d==1?62080*8*4ULL:130368000ULL+2*62080*8*4+32768),"future four-way head remote budget too small");four_reference_v11::need(f.controller_other>=(16ULL<<20),"controller future excludes16MiB buffers; Verifier+graphs need additional explicit budget");four_reference_v11::need(f.safety_reserve>=(768ULL<<20),"physical loader safety reserve below768MiB");}return o;
}
// Declare before stages, every Prefill/Verifier and their leases. Destruction is
// strictly after each Verifier destroys its parent graphs and unregisters while
// its own original stream is still live. No borrowed stream is kept here.
struct Resources {
 std::shared_ptr<strata::core::ArenaExpertSource>arena=std::make_shared<strata::core::ArenaExpertSource>();
 std::unique_ptr<four_loader_v9::BankOwner>banks;
 std::unique_ptr<resident_pp_v5::WorkspacePool>workspaces;
 std::array<resident_pp_v5::WorkspaceLease,2>pp;
 resident_pp_v12::OriginalReference pp_reference;
 bool runtime_prepared=false;
 ~Resources(){try{if(runtime_prepared){resident_runtime_v11::deactivate();resident_runtime_v11::close();}pp_reference={};pp={};workspaces.reset();if(banks){banks->close(true);banks.reset();}}catch(const std::exception&e){std::fprintf(stderr,"FOUR11_FATAL retirement: %s\n",e.what());std::fflush(stderr);std::_Exit(2);}}
 void load(const std::array<uint64_t,2>&prices){four_reference_v11::need(!banks,"physical banks already loaded");banks=four_loader_v9::BankOwner::load(four_loader_v9::from_original_arena(arena,positive("STRATA_FOUR_SOURCE_EPOCH_V11")),budget(prices));four_reference_v11::need(banks->receipt().four_uploads_complete&&banks->receipt().metadata_readback&&banks->receipt().outer_guards_identical,"incomplete physical bank publication");}
 void prepare(const int32_t*R,size_t n){four_reference_v11::need(banks&&!runtime_prepared,"bad runtime preparation phase");resident_runtime_v11::Config c;for(int d=0;d<4;++d)c.physical_views[size_t(d)]=banks->view(d);c.physical_owner=banks->retain();c.host_res=R;c.host_res_count=n;c.logical_epoch=1;c.adapt_swaps=0;resident_runtime_v11::prepare(c);runtime_prepared=true;}
};
}
