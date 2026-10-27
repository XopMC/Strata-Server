#pragma once
// CPU metadata only. Never changes the authoritative logical R or real banks.
#include "../four-gpu-resident-prefill-v5/resident_plan_v5.hpp"
#include "strata/program/split_ring.hpp"
#include <array>
#include <memory>
#include <span>
#include <vector>
namespace resident_pp_v12 {
class OriginalReference {
 struct State {int rows=0;uint64_t epoch=0;std::array<int32_t,2>first{},slots{};std::array<uint64_t,2>prices{};std::vector<int32_t>R;};
 std::shared_ptr<const State>s_;
public:
 OriginalReference()=default;
 // original_PP_prices are the SAME original full-ring bytes_needed prices for
 // the actual selected profile on each home; V5 pool conservatively owns these.
 static OriginalReference from_virtual_tail_loans(int rows,std::span<const int32_t>full_original_R,
   const std::array<std::span<const uint64_t>,2>&home_offsets,const std::array<uint64_t,2>&original_PP_prices,uint64_t epoch){
  using resident_pp_v5::require;
  require((rows==16384||rows==8192)&&epoch&&full_original_R.size()==24576,"PP12 reference profile/epoch/full R");
  auto s=std::make_shared<State>();s->rows=rows;s->epoch=epoch;s->prices=original_PP_prices;s->R.assign(full_original_R.begin(),full_original_R.end());
  for(int d=0;d<2;++d){
   auto offsets=home_offsets[size_t(d)];require(offsets.size()>129&&offsets.size()<=12289&&offsets.front()==0,"PP12 exact original virtual offsets required");
   const int slots=int(offsets.size()-1);s->slots[size_t(d)]=slots;
   for(int i=0;i<slots;++i)require(offsets[size_t(i)+1]-offsets[size_t(i)]==four_banks_v3::SlotStride,"PP12 original native40 virtual stride");
   require(offsets.back()==uint64_t(slots)*four_banks_v3::SlotStride&&original_PP_prices[size_t(d)]>0,"PP12 exact original virtual extent/price");
   const int64_t k=strata::program::split_lend_slots(slots,int64_t(offsets.back()),offsets.data(),four_banks_v3::BlobBytes,original_PP_prices[size_t(d)]);
   require(k>0&&k+128<=slots,"PP12 original virtual tail loan would not fit; no profile shrink");
   const int first=slots-int(k);s->first[size_t(d)]=first;std::vector<uint8_t>seen(size_t(slots),0);
   for(int key=d*12288;key<(d+1)*12288;++key){int32_t r=s->R[size_t(key)];require(r>=-1&&r<slots,"PP12 invalid original home slot");if(r>=0){require(++seen[size_t(r)]==1,"PP12 duplicate original virtual slot");if(r>=first)s->R[size_t(key)]=-1;}}
   for(auto n:seen)require(n==1,"PP12 original logical home capacity coverage");
  }
  OriginalReference r;r.s_=std::move(s);return r;
 }
 void validate(int rows)const{resident_pp_v5::require(s_&&s_->rows==rows&&s_->epoch&&s_->R.size()==24576,"PP12 absent/wrong-profile reference");}
 int rows()const{resident_pp_v5::require(bool(s_),"PP12 missing reference");return s_->rows;}
 uint64_t epoch()const{validate(rows());return s_->epoch;}
 std::span<const int32_t>residency()const{validate(rows());return s_->R;}
 uint64_t original_price(int d)const{validate(rows());resident_pp_v5::require(d>=0&&d<2,"PP12 home bound");return s_->prices[size_t(d)];}
 int virtual_first(int d)const{validate(rows());resident_pp_v5::require(d>=0&&d<2,"PP12 home bound");return s_->first[size_t(d)];}
};
} // namespace resident_pp_v12
