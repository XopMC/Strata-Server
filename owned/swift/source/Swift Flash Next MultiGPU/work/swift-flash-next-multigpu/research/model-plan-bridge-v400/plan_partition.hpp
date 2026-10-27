#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace model_ep_v400 {
constexpr int MaxRows=80, Experts=512, K=10;
struct Physical {int bank=-1,slot=-1;};
struct Home {
 std::uint64_t base=0,bytes=0,payload=0;
 std::span<const std::uint64_t> offsets;
};
enum class Error : std::uint32_t {
 Ok, Geometry, Slot, Pointer, Counts, Group, Route, Classification, Unbound
};
struct Original {
 int home=-1,layer=-1,T=0,cpu_jobs=0;
 std::span<const std::int32_t> ids,resident,counts,start,dst,tok;
 std::span<const std::uint64_t> ptr;
};
struct Plan {
 std::array<std::uint64_t,MaxRows> ptr{};
 std::array<std::int32_t,MaxRows+1> start{};
 std::array<std::int32_t,MaxRows> dst{},tok{};
 int groups=0,entries=0;
};
struct Result {
 Error error=Error::Unbound;
 std::uint64_t epoch=0;
 int T=0,layer=-1,cpu_jobs=0,native_rows=0;
 std::array<std::int32_t,MaxRows> union_dst{}; // EXACT original grouped order; not compacted in original sink.
 std::array<std::int32_t,MaxRows> row_bank{},row_slot{},row_logical{};
 std::array<Plan,2> plan{};
};

// Immutable registration, BEFORE capture or callbacks. Home offsets are copied
// once; no original cache pointer/table is retained by a callback result.
class Placement {
 std::array<Home,2> home_{};
 std::array<std::vector<std::uint64_t>,2> offsets_;
 std::array<std::vector<Physical>,2> map_;
 std::uint64_t epoch_=0;
 bool bound_=false;
 static bool span_ok(std::uint64_t base,std::uint64_t bytes) noexcept {
  return base&&bytes&&bytes<=std::numeric_limits<std::uint64_t>::max()-base;
 }
public:
 Placement()=default;
 Placement(const Placement&)=delete;
 Placement& operator=(const Placement&)=delete;
 Placement(Placement&&)=delete;
 Placement& operator=(Placement&&)=delete;
 bool initialize(std::array<Home,2> home,
   std::array<std::span<const Physical>,2> mapping,std::uint64_t epoch) {
  if(bound_||!epoch)return false;
  for(int b=0;b<2;++b){
   const auto&h=home[b];
   if(!span_ok(h.base,h.bytes)||(h.base&15u)||!h.payload||h.payload>h.bytes||h.offsets.empty()||
     h.offsets.size()>Experts*48||mapping[b].size()!=h.offsets.size())return false;
   std::vector<std::uint64_t> sorted(h.offsets.begin(),h.offsets.end());
   std::sort(sorted.begin(),sorted.end());
   for(std::size_t i=0;i<sorted.size();++i){auto off=sorted[i];
    if(off>h.bytes||h.payload>h.bytes-off||(off&15u)||
      (i&&off-sorted[i-1]<h.payload))return false;
   }
  }
  if(home[0].base<home[1].base+home[1].bytes&&home[1].base<home[0].base+home[0].bytes)return false;
  std::array<std::vector<bool>,2> used;
  for(int b=0;b<2;++b)used[b].resize(home[b].offsets.size(),false);
  for(int b=0;b<2;++b)for(auto p:mapping[b]){
   if(p.bank<0||p.bank>1||p.slot<0||std::size_t(p.slot)>=used[p.bank].size()||
     home[b].payload!=home[p.bank].payload||used[p.bank][p.slot])return false;
   used[p.bank][p.slot]=true;
  }
  for(int b=0;b<2;++b)if(std::find(used[b].begin(),used[b].end(),false)!=used[b].end())return false;
  for(int b=0;b<2;++b){
   offsets_[b].assign(home[b].offsets.begin(),home[b].offsets.end());
   map_[b].assign(mapping[b].begin(),mapping[b].end());
   home_[b]=home[b];home_[b].offsets=offsets_[b];
  }
  epoch_=epoch;bound_=true;return true;
 }

 // Bounded host publication transform. No allocation, GPU dereference, CUDA
 // call, file I/O, clock, arithmetic leaf or mutation of Original spans here.
 Error partition(const Original&in,Result&out) const noexcept {
  out={};
  auto fail=[&](Error e)noexcept{out={};out.error=e;return e;};
  if(!bound_)return fail(Error::Unbound);
  if(in.home<0||in.home>1||in.layer<0||in.layer>=48||in.home!=(in.layer<24?0:1)||
    in.T<1||in.T>8||in.ids.size()!=std::size_t(in.T*K)||in.resident.size()!=Experts||
    in.counts.size()<3||in.start.empty())return fail(Error::Geometry);
  const int rows=in.T*K,groups=in.counts[0],entries=in.counts[1];
  if(groups<0||groups>rows||entries<0||entries>rows||in.counts[2]!=0||
    in.cpu_jobs<0||in.cpu_jobs>rows-entries||in.start.size()<std::size_t(groups+1)||
    in.dst.size()<std::size_t(entries)||in.tok.size()<std::size_t(entries)||
    in.ptr.size()<std::size_t(groups))return fail(Error::Counts);
  if(in.start[0]!=0||in.start[groups]!=entries)return fail(Error::Group);
  out.error=Error::Ok;out.epoch=epoch_;out.T=in.T;out.layer=in.layer;
  out.cpu_jobs=in.cpu_jobs;out.native_rows=entries;
  out.row_bank.fill(-1);out.row_slot.fill(-1);out.row_logical.fill(-1);
  std::array<bool,MaxRows> seen{};
  std::array<bool,Experts> expert_group{};
  std::array<int,MaxRows> group_slot{};group_slot.fill(-1);
  int previous_first=-1;
  for(int g=0;g<groups;++g){
   int a=in.start[g],z=in.start[g+1];
   if(a<0||z<=a||z>entries)return fail(Error::Group);
   int first=in.dst[a];
   if(first<=previous_first||first>=rows)return fail(Error::Route);
   previous_first=first;int e=in.ids[first];
   if(e<0||e>=Experts||expert_group[e])return fail(Error::Group);
   expert_group[e]=true;int logical=in.resident[e];
   if(logical<0||std::size_t(logical)>=map_[in.home].size())return fail(Error::Slot);
   for(int prior=0;prior<g;++prior)if(group_slot[prior]==logical)return fail(Error::Slot);
   group_slot[g]=logical;
   const auto&source=home_[in.home];
   if(in.ptr[g]!=source.base+source.offsets[logical])return fail(Error::Pointer);
   auto dest=map_[in.home][logical];const auto&target=home_[dest.bank];auto&p=out.plan[dest.bank];
   int q=p.groups++;p.start[q]=p.entries;p.ptr[q]=target.base+target.offsets[dest.slot];
   int previous=-1;
   for(int j=a;j<z;++j){int row=in.dst[j];
    if(row<0||row>=rows||row<=previous||seen[row]||in.tok[j]!=row/K||in.ids[row]!=e)return fail(Error::Route);
    previous=row;seen[row]=true;
    if(in.resident[in.ids[row]]!=logical)return fail(Error::Classification);
    out.union_dst[j]=row;out.row_bank[row]=dest.bank;out.row_slot[row]=dest.slot;out.row_logical[row]=logical;
    int k=p.entries++;p.dst[k]=row;p.tok[k]=in.tok[j];
   }
  }
  for(int b=0;b<2;++b)out.plan[b].start[out.plan[b].groups]=out.plan[b].entries;
  for(int row=0;row<rows;++row){int e=in.ids[row];
   if(e<0||e>=Experts)return fail(Error::Route);
   int s=in.resident[e];
   if(s>=0&&std::size_t(s)>=map_[in.home].size())return fail(Error::Slot);
   if(seen[row]!=(s>=0))return fail(Error::Classification);
  }
  return Error::Ok;
 }
};
}
