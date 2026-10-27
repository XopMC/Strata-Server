#pragma once
// Host-only descriptors. This does not modify logical CPU/native residency.
#include "../four-gpu-physical-banks-v3/physical_types_v3.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>
namespace resident_pp_v5 {
constexpr int N=2560,F=640,K=10,NE=512,MaxGroup=16;
constexpr uint64_t GuBytes=1843200,DnBytes=921600,UpOffset=921600,DownOffset=1843200,TailOffset=2764800,TailZero=4096;
static_assert(TailOffset+16==four_banks_v3::BlobBytes);
enum class Profile:int {Allocated16K=16384,Allocated8K=8192};
inline void require(bool b,const char*why){if(!b)throw std::runtime_error(why);}
struct AllocatedShape {
 Profile profile=Profile::Allocated16K;
 int arena_rows=0,T_max=0,configured_chunk=0,successor_rows=0;
 void validate()const{
  const int c=int(profile);require(c==16384||c==8192,"PP5 only explicit allocated16k/8k profiles");
  require(arena_rows==c&&T_max==c&&configured_chunk==c&&successor_rows==c,"PP5 actual allocated capacities differ; no request-size relabel/retry");
 }
};
struct Group {
 int layer=-1,tokens=0,allocated_rows=0,original_group=-1,order0=0,n=0,row0=0,nrows=0,max_rows=0;
 uint64_t physical_epoch=0;
 std::array<int32_t,MaxGroup>experts{};
 std::array<int32_t,MaxGroup+1>absolute{},relative{};
 std::array<four_banks_v3::PhysicalRef,MaxGroup>physical{};
};
struct Plan {int layer=-1,tokens=0,allocated_rows=0;uint64_t physical_epoch=0;std::vector<Group>groups;};
// These are the ALREADY SELECTED original grouping tables, including any admitted
// original count-sort policy. This adapter never sorts, splits or regroups them.
struct OriginalRows {
 std::span<const int32_t>route_ids; // T*K in original token/rank order
 std::span<const int32_t>expert_order; // every nonempty expert exactly once
 std::span<const int32_t>offsets; // NE+1, expert-indexed original offsets + final rows
 std::span<const int32_t>slot; // original rank -> expert-ordered Dm row, T*K
 std::span<const int32_t>src; // expert-ordered row -> original token, T*K
};
inline Plan make_plan(int layer,int tokens,const AllocatedShape&shape,const OriginalRows&original,
                     std::span<const four_banks_v3::PhysicalRef>refs,uint64_t epoch){
 shape.validate();require(layer>=0&&layer<48&&tokens>=1&&tokens<=shape.arena_rows,"PP5 layer/T bounds");
 require(epoch&&refs.size()==four_banks_v3::Keys,"PP5 complete physical snapshot required");
 const int rows=tokens*K;
 require(original.route_ids.size()==size_t(rows)&&original.offsets.size()==NE+1&&original.slot.size()==size_t(rows)&&original.src.size()==size_t(rows),"PP5 original route table spans");
 std::array<int32_t,NE>counts{},off{},fill{};std::array<bool,NE>seen{};
 std::array<uint64_t,8>token_seen{};
 for(int i=0;i<rows;++i){if(i%K==0)token_seen={};int e=original.route_ids[size_t(i)];require(e>=0&&e<NE,"PP5 invalid original expert ID");uint64_t bit=uint64_t(1)<<(e%64);require(!(token_seen[size_t(e/64)]&bit),"PP5 duplicate original expert rank in token");token_seen[size_t(e/64)]|=bit;++counts[size_t(e)];}
 int cursor=0;
 for(auto e:original.expert_order){
  require(e>=0&&e<NE&&counts[size_t(e)]>0&&!seen[size_t(e)],"PP5 missing/duplicate/empty original expert order");
  seen[size_t(e)]=true;off[size_t(e)]=cursor;require(original.offsets[size_t(e)]==cursor,"PP5 original expert row offsets changed");cursor+=counts[size_t(e)];
 }
 require(cursor==rows&&original.offsets[NE]==rows,"PP5 complete original row coverage");
 for(int e=0;e<NE;++e)require(seen[size_t(e)]==(counts[size_t(e)]>0),"PP5 routed expert omitted");
 fill=off;
 for(int i=0;i<rows;++i){int e=original.route_ids[size_t(i)],p=fill[size_t(e)]++;require(original.slot[size_t(i)]==p&&original.src[size_t(p)]==i/K,"PP5 original stable within-expert/rank row order changed");}
 Plan out;out.layer=layer;out.tokens=tokens;out.allocated_rows=shape.arena_rows;out.physical_epoch=epoch;
 out.groups.reserve((original.expert_order.size()+15)/16);
 for(size_t j=0;j<original.expert_order.size();j+=MaxGroup){
  Group g;g.layer=layer;g.tokens=tokens;g.allocated_rows=shape.arena_rows;g.original_group=int(j/16);g.order0=int(j);g.n=int(std::min(size_t(MaxGroup),original.expert_order.size()-j));g.physical_epoch=epoch;
  g.row0=off[size_t(original.expert_order[j])];g.absolute[0]=g.row0;
  for(int q=0;q<g.n;++q){
   int e=original.expert_order[j+size_t(q)];g.experts[size_t(q)]=e;g.max_rows=std::max(g.max_rows,counts[size_t(e)]);
   g.absolute[size_t(q)+1]=g.absolute[size_t(q)]+counts[size_t(e)];
   const auto&r=refs[size_t(layer*NE+e)];require(r.key==uint32_t(layer*NE+e)&&r.physical_epoch==epoch&&r.device>=0&&r.device<4&&r.physical_slot>=0&&r.physical_slot<four_banks_v3::Keys&&r.blob_bytes==four_banks_v3::BlobBytes&&r.byte_offset==uint64_t(r.physical_slot)*four_banks_v3::SlotStride,"PP5 invalid/stale physical key descriptor");g.physical[size_t(q)]=r;
  }
  g.nrows=g.absolute[size_t(g.n)]-g.row0;
  for(int q=0;q<=MaxGroup;++q){if(q>g.n)g.absolute[size_t(q)]=g.absolute[size_t(g.n)];g.relative[size_t(q)]=g.absolute[size_t(q)]-g.row0;}
  require(g.nrows>0&&g.max_rows>0&&g.row0+g.nrows<=rows,"PP5 group row extent");out.groups.push_back(g);
 }
 return out;
}
struct Traffic {uint64_t local_weight_bytes=0,peer_weight_bytes=0,group_write_bytes=0,activation_peer_bytes=0,FP32_return_peer_bytes=0;int groups=0;};
inline Traffic traffic(const Plan&p,int owner){
 require(owner==p.layer/24,"PP5 actual dense owner mismatch");Traffic t;
 for(const auto&g:p.groups){++t.groups;for(int i=0;i<g.n;++i)(g.physical[size_t(i)].device==owner?t.local_weight_bytes:t.peer_weight_bytes)+=four_banks_v3::BlobBytes;t.group_write_bytes+=uint64_t(g.n)*four_banks_v3::BlobBytes+2*TailZero;}
 return t;
}
inline constexpr uint64_t gu_workspace_bytes(){return MaxGroup*GuBytes+TailZero;}
inline constexpr uint64_t dn_workspace_bytes(){return MaxGroup*DnBytes+TailZero;}
inline constexpr uint64_t tail_workspace_bytes(){return MaxGroup*16;}
} // namespace resident_pp_v5
