#pragma once
#include "../four-gpu-resident-prefill-v5/resident_plan_v5.hpp"
#include <type_traits>
namespace resident_pp_v12 {
// The ORIGINAL V115 GPU route barrier has already range/error/stable-slot/src
// qualified its maps. This descriptor walk uses original CPU counts/order/off;
// it does not add an ID/activation/row readback to production PP.
inline resident_pp_v5::Plan count_plan(int layer,int tokens,const resident_pp_v5::AllocatedShape&shape,
 std::span<const int32_t>counts,std::span<const int32_t>order,std::span<const int32_t>offsets,
 std::span<const four_banks_v3::PhysicalRef>refs,uint64_t epoch){
 using namespace resident_pp_v5;
 shape.validate();require(layer>=0&&layer<48&&tokens>0&&tokens<=shape.arena_rows&&counts.size()==512&&offsets.size()==513&&refs.size()==24576&&epoch,"PP12 original count descriptor spans");
 std::array<bool,512>seen{};int rows=0;
 for(auto c:counts){require(c>=0&&c<=tokens,"PP12 original perexpert count range");rows+=c;}require(rows==tokens*10&&offsets[512]==rows,"PP12 original row coverage");
 int cursor=0;for(auto e:order){require(e>=0&&e<512&&!seen[size_t(e)]&&counts[size_t(e)]>0&&offsets[size_t(e)]==cursor,"PP12 original order/offset changed");seen[size_t(e)]=true;cursor+=counts[size_t(e)];}
 require(cursor==rows,"PP12 routed expert omitted");for(int e=0;e<512;++e)require(seen[size_t(e)]==(counts[size_t(e)]>0),"PP12 original group expert coverage");
 Plan p;p.layer=layer;p.tokens=tokens;p.allocated_rows=shape.arena_rows;p.physical_epoch=epoch;
 for(size_t at=0;at<order.size();at+=16){Group g;g.layer=layer;g.tokens=tokens;g.allocated_rows=shape.arena_rows;g.original_group=int(at/16);g.order0=int(at);g.n=int(std::min(size_t(16),order.size()-at));g.row0=offsets[size_t(order[at])];g.physical_epoch=epoch;g.absolute[0]=g.row0;
  for(int q=0;q<g.n;++q){int e=order[at+size_t(q)];g.experts[size_t(q)]=e;g.max_rows=std::max(g.max_rows,counts[size_t(e)]);g.absolute[size_t(q)+1]=g.absolute[size_t(q)]+counts[size_t(e)];g.physical[size_t(q)]=refs[size_t(layer*512+e)];}
  g.nrows=g.absolute[size_t(g.n)]-g.row0;for(int q=0;q<=16;++q){if(q>g.n)g.absolute[size_t(q)]=g.absolute[size_t(g.n)];g.relative[size_t(q)]=g.absolute[size_t(q)]-g.row0;}p.groups.push_back(g);
 }return p;
}
} // namespace resident_pp_v12
