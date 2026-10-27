#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace row_fragment_v410 {
constexpr std::size_t GuRow=1440,DownRow=360,Hidden=640,Width=2560;
constexpr std::size_t Up=Hidden*GuRow,Down=2*Up,Tail=Down+Width*DownRow,Blob=Tail+16;
struct Slice {std::size_t source,destination,bytes;};
struct Layout {
 int bank=0,hidden_begin=0,hidden_rows=320,output_begin=0,output_rows=0;
 std::size_t gate=0,up=0,down=0,scales=0,bytes=0;
 std::array<Slice,4> slices{};
};
inline Layout layout(int bank,int cut){
 if(bank<0||bank>1||cut<=0||cut>=int(Width)||cut%32)throw std::invalid_argument("whole-row fragment bank/cut");
 Layout x;x.bank=bank;x.hidden_begin=bank*320;x.output_begin=bank?cut:0;x.output_rows=bank?int(Width)-cut:cut;
 x.up=320*GuRow;x.down=2*x.up;x.scales=x.down+std::size_t(x.output_rows)*DownRow;x.bytes=x.scales+16;
 x.slices={Slice{std::size_t(x.hidden_begin)*GuRow,x.gate,320*GuRow},
           Slice{Up+std::size_t(x.hidden_begin)*GuRow,x.up,320*GuRow},
           Slice{Down+std::size_t(x.output_begin)*DownRow,x.down,std::size_t(x.output_rows)*DownRow},
           Slice{Tail,x.scales,16}};
 return x;
}
inline void pack(std::span<const std::uint8_t>source,std::span<std::uint8_t>target,const Layout&x){
 if(source.size()!=Blob||target.size()!=x.bytes)throw std::invalid_argument("exact fragment extents required");
 const auto canonical=layout(x.bank,x.bank?x.output_begin:x.output_rows);
 if(std::memcmp(&x.slices,&canonical.slices,sizeof(x.slices))||x.bytes!=canonical.bytes||x.hidden_begin!=canonical.hidden_begin||x.hidden_rows!=canonical.hidden_rows||x.output_begin!=canonical.output_begin||x.output_rows!=canonical.output_rows||x.gate!=canonical.gate||x.up!=canonical.up||x.down!=canonical.down||x.scales!=canonical.scales)
  throw std::invalid_argument("fragment descriptor is not canonical");
 const auto a=reinterpret_cast<std::uintptr_t>(source.data()),b=reinterpret_cast<std::uintptr_t>(target.data());
 if(!a||!b||source.size()>UINTPTR_MAX-a||target.size()>UINTPTR_MAX-b||
    (a<b+target.size()&&b<a+source.size()))throw std::invalid_argument("fragment source/destination span or alias");
 for(const auto&s:x.slices){
  if(s.source>source.size()||s.bytes>source.size()-s.source||s.destination>target.size()||s.bytes>target.size()-s.destination)
   throw std::invalid_argument("fragment slice exceeds admitted span");
  std::memcpy(target.data()+s.destination,source.data()+s.source,s.bytes);
 }
}
struct Slot {int logical_home,logical_slot,layer;};
struct Registered {Slot source;int cut;std::array<std::uint64_t,2>offset;std::array<Layout,2>fragment;};
struct Index {
 std::vector<Registered>slots;
 std::array<std::uint64_t,2>used{};
 // Original logical slot identity/layer are explicit; never recovered from a
 // fictitious whole-blob pointer, modulo stride, routing heat or CUDA ordinal.
 void initialize(std::span<const Slot>source,std::array<std::uint64_t,2>budgets,int alignment){
  if(!slots.empty()||used[0]||used[1]||source.empty()||(alignment!=16&&alignment!=256))throw std::invalid_argument("fragment registration lifecycle/alignment");
  std::vector<Registered>pending;std::array<std::uint64_t,2>next{};
  std::array<std::vector<bool>,2>seen;for(auto&v:seen)v.resize(48*512);
  for(auto s:source){
   if(s.logical_home<0||s.logical_home>1||s.logical_slot<0||s.logical_slot>=48*512||s.layer<24*s.logical_home||s.layer>=24*s.logical_home+24||seen[s.logical_home][s.logical_slot])throw std::invalid_argument("logical slot/layer/duplicate");
   seen[s.logical_home][s.logical_slot]=true;
   // Memory-only candidate from the sealed profile analysis, NOT a measured
   // cost-balanced execution map. Admission still checks actual total budgets.
   Registered r;r.source=s;r.cut=s.layer==34||s.layer==37||s.layer==38?1472:1504;
   for(int b=0;b<2;++b){r.fragment[b]=layout(b,r.cut);r.offset[b]=next[b];const auto n=r.fragment[b].bytes;
    if(n>UINT64_MAX-(alignment-1))throw std::overflow_error("fragment stride");
    const auto stride=(n+alignment-1)&~std::uint64_t(alignment-1);
    if(next[b]>budgets[b]||stride>budgets[b]-next[b])throw std::length_error("row fragments do not fit actual arena");next[b]+=stride;
   }
   pending.push_back(r);
  }
  slots=std::move(pending);used=next;
 }
};
static_assert(Blob==2764816&&Tail==2764800);
}
