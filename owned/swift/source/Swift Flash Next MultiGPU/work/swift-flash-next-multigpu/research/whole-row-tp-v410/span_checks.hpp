#pragma once
#include "row_leaves_v410.hpp"
#include <initializer_list>
namespace whole_row_tp_v410::detail {
inline bool span(ConstSpan s,std::uint64_t n,std::uintptr_t alignment=4) {
 const auto p=reinterpret_cast<std::uintptr_t>(s.data);
 return p && !(p&(alignment-1)) && s.bytes>=n && s.bytes<=UINTPTR_MAX-p;
}
inline ConstSpan read(Span s){return {s.data,s.bytes};}
inline bool overlap(ConstSpan a,ConstSpan b){
 const auto x=reinterpret_cast<std::uintptr_t>(a.data),y=reinterpret_cast<std::uintptr_t>(b.data);
 return x<y+b.bytes && y<x+a.bytes;
}
inline bool distinct(std::initializer_list<ConstSpan> spans){
 for(auto a=spans.begin();a!=spans.end();++a)for(auto b=a+1;b!=spans.end();++b)
  if(overlap(*a,*b))return false;
 return true;
}
inline bool plan_admitted(const Plan&p){
 if(p.tokens<1||p.tokens>8||p.capacity_groups<1||p.capacity_groups>80||p.capacity_entries<1||p.capacity_entries>80)return false;
 if(!span(p.fragments,std::uint64_t(p.capacity_groups)*sizeof(Fragment),8)||!span(p.counts,16)||
    !span(p.start,std::uint64_t(p.capacity_groups+1)*4)||!span(p.dst,std::uint64_t(p.capacity_entries)*4)||
    !span(p.tok,std::uint64_t(p.capacity_entries)*4)||!span(read(p.error),4))return false;
 return distinct({p.fragments,p.counts,p.start,p.dst,p.tok,read(p.error)});
}
inline bool outputs_separate(const Plan&p,std::initializer_list<ConstSpan> payload){
 // The parent owns the Fragment allocation list: it must separately reject any
 // packed-payload overlap with these outputs before uploading its host image.
 for(auto a:payload)for(auto b:{p.fragments,p.counts,p.start,p.dst,p.tok,read(p.error)})
  if(overlap(a,b))return false;
 return distinct(payload);
}
}
