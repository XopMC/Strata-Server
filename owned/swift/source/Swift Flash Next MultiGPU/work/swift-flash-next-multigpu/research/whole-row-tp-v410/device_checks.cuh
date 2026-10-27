#pragma once
#include "row_leaves_v410.hpp"
#include <cuda_runtime.h>
namespace whole_row_tp_v410::detail {
__device__ __forceinline__ bool dspan(ConstSpan s,unsigned long long n,unsigned long long alignment=4){
 const auto p=reinterpret_cast<unsigned long long>(s.data);
 return p&&!(p&(alignment-1))&&s.bytes>=n&&s.bytes<=~0ull-p;
}
__device__ __forceinline__ bool fragment_ok(const Fragment&f,Rows r){
 const auto b=reinterpret_cast<unsigned long long>(f.allocation.data);
 const auto dw=static_cast<unsigned long long>(r.output_rows)*360;
 return dspan(f.allocation,921600+dw+16,16)&&f.allocation.bytes==921600+dw+16&&
  reinterpret_cast<unsigned long long>(f.gate.data)==b&&f.gate.bytes==460800&&
  reinterpret_cast<unsigned long long>(f.up.data)==b+460800&&f.up.bytes==460800&&
  reinterpret_cast<unsigned long long>(f.down.data)==b+921600&&f.down.bytes==dw&&
  reinterpret_cast<unsigned long long>(f.scales.data)==b+921600+dw&&f.scales.bytes==16;
}
__device__ __forceinline__ bool clean(const Plan&p){return *static_cast<const volatile unsigned*>(p.error.data)==0;}
__device__ __forceinline__ bool fail(const Plan&p,unsigned e){atomicOr(static_cast<unsigned*>(p.error.data),e);return false;}
__device__ __forceinline__ bool counts_ok(const Plan&p,int&ng,int&ne){
 if(!clean(p))return false;
 const auto*c=static_cast<const int*>(p.counts.data);ng=c[0];ne=c[1];
 if(ng<0||ng>p.capacity_groups||ne<0||ne>p.capacity_entries||ne>p.tokens*10||c[2]||c[3]||
    (ng==0)!=(ne==0))return fail(p,BadCounts);
 const auto*s=static_cast<const int*>(p.start.data);
 if(s[0]!=0||s[ng]!=ne)return fail(p,BadStart);
 return true;
}
__device__ __forceinline__ bool group_ok(const Plan&p,Rows r,int g,int ne){
 const auto*s=static_cast<const int*>(p.start.data);const int a=s[g],z=s[g+1];
 if(a<0||z<=a||z>ne)return fail(p,BadStart);
 if(!fragment_ok(static_cast<const Fragment*>(p.fragments.data)[g],r))return fail(p,BadFragment);
 const auto*dst=static_cast<const int*>(p.dst.data);const auto*tok=static_cast<const int*>(p.tok.data);
 for(int e=a;e<z;++e)if(dst[e]<0||dst[e]>=p.tokens*10||tok[e]!=dst[e]/10)return fail(p,BadRoute);
 return true;
}
}
