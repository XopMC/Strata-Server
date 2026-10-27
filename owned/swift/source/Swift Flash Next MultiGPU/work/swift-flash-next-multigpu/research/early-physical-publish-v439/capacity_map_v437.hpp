#pragma once
#include <array>
#include <cstddef>
#include <span>
#include <vector>
#include <string>
namespace capacity_map_v437 {
// A calibrated swap is an indivisible pair. If either endpoint is absent at
// this capacity, BOTH original slots remain identity. Never truncate one side.
template<class Mapping>
bool build(const std::array<size_t,2>&slots,std::span<const int>a,std::span<const int>b,Mapping&out,size_t&pairs,std::string&error){
 out={};pairs=0;auto fail=[&](const char*e){out={};pairs=0;error=e;return false;};
 if(a.size()!=b.size()||a.empty()||slots[0]==0||slots[1]==0||slots[0]>24*512||slots[1]>24*512)return fail("V437 invalid complete capacity geometry");
 std::vector<bool>seen0(7460),seen1(6645);
 for(int h=0;h<2;++h)for(size_t z=0;z<slots[h];++z)out[h].push_back({h,int(z)});
 for(size_t i=0;i<a.size();++i){
  if(a[i]<0||b[i]<0||a[i]>=7460||b[i]>=6645||seen0[a[i]]||seen1[b[i]])return fail("V437 invalid calibrated disjoint pair");
  seen0[a[i]]=true;seen1[b[i]]=true;
  if(size_t(a[i])<slots[0]&&size_t(b[i])<slots[1]){out[0][a[i]]={1,b[i]};out[1][b[i]]={0,a[i]};++pairs;}
 }
 if(!pairs)return fail("V437 no admitted concurrent expert placement pairs");
 return true;
}
}
