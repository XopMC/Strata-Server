#pragma once
#include "fragment_layout.hpp"
#include <algorithm>
#include <climits>
#include <cstdlib>

namespace row_fragment_v410 {
struct Schedule {std::array<int,48>cut{};Index index;int changed_layers=0;};
inline Schedule fit_schedule(std::span<const Slot>source,std::array<std::uint64_t,2>budget){
 if(source.empty()||source.size()>48*512||budget[0]>UINT64_MAX-budget[1])throw std::invalid_argument("bounded source/budgets");
 constexpr std::uint64_t step=32*DownRow,baseline=921600+1504*DownRow+16,per_pair=Blob+16;
 const auto total=source.size()*per_pair,base=source.size()*baseline;
 if(budget[0]+budget[1]<total)throw std::length_error("numeric fragments exceed both arenas");
 const auto lo=total>budget[1]?total-budget[1]:0,hi=std::min<std::uint64_t>(total,budget[0]);
 if(lo>hi)throw std::length_error("incompatible fragment budget interval");
 std::array<int,48>population{};std::array<std::vector<bool>,2>seen;for(auto&v:seen)v.resize(48*512);
 for(auto s:source){if(s.logical_home<0||s.logical_home>1||s.logical_slot<0||s.logical_slot>=48*512||s.layer<s.logical_home*24||s.layer>=s.logical_home*24+24||seen[s.logical_home][s.logical_slot])throw std::invalid_argument("logical home/slot/layer/duplicate");seen[s.logical_home][s.logical_slot]=true;++population[s.layer];}
 Schedule result;result.cut.fill(1504);std::uint64_t mask=0;
 if(base<lo||base>hi){
  const bool increase=base<lo;const auto lower=increase?lo-base:base-hi,upper=increase?hi-base:base-lo;
  const auto first=(lower+step-1)/step,last=std::min<std::uint64_t>(source.size(),upper/step);
  if(first>last)throw std::length_error("no32row-aligned payload fit");
  struct Choice {int count=INT_MAX;std::uint64_t mask=0;};std::vector<Choice>dp(source.size()+1);dp[0]={0,0};
  for(int layer=0;layer<48;++layer)if(population[layer])for(std::size_t s=source.size();s>=std::size_t(population[layer]);--s){
   const auto&from=dp[s-population[layer]];if(from.count==INT_MAX)continue;Choice candidate{from.count+1,from.mask|(1ull<<layer)};
   if(candidate.count<dp[s].count||(candidate.count==dp[s].count&&candidate.mask<dp[s].mask))dp[s]=candidate;
  }
  int best=INT_MAX;std::uint64_t best_slack=0;
  for(auto s=first;s<=last;++s)if(dp[s].count!=INT_MAX){const auto used=increase?base+s*step:base-s*step;const auto slack=std::min(used-lo,hi-used);
   if(dp[s].count<best||(dp[s].count==best&&(slack>best_slack||(slack==best_slack&&dp[s].mask<mask)))){best=dp[s].count;best_slack=slack;mask=dp[s].mask;}
  }
  if(best==INT_MAX)throw std::length_error("layer-common32row cuts cannot fit current quotas");result.changed_layers=best;
  for(int l=0;l<48;++l)if(mask&(1ull<<l))result.cut[l]=increase?1536:1472;
 }
 for(auto s:source){Registered entry;entry.source=s;entry.cut=result.cut[s.layer];
  for(int b=0;b<2;++b){entry.fragment[b]=layout(b,entry.cut);entry.offset[b]=result.index.used[b];const auto n=entry.fragment[b].bytes;
   if(result.index.used[b]>budget[b]||n>budget[b]-result.index.used[b])throw std::length_error("final fragment registration exceeds budget");result.index.used[b]+=n;
  }result.index.slots.push_back(entry);
 }
 return result;
}
}
