#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::ds41 {
struct NativeImageRun {int first,count;bool image;int feature_first;};
class NativeImagePlan {
public:
 const std::vector<int> positions;const int capacity;
 NativeImagePlan(std::vector<int>p,int cap):positions(std::move(p)),capacity(cap){
  if(cap<1||cap>1048576||positions.size()>32768)throw std::invalid_argument("image position budget exceeded");
  for(size_t i=0;i<positions.size();i++)if(positions[i]<0||positions[i]>=cap||(i&&positions[i]<=positions[i-1]))throw std::invalid_argument("image positions must be sorted, unique and inside context");
 }
 bool contains(int position)const{return std::binary_search(positions.begin(),positions.end(),position);}
 std::vector<NativeImageRun> runs(int first,int count)const{
  if(first<0||count<1||first>capacity-count)throw std::invalid_argument("image subrange outside context");
  std::vector<NativeImageRun>out;int pos=first,end=first+count;
  auto it=std::lower_bound(positions.begin(),positions.end(),first);
  while(pos<end){
   if(it==positions.end()||*it>pos){int stop=it==positions.end()?end:std::min(end,*it);out.push_back({pos-first,stop-pos,false,0});pos=stop;}
   else{int start=pos,index=int(it-positions.begin());while(it!=positions.end()&&*it==pos&&pos<end){++pos;++it;}out.push_back({start-first,pos-start,true,index});}
  }return out;
 }
};
inline std::vector<int64_t> native_ngram_lookback(const std::vector<int32_t>&history,int pos,int n,int64_t pad){
 if(pos<0||pos>=int(history.size())||n<1||n>4)throw std::invalid_argument("invalid native ngram lookback");
 std::vector<int64_t>tokens(n);bool dead=false;
 for(int shift=0;shift<n;shift++){dead|=pos<shift||(pos>=shift&&history[pos-shift]<0);tokens[shift]=dead?pad:history[pos-shift];}
 return tokens;
}
}
