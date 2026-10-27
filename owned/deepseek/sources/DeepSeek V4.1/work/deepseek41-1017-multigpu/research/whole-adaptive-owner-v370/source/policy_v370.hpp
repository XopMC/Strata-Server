#pragma once
#include <array>
#include <bitset>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>
namespace ds370 {
inline void require370(bool x,const char*s){if(!x)throw std::runtime_error(s);}
struct PolicyConfig370 {unsigned physical=1078,spares=8,statics=1014;bool component=false;int component_layer=0;};
struct Offer370 {uint64_t nonce=0;int candidate=-1,victim=-1;unsigned spare=0,victim_slot=0;};
// Pure integer/state policy. All methods externallyserialized, NO CUDA/thread.
class Policy370 {
 struct Group {uint16_t id=0;uint8_t count=0;};
 struct Layer {std::array<Group,36>groups{};unsigned size=0;};
 struct Forward {std::array<Layer,40>layers{};std::bitset<15360>miss{};};
 PolicyConfig370 c_;std::array<int,15360>slot_{};std::array<bool,15360>fixed_{};
 std::vector<unsigned>free_;std::vector<Offer370>pending_;std::array<Forward,64>history_{};
 Forward current_{};std::array<uint8_t,15360>repeat_{};unsigned head_=0,history_size_=0,expected_=0,rows_=0;bool observing_=false,armed_=false;
 unsigned current_layer_=0;uint64_t next_=0,forwards_=0,last_visits_=0;bool limited_=false;struct BudgetStop{};
 void charge(unsigned n){if(last_visits_>(1ull<<20)-n)throw BudgetStop{};last_visits_+=n;}
 static int64_t cost(unsigned n){return int64_t(3)*11520*((n<=3?8:16)+2+18*n);}
 static int node(int e){return (e/384+e%384)&1;}
 unsigned layers()const{return c_.component?1:40;}
 unsigned actual(unsigned l)const{return c_.component?unsigned(c_.component_layer):l;}
 bool resident_in(const std::array<bool,15360>&r,int e)const{return r[unsigned(e)];}
 template<class F>void each_history(F&&f)const{
  for(unsigned h=0;h<history_size_;h++){const auto&v=history_[(head_+64-history_size_+h)%64];for(unsigned l=0;l<layers();l++)f(v.layers[actual(l)]);}
 }
public:
 explicit Policy370(PolicyConfig370 c,const std::vector<int>&ranked):c_(c){
  require370(c.physical>=c.spares+1&&c.spares==8&&c.statics<c.physical-c.spares,"370 physical/8spare/staticbounds");
  require370(c.component?(c.physical==16&&c.statics==0&&c.component_layer>=0&&c.component_layer<40):(c.physical==1078&&c.statics==1014),"370 strictproduction/componentgeometry");
  require370(ranked.size()>=c.physical&&ranked.size()<=15360,"370 catalogcapacity");slot_.fill(-1);std::bitset<15360>seen;
  for(int id:ranked){require370(id>=0&&id<15360&&!seen[id],"370 unique catalogid");seen[id]=true;}
  for(unsigned i=0;i<c.physical-c.spares;i++){slot_[ranked[i]]=int(i);fixed_[ranked[i]]=i<c.statics;}
  free_.reserve(c.spares);pending_.reserve(c.spares);for(unsigned i=c.physical-c.spares;i<c.physical;i++)free_.push_back(i);
 }
 bool resident(int id)const{return id>=0&&id<15360&&slot_[id]>=0;}
 int slot(int id)const{return resident(id)?slot_[id]:-1;}
 uint64_t planner_visits()const{return last_visits_;}bool planner_limited()const{return limited_;}
 unsigned active()const{return c_.physical-c_.spares;}
 unsigned free_count()const{return unsigned(free_.size());}
 uint64_t forwards()const{return forwards_;}
 const std::vector<Offer370>&pending()const{return pending_;}
 void arm(){require370(!armed_&&!observing_&&!expected_&&pending_.empty()&&!forwards_,"370 armfreshonly");armed_=true;}
 bool armed()const{return armed_;}
 void validate_before(int layer,unsigned rows,const std::array<uint8_t,384>&counts)const{
  require370(armed_&&!observing_&&layer==int(actual(expected_))&&rows>=1&&rows<=6&&(!expected_||rows==rows_),"370 forward/layer/Morder");unsigned sum=0;for(auto n:counts){require370(n<=rows,"370 Count<=M");sum+=n;}require370(sum<=rows*6,"370 top6positivecount");
 }
 void before(int layer,unsigned rows,const std::array<uint8_t,384>&counts){
  if(!armed_)return;validate_before(layer,rows,counts);if(!expected_)rows_=rows;
  auto&out=current_.layers[unsigned(layer)];out.size=0;unsigned sum=0;
  for(unsigned e=0;e<384;e++){require370(counts[e]<=rows,"370 Count<=actualM");if(counts[e]){require370(out.size<36,"370 groups<=36");int id=layer*384+int(e);out.groups[out.size++]={uint16_t(id),counts[e]};sum+=counts[e];if(!resident(id))current_.miss[id]=true;}}
  require370(sum<=rows*6,"370 positive top6routecount");current_layer_=unsigned(layer);observing_=true;
 }
 void after(bool success){
  if(!armed_)return;require370(observing_,"370 afterwithoutbefore");observing_=false;require370(success,"370 failedTask poisonsforward");
  if(++expected_!=layers())return;expected_=0;
  if(history_size_==64){const auto&old=history_[head_];for(unsigned i=0;i<15360;i++)if(old.miss[i]){require370(repeat_[i]>0,"370 repeatunderflow");repeat_[i]--;}}
  else history_size_++;
  history_[head_]=current_;for(unsigned i=0;i<15360;i++)if(current_.miss[i])repeat_[i]++;
  head_=(head_+1)%64;current_=Forward{};forwards_++;
 }
 bool boundary()const{return !observing_&&!expected_;}
 bool can_publish(uint64_t nonce,const std::array<uint8_t,384>&counts,int layer)const{
  const auto it=std::find_if(pending_.begin(),pending_.end(),[&](const auto&x){return x.nonce==nonce;});
  require370(it!=pending_.end(),"370 unknownoffernonce");return it->victim/384!=layer||!counts[unsigned(it->victim%384)];
 }
 void publish(uint64_t nonce){
  auto it=std::find_if(pending_.begin(),pending_.end(),[&](const auto&x){return x.nonce==nonce;});require370(it!=pending_.end()&&slot_[it->victim]==int(it->victim_slot)&&!resident(it->candidate),"370 publishbinding");
  slot_[it->victim]=-1;slot_[it->candidate]=int(it->spare);free_.push_back(it->victim_slot);pending_.erase(it);
 }
 void cancel(uint64_t nonce){auto it=std::find_if(pending_.begin(),pending_.end(),[&](const auto&x){return x.nonce==nonce;});require370(it!=pending_.end(),"370 cancelbinding");free_.push_back(it->spare);pending_.erase(it);}
 std::vector<Offer370>propose(unsigned limit){
  require370(armed_&&boundary()&&forwards_&&limit<=2,"370 gaponlyaftercompleteforward/cap2");std::vector<Offer370>made;made.reserve(2);last_visits_=0;limited_=false;
  try{while(made.size()<limit&&!free_.empty()){
   std::array<bool,15360>planned{},eligible{},reserved{};for(unsigned e=0;e<15360;e++)planned[e]=resident(int(e));
   for(const auto&p:pending_){planned[unsigned(p.victim)]=false;planned[unsigned(p.candidate)]=true;reserved[unsigned(p.victim)]=true;}
   for(unsigned e=0;e<15360;e++)eligible[e]=repeat_[e]>=2&&!planned[e]&&!resident(int(e));
   std::array<int64_t,15360>score{};
   each_history([&](const Layer&l){charge(l.size);std::array<int64_t,2>n{};for(unsigned j=0;j<l.size;j++){auto g=l.groups[j];if(!planned[g.id])n[node(g.id)]+=cost(g.count);}auto cur=std::max(n[0],n[1]);for(unsigned j=0;j<l.size;j++){auto g=l.groups[j];if(eligible[g.id]){auto r=node(g.id);score[g.id]+=cur-std::max(n[r]-cost(g.count),n[1-r]);}}});
   int candidate=-1;for(unsigned e=0;e<15360;e++)if(eligible[e]&&(candidate<0||score[e]>score[unsigned(candidate)]))candidate=int(e);
   if(candidate<0||score[unsigned(candidate)]<=0)break;
   int victim=-1;int64_t gain=0;
   for(unsigned v=0;v<15360;v++)if(resident(int(v))&&!fixed_[v]&&!reserved[v]){
    int64_t delta=0;each_history([&](const Layer&l){if(!l.size||(l.groups[0].id/384!=candidate/384&&l.groups[0].id/384!=int(v)/384))return;charge(l.size);std::array<int64_t,2>n{};for(unsigned j=0;j<l.size;j++){auto g=l.groups[j];if(!planned[g.id])n[node(g.id)]+=cost(g.count);}auto old=std::max(n[0],n[1]);for(unsigned j=0;j<l.size;j++){auto g=l.groups[j];if(g.id==candidate&&!planned[g.id])n[node(g.id)]-=cost(g.count);if(g.id==v&&planned[g.id])n[node(g.id)]+=cost(g.count);}delta+=old-std::max(n[0],n[1]);});
    if(delta>gain){gain=delta;victim=int(v);}
   }
   if(victim<0)break;require370(next_<std::numeric_limits<uint64_t>::max(),"370 offernonceoverflow");
   auto it=std::min_element(free_.begin(),free_.end());Offer370 offer{++next_,candidate,victim,*it,unsigned(slot_[unsigned(victim)])};free_.erase(it);pending_.push_back(offer);made.push_back(offer);
  }}catch(const BudgetStop&){limited_=true;}return made;
 }
};
}
