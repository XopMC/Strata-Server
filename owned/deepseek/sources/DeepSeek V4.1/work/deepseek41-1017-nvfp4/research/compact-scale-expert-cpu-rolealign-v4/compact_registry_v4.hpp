#pragma once
#include "../compact-scale-expert-rolealign-v4/compact_projection_v4.hpp"
#include "../native-nvfp4-registry-v1/borrowed_weights_v2.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
namespace ds1017_compact_rolealign_checkpoint_v4 {
using View=ds1017_compact_rolealign_v4::CompactProjectionView;
using Owner=ds1017_compact_rolealign_v4::OwnerLease;
using MemoryRegion=strata::ds41::MemoryRegion;
using RawTensor=strata::ds41::BorrowedTensor;
using RawEncoding=strata::ds41::TensorEncoding;
struct Binding {View projection;Owner owner;};
inline bool enabled(){static const bool on=[](){const char*p=std::getenv("DS1017_COMPACT_ROLEALIGN_CHECKPOINT_V4");return p&&std::strcmp(p,"1")==0;}();return on;}
inline bool fixture_enabled(){const char*p=std::getenv("DS1017_COMPACT_ROLEALIGN_CHECKPOINT_FIXTURE_V4");return p&&std::strcmp(p,"1")==0;}
// Sibling typed registry. Raw encodings0..10 use the ORIGINAL protected reader;
// encoding14 has its OWN map/descriptor and never enters legacy raw projection.
class CompactRegistry {
 std::shared_ptr<const void>owner_;std::vector<MemoryRegion>regions_;uint64_t generation_;
 std::shared_ptr<strata::ds41::BorrowedWeights>raw_;
 std::unordered_map<std::string,Binding>compact_;std::set<std::string>names_;
 mutable std::mutex mutex_;mutable bool frozen_=false;
 static void need(bool b,const char*message){if(!b)throw std::invalid_argument(message);}
 static char expert_role(const std::string&name){
  int layer=-1,expert=-1,count=0;char role=0;
  need(std::sscanf(name.c_str(),"layers.%d.ffn.experts.%d.w%c.weight%n",&layer,&expert,&role,&count)==3&&count==int(name.size())&&
   layer>=0&&layer<40&&expert>=0&&expert<384&&(role=='1'||role=='2'||role=='3'),"compact registration needs canonical main-expert weight name");
  need(name=="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+".w"+role+".weight","noncanonical compact expert identity");return role;
 }
 Owner owner_for(const View&v)const{
  const auto address=reinterpret_cast<uintptr_t>(v.data);
  const auto it=std::upper_bound(regions_.begin(),regions_.end(),address,[](uintptr_t p,const MemoryRegion&r){return p<reinterpret_cast<uintptr_t>(r.data);});
  need(it!=regions_.begin(),"compact matrix missing retained allocation");const auto&r=*(it-1);
  Owner owner{owner_,r.data,r.bytes,generation_};
  need(ds1017_compact_rolealign_v4::validate(v,owner,generation_)==ds1017_compact_rolealign_v4::Status::Ok,"compact matrix metadata/owner/generation/extent refused");return owner;
 }
 void raw_admission(const std::string&name,const RawTensor&t)const{
  need(enabled(),"compact checkpoint defaultOFF");need(!frozen_,"compact registry already frozen");
  need(!names_.contains(name),"duplicate tensor name across raw/compact registry");
  need(!(name.starts_with("layers.")&&name.find(".ffn.experts.")!=std::string::npos),"main expert weights/scales require typed compact registration");
  const int encoding=static_cast<int>(t.encoding);
  need(encoding>=0&&encoding<=10&&!t.bits&&!t.protected_native_rows,"protected tensors retain original raw precision; old11/12/13/14 raw path refused");
 }
public:
 CompactRegistry(std::shared_ptr<const void>owner,std::vector<MemoryRegion>regions,uint64_t generation)
  :owner_(std::move(owner)),regions_(std::move(regions)),generation_(generation),raw_(std::make_shared<strata::ds41::BorrowedWeights>(owner_,regions_)){
  need(enabled(),"compact checkpoint exact1 flag required");need(generation_!=0,"nonzero compact bundle generation required");
  std::sort(regions_.begin(),regions_.end(),[](const MemoryRegion&a,const MemoryRegion&b){return reinterpret_cast<uintptr_t>(a.data)<reinterpret_cast<uintptr_t>(b.data);});
 }
 CompactRegistry(const CompactRegistry&)=delete;CompactRegistry&operator=(const CompactRegistry&)=delete;
 void add_raw(const std::string&name,RawTensor tensor){
  std::lock_guard<std::mutex>lock(mutex_);raw_admission(name,tensor);const auto inserted=names_.insert(name);try{raw_->add(name,std::move(tensor));}catch(...){names_.erase(inserted.first);throw;}
 }
 void add_raw_segmented(const std::string&name,RawTensor tensor,std::vector<MemoryRegion>segments){
  std::lock_guard<std::mutex>lock(mutex_);raw_admission(name,tensor);const auto inserted=names_.insert(name);try{raw_->add_segmented(name,std::move(tensor),std::move(segments));}catch(...){names_.erase(inserted.first);throw;}
 }
 void add_compact(const std::string&name,uint32_t encoding,const std::string&codec,View v){
  std::lock_guard<std::mutex>lock(mutex_);need(enabled()&&!frozen_,"compact registry disabled/frozen");
  need(encoding==ds1017_compact_rolealign_v4::encoding_id&&codec==ds1017_compact_rolealign_v4::codec_id,"explicit encoding14/codec handshake required");
  need(!names_.contains(name),"duplicate compact/raw tensor name");const char role=expert_role(name);
  need(v.rows==uint32_t(role=='2'?5120:2304)&&v.columns==uint32_t(role=='2'?2304:5120),"compact named projection role/geometry differs");
  const auto owner=owner_for(v);
  // Initial quality-first slice validates encoded scale fields BEFORE metadata
  // publication. No codes/weights are decoded or copied. The leaf retains its
  // original per-row checks too. Cold registration cost is not a speed claim.
  for(uint32_t row=0;row<v.rows;++row)need(ds1017_compact_rolealign_v4::validate_row(v,row)==ds1017_compact_rolealign_v4::Status::Ok,"compact scale field exceeds actual matrix span");
  auto [it,inserted]=compact_.emplace(name,Binding{v,owner});need(inserted,"duplicate compact binding");
  try{names_.insert(name);}catch(...){compact_.erase(it);throw;}
 }
 void freeze()const{std::lock_guard<std::mutex>lock(mutex_);frozen_=true;}
 bool frozen()const{std::lock_guard<std::mutex>lock(mutex_);return frozen_;}
 size_t compact_count()const{std::lock_guard<std::mutex>lock(mutex_);return compact_.size();}
 Binding compact_projection(const std::string&prefix,int rows,int columns)const{
  std::lock_guard<std::mutex>lock(mutex_);need(frozen_,"compact views require checkpoint freeze before consumers");
  const auto it=compact_.find(prefix+".weight");need(it!=compact_.end(),"missing typed compact expert projection");const auto&v=it->second.projection;
  need(v.rows==uint32_t(rows)&&v.columns==uint32_t(columns),"requested compact shape differs");
  need(ds1017_compact_rolealign_v4::validate(v,it->second.owner,generation_)==ds1017_compact_rolealign_v4::Status::Ok,"compact owner generation/extent changed");return it->second;
 }
 std::shared_ptr<const strata::ds41::BorrowedWeights>raw()const{
  std::lock_guard<std::mutex>lock(mutex_);need(frozen_,"raw protected reader requires frozen compact checkpoint registry");return raw_;
 }
 const RawTensor&at(const std::string&name)const{
  std::lock_guard<std::mutex>lock(mutex_);need(frozen_,"raw access requires frozen registry");need(!compact_.contains(name),"typed encoding14 is not a raw weight tensor");return raw_->at(name);
 }
 uint64_t generation()const noexcept{return generation_;}
};
} // namespace ds1017_compact_rolealign_checkpoint_v4
