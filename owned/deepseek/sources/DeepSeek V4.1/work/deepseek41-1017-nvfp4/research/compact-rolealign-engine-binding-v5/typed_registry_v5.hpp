#pragma once
#include "engram_q4_record_v5.h"
#include "strata/ds41/borrowed_rows.hpp"
#include <cmath>
#include <bit>
#include <cfenv>
#include <xmmintrin.h>
#include "../compact-scale-expert-cpu-rolealign-v4/compact_registry_v4.hpp"
#include <map>
#include <limits>
namespace ds1017_rolealign_binding_v5 {
static_assert(std::endian::native==std::endian::little);
using MemoryRegion=strata::ds41::MemoryRegion;
using RawTensor=strata::ds41::BorrowedTensor;
using RawEncoding=strata::ds41::TensorEncoding;
using ExpertRegistry=ds1017_compact_rolealign_checkpoint_v4::CompactRegistry;
using View=ds1017_compact_rolealign_v4::CompactProjectionView;
struct Engram {ds1017_engram_q4_record_v5 record;std::string overlay_name;};
inline bool binding_enabled_v5(){const char*x=std::getenv("DS1017_COMPACT_ROLEALIGN_BINDING_V5");return x&&!std::strcmp(x,"1");}
inline bool fixture_enabled_v5(){const char*x=std::getenv("DS1017_ROLEALIGN_BINDING_FIXTURE_V5");return x&&!std::strcmp(x,"1");}
class Registry {
 std::shared_ptr<const void>owner_;std::vector<MemoryRegion>regions_;
 std::shared_ptr<ExpertRegistry>experts_;std::shared_ptr<strata::ds41::BorrowedWeights>raw_;
 std::map<std::string,Engram>engram_;bool frozen_=false,poisoned_=false;
 mutable std::mutex mutex_;
 static void need(bool b,const char*s){if(!b)throw std::invalid_argument(s);}
 void open()const{need(!frozen_&&!poisoned_,"typed V5 registration frozen/poisoned");}
 void contained(const std::vector<MemoryRegion>&segments,size_t expected)const{
  need(!segments.empty()&&segments.size()<=100000,"bounded retained segment list required");size_t total=0;
  for(const auto&s:segments){const auto p=reinterpret_cast<uintptr_t>(s.data);need(s.data&&s.bytes&&p+s.bytes>=p&&s.bytes<=expected-total,"segment extent overflow");bool found=false;
   for(const auto&r:regions_){const auto b=reinterpret_cast<uintptr_t>(r.data);if(p>=b&&p+s.bytes<=b+r.bytes){found=true;break;}}
   need(found,"typed source segment outside retained owner region");total+=s.bytes;
  }need(total==expected,"typed logical byte extent differs");
 }
public:
 Registry(std::shared_ptr<const void>o,std::vector<MemoryRegion>r,uint64_t g):owner_(std::move(o)),regions_(std::move(r)),experts_(std::make_shared<ExpertRegistry>(owner_,regions_,g)),raw_(std::make_shared<strata::ds41::BorrowedWeights>(owner_,regions_)){need(binding_enabled_v5(),"typed V5 binding defaultOFF");}
 void add_raw(const std::string&n,RawTensor t){std::lock_guard lock(mutex_);open();need(!n.ends_with(".rolealign_native_rows"),"reserved typed overlay name");try{experts_->add_raw(n,t);raw_->add(n,std::move(t));}catch(...){poisoned_=true;throw;}}
 void add_raw_segmented(const std::string&n,RawTensor t,std::vector<MemoryRegion>s){std::lock_guard lock(mutex_);open();need(!n.ends_with(".rolealign_native_rows"),"reserved typed overlay name");try{experts_->add_raw_segmented(n,t,s);raw_->add_segmented(n,std::move(t),std::move(s));}catch(...){poisoned_=true;throw;}}
 void add_compact(const std::string&n,uint32_t e,const std::string&c,View v){std::lock_guard lock(mutex_);open();experts_->add_compact(n,e,c,v);}
 void add_engram(const std::string&name,uint32_t encoding,const std::string&codec,const ds1017_engram_q4_record_v5&v,std::vector<MemoryRegion>base,std::vector<MemoryRegion>overlay){
  std::lock_guard lock(mutex_);open();const char*flag=std::getenv("DS1017_ROLEALIGN_ENGRAM_Q4_V5");need(flag&&!std::strcmp(flag,"1"),"typed Engram15 defaultOFF");
  const bool first=name=="layers.1.engram.embed.weight",second=name=="layers.14.engram.embed.weight";
  need(first||second,"typed Engram15 only source layers1/14");
  need(encoding==15&&codec==DS1017_ENGRAM_Q4_CODEC_V5&&v.encoding==15&&v.struct_size==64&&v.version==1&&!v.reserved&&!v.reserved2&&v.columns==256&&v.row_bytes==144&&v.bits==4&&v.group_size==32&&v.overlay_row_bytes==272,"typed Engram15 descriptor/codec mismatch");
  const uint64_t actual=first?384006168ull:384016682ull;
  need(v.rows==actual||(fixture_enabled_v5()&&v.rows>=1&&v.rows<=8),"Engram15 actual source row geometry differs");
  need(v.overlay_rows<=v.rows&&v.overlay_rows<=2000000,"bounded native overlay count");need(!engram_.contains(name),"duplicate typed Engram");
  contained(base,size_t(v.rows)*144);if(v.overlay_rows)contained(overlay,size_t(v.overlay_rows)*272);else need(overlay.empty(),"empty overlay requires no segments");
  // Internal code11 is ONLY the existing base-row decoder implementation detail;
  // no public/raw11 admission, expert11, or precision-selector bypass is added.
  try{
   raw_->add_segmented(name,{RawEncoding::Uniform,{int64_t(v.rows),256},nullptr,size_t(v.rows)*144,4,false},std::move(base));
   const auto overlay_name=name+".rolealign_native_rows";
   if(v.overlay_rows){raw_->add_segmented(overlay_name,{RawEncoding::U8,{int64_t(v.overlay_rows*272)},nullptr,size_t(v.overlay_rows)*272,0,false},std::move(overlay));
    int64_t last=-1;std::vector<uint8_t>scratch;
    for(uint64_t i=0;i<v.overlay_rows;++i){int64_t id;std::memcpy(&id,raw_->read_range(overlay_name,size_t(i)*272,8,scratch),8);need(id>last&&id>=0&&uint64_t(id)<v.rows,"native overlay IDs must be sorted unique source rows");last=id;}
   }
   engram_.emplace(name,Engram{v,overlay_name});
  }catch(...){poisoned_=true;throw;}
 }
 void freeze()const{std::lock_guard lock(mutex_);need(!poisoned_,"partially registered metadata is poisoned");experts_->freeze();const_cast<Registry*>(this)->frozen_=true;}
 size_t compact_count()const{return experts_->compact_count();}
 auto compact_projection(const std::string&p,int n,int k)const{return experts_->compact_projection(p,n,k);}
 std::shared_ptr<const ExpertRegistry>expert_registry()const{std::lock_guard lock(mutex_);need(frozen_&&!poisoned_,"expert registry requires typed checkpoint freeze");return experts_;}
 std::shared_ptr<const strata::ds41::BorrowedWeights>raw()const{std::lock_guard lock(mutex_);need(frozen_&&!poisoned_,"protected raw reader requires freeze");return raw_;}
 const RawTensor&at(const std::string&n)const{return raw()->at(n);}
 bool is_engram15(const std::string&n)const{std::lock_guard lock(mutex_);need(frozen_&&!poisoned_,"typed Engram lookup requires freeze");return engram_.contains(n);}
 const Engram*engram(const std::string&n)const{std::lock_guard lock(mutex_);need(frozen_&&!poisoned_,"typed Engram lookup requires freeze");auto i=engram_.find(n);return i==engram_.end()?nullptr:&i->second;}
};
class Rows {
 std::shared_ptr<const Registry>registry_;std::shared_ptr<const strata::ds41::BorrowedWeights>raw_;strata::ds41::BorrowedRows base_;
 static void need(bool b,const char*s){if(!b)throw std::invalid_argument(s);}
 static uint16_t rounded(float f){need(std::isfinite(f),"nonfinite native overlay result");uint32_t u;std::memcpy(&u,&f,4);auto y=uint16_t((u+0x7fff+((u>>16)&1))>>16);need((y&0x7f80)!=0x7f80,"native overlay BF16 overflow");return y;}
 static float fp8(uint8_t b){int e=(b>>3)&15,m=b&7;need(!(e==15&&m==7),"nonfinite native overlay FP8");float v=e?std::ldexp(1.f+float(m)/8.f,e-7):std::ldexp(float(m),-9);return b&128?-v:v;}
public:
 explicit Rows(std::shared_ptr<const Registry>r):registry_(std::move(r)),raw_(registry_->raw()),base_(raw_){}
 std::vector<uint16_t>gather_bf16(const std::string&name,const std::vector<int64_t>&ids,size_t maximum_bytes=64ull<<20)const{
  need(std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&0xe040)==0,"row decode requires original FTZ0/DAZ0 nearest environment");
  const auto*entry=registry_->engram(name);
  if(!entry)return base_.gather_bf16(name,ids,maximum_bytes);
  need(maximum_bytes<=64ull<<20&&maximum_bytes>=2&&ids.size()<=maximum_bytes/(256*2),"bounded typed Engram output");
  std::vector<uint16_t>out(ids.size()*256);std::vector<int64_t>base_ids;std::vector<size_t>base_slots;
  std::vector<uint8_t>scratch;auto id_at=[&](uint64_t at){int64_t id;std::memcpy(&id,raw_->read_range(entry->overlay_name,size_t(at)*272,8,scratch),8);return id;};
  for(size_t q=0;q<ids.size();++q){need(ids[q]>=0&&uint64_t(ids[q])<entry->record.rows,"typed Engram ID outside source table");uint64_t lo=0,hi=entry->record.overlay_rows;while(lo<hi){const auto m=lo+(hi-lo)/2;if(id_at(m)<ids[q])lo=m+1;else hi=m;}
   if(lo==entry->record.overlay_rows||id_at(lo)!=ids[q]){base_ids.push_back(ids[q]);base_slots.push_back(q);continue;}
   const auto*p=raw_->read_range(entry->overlay_name,size_t(lo)*272,272,scratch);
   for(int j=0;j<256;++j){const uint8_t scale=p[264+j/32];need(scale!=255,"nonfinite native overlay E8M0");out[q*256+j]=rounded(fp8(p[8+j])*std::ldexp(1.f,int(scale)-127));}
  }
  // Overlay hits never consume the lossy base row. Remaining rows use the EXACT
  // existing Q4/BF16 decoder; only a complete private result is returned.
  if(!base_ids.empty()){auto base=base_.gather_bf16(name,base_ids,maximum_bytes);for(size_t i=0;i<base_slots.size();++i)std::memcpy(out.data()+base_slots[i]*256,base.data()+i*256,512);}
  return out;
 }
};
}
