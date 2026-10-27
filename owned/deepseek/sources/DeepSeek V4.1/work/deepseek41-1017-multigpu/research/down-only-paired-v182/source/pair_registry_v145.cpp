#include "pair_registry_v145.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
namespace strata::ds41::pair145 {
namespace {
struct Entry {std::weak_ptr<BorrowedWeights>owner;std::unordered_map<std::string,Ds143PairDescriptor>matrices;std::unordered_set<const void*>pointers;uint64_t generation=0,epoch=0;std::array<uint64_t,12>upload{145,139,0,0,0,0,0,0,0,0,0,0};};
std::mutex mutex;
std::unordered_map<const BorrowedWeights*,Entry>entries;
[[noreturn]]void bad(const char*s){throw std::invalid_argument(std::string("pair145 registry: ")+s);}
Entry*find(const BorrowedWeights&owner){
 auto it=entries.find(&owner);if(it==entries.end())return nullptr;
 auto live=it->second.owner.lock();
 if(!live||live.get()!=&owner){entries.erase(it);return nullptr;}
 return &it->second;
}
void validate(const std::string&name,const Ds143PairDescriptor&d){
 int layer=-1,expert=-1,n=0;char p=0;
 if(std::sscanf(name.c_str(),"layers.%d.ffn.experts.%d.w%c.weight%n",&layer,&expert,&p,&n)!=3||n!=int(name.size())||layer<0||layer>=40||expert<0||expert>=384||(p!='1'&&p!='2'&&p!='3'))bad("canonical expert name required");
 const int role=p=='1'?0:p=='3'?1:2;
 const std::string expected="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+".w"+p+".weight";
 if(name!=expected)bad("noncanonical numeric name spelling");
 if(d.struct_bytes!=sizeof(d)||d.layout_tag!=139||d.bits||!d.data||d.bytes!=MatrixBytes||
    (reinterpret_cast<uintptr_t>(d.data)&((2u<<20)-1))||d.layer!=layer||d.expert!=expert||d.role!=role||d.reserved||
    d.rows!=(role==2?5120:2304)||d.columns!=(role==2?2304:5120)||!d.owner_generation||!d.bank_format_epoch)bad("invalid typed139 descriptor");
}
}
bool enabled(){static const bool value=[](){const char*v=std::getenv("DS1017_PAIR32_REGISTRY_V145");return v&&std::strcmp(v,"139")==0;}();return value;}
void add(const std::shared_ptr<BorrowedWeights>&owner,const std::string&name,const Ds143PairDescriptor&d){
 if(!enabled()||!owner)bad("explicit139 startup opt-in and retained owner required");validate(name,d);
 std::lock_guard<std::mutex>lock(mutex);
 Entry*old=find(*owner);
 if(!old){
  // Reap expired metadata once at startup of a new registry; no RAM owner is
  // retained, and repeated server reloads do not leak old name tables.
  for(auto it=entries.begin();it!=entries.end();)if(it->second.owner.expired())it=entries.erase(it);else ++it;
  Entry value;value.owner=owner;entries.emplace(owner.get(),std::move(value));old=&entries.at(owner.get());
 }
 auto&e=*old;if(e.matrices.count(name)||e.pointers.count(d.data))bad("duplicate pair name/allocation");
 if(!e.matrices.empty()&&(e.generation!=d.owner_generation||e.epoch!=d.bank_format_epoch))bad("mixed owner generation/format epoch");
 // Stage sidecar allocation before changing original registry; rollback if its
 // region/name/byte validation refuses. No callback or CUDA while lock held.
 e.matrices.emplace(name,d);
 try{e.pointers.insert(d.data);}catch(...){e.matrices.erase(name);throw;}
 try{owner->add(name,{Encoding,{d.rows,d.columns},static_cast<const uint8_t*>(d.data),size_t(d.bytes),0,false});}
 catch(...){e.pointers.erase(d.data);e.matrices.erase(name);throw;}
 e.generation=d.owner_generation;e.epoch=d.bank_format_epoch;
}
bool contains(const BorrowedWeights&owner,const std::string&prefix){std::lock_guard<std::mutex>lock(mutex);auto*e=find(owner);return e&&e->matrices.count(prefix+".weight");}
Ds143PairDescriptor projection(const BorrowedWeights&owner,const std::string&prefix,int rows,int columns){
 if(!enabled())bad("139 reader disabled");std::lock_guard<std::mutex>lock(mutex);auto*e=find(owner);if(!e)bad("missing retained typed registry");
 auto it=e->matrices.find(prefix+".weight");if(it==e->matrices.end())bad("missing paired projection");const auto d=it->second;
 if(d.rows!=rows||d.columns!=columns)bad("logical projection shape differs");return d;
}
std::array<uint64_t,8>info(const BorrowedWeights&owner){std::lock_guard<std::mutex>lock(mutex);auto*e=find(owner);const uint64_t n=e?e->matrices.size():0;return {145,139,n,n*MatrixBytes,e?e->generation:0,e?e->epoch:0,uint64_t(enabled()),0};}
void record_prefill(const BorrowedWeights&owner,int cap,int sub){
 std::lock_guard<std::mutex>lock(mutex);auto*e=find(owner);if(!e||e->matrices.size()!=46080||cap<1||cap>524288||sub<1||sub>cap)bad("actual paired prefill geometry required");
 auto&v=e->upload;++v[2];v[6]=TripleBytes;v[7]=cap;v[8]=sub;v[9]=TripleBytes;
}
void record_inverse(const BorrowedWeights&owner,uint64_t calls,uint64_t bytes){
 std::lock_guard<std::mutex>lock(mutex);auto*e=find(owner);if(!e||!e->upload[2]||calls!=3||bytes!=TripleBytes)bad("actual triple inverse counters required");
 auto&v=e->upload;v[3]+=calls;v[4]+=bytes;v[5]+=bytes;
}
std::array<uint64_t,12>upload_info(const BorrowedWeights&owner){std::lock_guard<std::mutex>lock(mutex);auto*e=find(owner);return e?e->upload:std::array<uint64_t,12>{145,139,0,0,0,0,0,0,0,0,0,0};}
}
