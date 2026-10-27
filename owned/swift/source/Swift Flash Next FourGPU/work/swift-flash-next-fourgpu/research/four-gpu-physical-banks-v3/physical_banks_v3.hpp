#pragma once
#include "physical_types_v3.hpp"
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace four_banks_v3 {
inline void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
inline constexpr std::array<int32_t,Devices> DefaultSlots{4800,4000,7888,7888};

// A layout is metadata only. No payload allocation, weight read or CUDA action.
struct PhysicalLayout {
 std::array<int32_t,Devices> slots=DefaultSlots;
 std::array<uint64_t,Devices> bank_bytes{};
 std::array<std::vector<uint64_t>,Devices> offsets; // slots+1, final extent
 std::array<std::vector<int32_t>,Devices> slot_keys;
 std::vector<PhysicalRef> refs;
 static PhysicalLayout canonical(){
  PhysicalLayout x; x.refs.resize(Keys);
  std::array<int32_t,Devices> assigned{};
  for(int d=0;d<Devices;++d){
   x.bank_bytes[d]=uint64_t(x.slots[d])*SlotStride;
   x.offsets[d].resize(size_t(x.slots[d])+1);
   for(int s=0;s<=x.slots[d];++s)x.offsets[d][size_t(s)]=uint64_t(s)*SlotStride;
   x.slot_keys[d].reserve(size_t(x.slots[d]));
  }
  // Whole original-ID blocks16. Choose least normalized next-block midpoint;
  // integer cross multiplication makes ties deterministic (lower device wins).
  for(int key0=0;key0<Keys;key0+=16){
   int best=-1;
   for(int d=0;d<Devices;++d){
    const int q=x.slots[d]/16;
    if(assigned[d]==q)continue;
    if(best<0 || int64_t(2*assigned[d]+1)*(x.slots[best]/16)<
                 int64_t(2*assigned[best]+1)*q)best=d;
   }
   require(best>=0,"physical block quotas exhausted");
   ++assigned[best];
   for(int j=0;j<16;++j){
    int key=key0+j,s=int(x.slot_keys[best].size());
    x.slot_keys[best].push_back(key);
    x.refs[size_t(key)]={uint64_t(s)*SlotStride,0,best,s,BlobBytes,uint32_t(key)};
   }
  }
  x.validate();return x;
 }
 void validate()const{
  require(refs.size()==Keys,"physical coverage length");
  std::vector<uint8_t>seen(Keys,0);int64_t total=0;
  for(int d=0;d<Devices;++d){
   require(slots[d]>0&&slots[d]%16==0,"physical quota not block16");total+=slots[d];
   require(bank_bytes[d]==uint64_t(slots[d])*SlotStride,"physical bank extent");
   require(offsets[d].size()==size_t(slots[d])+1&&slot_keys[d].size()==size_t(slots[d]),"physical slot table length");
   for(int s=0;s<=slots[d];++s)require(offsets[d][size_t(s)]==uint64_t(s)*SlotStride,"physical aligned stride");
   for(int s=0;s<slots[d];++s){
    int key=slot_keys[d][size_t(s)];require(key>=0&&key<Keys,"physical key bound");
    require(++seen[size_t(key)]==1,"duplicate physical key");
    const auto&r=refs[size_t(key)];
    require(r.key==uint32_t(key)&&r.device==d&&r.physical_slot==s,"physical reverse map");
    require(r.blob_bytes==BlobBytes&&r.byte_offset==offsets[d][size_t(s)]&&r.byte_offset%256==0,"physical blob layout");
    require(r.byte_offset<=bank_bytes[d]&&r.blob_bytes<=bank_bytes[d]-r.byte_offset,"physical blob extent");
    if(s%16)require(key==slot_keys[d][size_t(s-1)]+1&&key/16==slot_keys[d][size_t(s-s%16)]/16,"physical block continuity");
    else require(key%16==0,"physical block start");
   }
  }
  require(total==Keys,"physical quota coverage");for(auto n:seen)require(n==1,"missing physical key");
 }
};

// Caller supplies already-filled, immutable physical allocations. A retained
// owner must own each allocation (or the enclosing bank-resource aggregate).
// Ready is an explicit Root-side positive upload/drain/readback attestation;
// setting it does not perform or prove CUDA publication in this host library.
struct BankBinding {
 const unsigned char*base=nullptr;
 uint64_t bytes=0;
 std::shared_ptr<const void>owner;
 bool payload_ready=false;
};
namespace detail {
struct PhysicalState {
 PhysicalLayout layout;std::array<BankBinding,Devices>banks;uint64_t epoch=0;
 std::mutex mutex;size_t leases=0;bool retired=false;
};
struct PhysicalHold {
 std::shared_ptr<PhysicalState>state;
 explicit PhysicalHold(std::shared_ptr<PhysicalState>s):state(std::move(s)){}
 ~PhysicalHold(){std::lock_guard<std::mutex>lock(state->mutex);--state->leases;}
};
}
class PhysicalLease {
 std::shared_ptr<detail::PhysicalHold>hold_;
 explicit PhysicalLease(std::shared_ptr<detail::PhysicalHold>x):hold_(std::move(x)){}
 friend class PhysicalBanks;
 const detail::PhysicalState&state()const{validate_live();return *hold_->state;}
public:
 PhysicalLease()=default;
 void validate_live()const{
  require(bool(hold_),"missing physical lease");
  std::lock_guard<std::mutex>lock(hold_->state->mutex);
  require(!hold_->state->retired,"retired physical lease");
 }
 uint64_t epoch()const{return state().epoch;}
 std::span<const PhysicalRef>host_refs()const{return state().layout.refs;}
 const PhysicalRef&resolve(int key)const{require(key>=0&&key<Keys,"resolve key bound");return state().layout.refs[size_t(key)];}
 const unsigned char*base(int device)const{require(device>=0&&device<Devices,"base device bound");return state().banks[size_t(device)].base;}
 uint64_t bytes(int device)const{require(device>=0&&device<Devices,"bytes device bound");return state().banks[size_t(device)].bytes;}
 int32_t slots(int device)const{require(device>=0&&device<Devices,"slots device bound");return state().layout.slots[size_t(device)];}
 std::span<const uint64_t>offsets(int device)const{require(device>=0&&device<Devices,"offset device bound");return state().layout.offsets[size_t(device)];}
 std::shared_ptr<const void>retain()const{validate_live();return hold_;}
 // published_refs is a completed/readback-qualified DEVICE table containing
 // byte-identical host_refs(). Its allocation must be retained by the caller.
 PhysicalDeviceView device_view(const PhysicalRef*published_refs,uint64_t published_epoch)const{
  require(published_refs&&published_epoch==epoch(),"unpublished physical device view");
  PhysicalDeviceView v;v.refs=published_refs;v.physical_epoch=epoch();
  for(int d=0;d<Devices;++d){v.base[d]=base(d);v.bank_bytes[d]=bytes(d);}return v;
 }
};
class PhysicalBanks {
 std::shared_ptr<detail::PhysicalState>s_;
public:
 PhysicalBanks(PhysicalLayout layout,std::array<BankBinding,Devices>banks,uint64_t epoch){
  layout.validate();require(epoch!=0,"zero physical epoch");
  for(int d=0;d<Devices;++d){const auto&b=banks[d];
   require(b.base&&uintptr_t(b.base)%256==0&&b.owner&&b.payload_ready,"unready physical bank");
   require(b.bytes==layout.bank_bytes[d],"physical binding extent mismatch");}
  for(auto&r:layout.refs)r.physical_epoch=epoch;
  s_=std::make_shared<detail::PhysicalState>();s_->layout=std::move(layout);s_->banks=std::move(banks);s_->epoch=epoch;
 }
 PhysicalLease lease()const{
  std::lock_guard<std::mutex>lock(s_->mutex);require(!s_->retired,"physical bank retired");
  auto h=std::make_shared<detail::PhysicalHold>(s_);++s_->leases;return PhysicalLease(std::move(h));
 }
 // Only after all captured graphs/consumers are positively drained/destroyed
 // and all their leases released. No destructor touches a borrowed stream.
 void retire(bool consumers_positively_drained){
  std::lock_guard<std::mutex>lock(s_->mutex);
  require(consumers_positively_drained&&s_->leases==0,"physical retirement has live consumers");
  s_->retired=true;for(auto&b:s_->banks)b.owner.reset();
 }
};
}
