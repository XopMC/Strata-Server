#pragma once
// New uniquely named metadata builder. The original V3 type, canonical()
// definition, validation, physical offsets and lease/publication ABI are intact.
#include "../four-gpu-physical-banks-v3/physical_banks_v3.hpp"
#include <array>
#include <cstddef>
namespace four_owner_affine_v17 {
namespace B=four_banks_v3;
static_assert(B::Layers==48&&B::ExpertsPerLayer==512&&B::Keys==24576&&B::Devices==4);
static_assert(B::BlobBytes==2764816&&B::SlotStride==2765056);
static_assert(B::DefaultSlots==std::array<int32_t,4>{4800,4000,7888,7888});
inline B::PhysicalLayout make_owner_affine_layout_v17(){
 B::PhysicalLayout x;x.refs.resize(B::Keys);
 for(int d=0;d<4;++d){
  x.bank_bytes[size_t(d)]=uint64_t(x.slots[size_t(d)])*B::SlotStride;
  x.offsets[size_t(d)].resize(size_t(x.slots[size_t(d)])+1);
  for(int s=0;s<=x.slots[size_t(d)];++s)x.offsets[size_t(d)][size_t(s)]=uint64_t(s)*B::SlotStride;
  x.slot_keys[size_t(d)].reserve(size_t(x.slots[size_t(d)]));
 }
 for(int home=0;home<2;++home){
  const std::array<int,3>devices{home,2,3};
  const std::array<int,3>quota=home==0?std::array<int,3>{300,234,234}:std::array<int,3>{250,259,259};
  std::array<int,3>assigned{};
  for(int block=0;block<768;++block){
   int best=-1;
   for(int i=0;i<3;++i){
    if(assigned[size_t(i)]==quota[size_t(i)])continue;
    // Least normalized next-block midpoint. All operations are exact integer
    // metadata arithmetic; ties retain the lower visible device ordinal.
    if(best<0||int64_t(2*assigned[size_t(i)]+1)*quota[size_t(best)]<
               int64_t(2*assigned[size_t(best)]+1)*quota[size_t(i)])best=i;
   }
   B::require(best>=0,"V17 owner-affine block quota exhausted");
   ++assigned[size_t(best)];const int device=devices[size_t(best)];
   const int key0=home*12288+block*16;
   for(int j=0;j<16;++j){int key=key0+j;int slot=int(x.slot_keys[size_t(device)].size());
    B::require(slot<x.slots[size_t(device)],"V17 owner-affine physical slot overflow");
    x.slot_keys[size_t(device)].push_back(key);
    x.refs[size_t(key)]={uint64_t(slot)*B::SlotStride,0,device,slot,B::BlobBytes,uint32_t(key)};
   }
  }
  B::require(assigned==quota,"V17 owner-affine block coverage");
 }
 x.validate(); // original unchanged V3 validator, exhaustive24576 unique keys
 for(int key=0;key<B::Keys;++key){auto d=x.refs[size_t(key)].device;
  B::require(d>=2||d==key/12288,"V17 nonlocal key in dense-owner bank");}
 return x;
}
} // namespace four_owner_affine_v17
