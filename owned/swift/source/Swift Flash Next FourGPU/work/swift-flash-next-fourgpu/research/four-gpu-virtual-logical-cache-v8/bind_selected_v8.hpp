#pragma once
#include "strata/core/expert_cache.hpp"
#include "strata/core/verify.hpp"
#include <span>
#include <stdexcept>
namespace four_logical_v8 {
inline void need(bool value,const char*why){if(!value)throw std::runtime_error(why);}
// Caller first validates the independent-control metadata against actual
// checkpoint/profile/context/KV/RoPE/reserves/spec/MTP/adapt tuple. Never use
// live free VRAM AFTER removing old pools to choose a larger logical union.
inline void bind_selected_home(strata::core::ExpertCache&cache,int home,
    std::span<const int32_t>original_global_R,std::span<const uint64_t>selected_offsets,uint64_t selected_virtual_bytes){
 need(home==0||home==1,"V8 invalid logical home");need(!cache.valid(),"V8 requires fresh metadata-only cache");
 need(original_global_R.size()==48*512&&selected_offsets.size()>1&&selected_offsets.size()<=24*512+1,"V8 selected metadata geometry");
 need(selected_offsets.front()==0&&selected_offsets.back()==selected_virtual_bytes,"V8 selected virtual extent");
 // First concrete profile is unchanged native40/guard0. A future different
 // virtual layout must be explicitly implemented, never inferred or shrunk.
 for(size_t s=1;s<selected_offsets.size();++s)need(selected_offsets[s]-selected_offsets[s-1]==2765056&&selected_offsets[s]>=selected_offsets[s-1],"V8 selected stride is unsupported");
 std::string error;cache.set_slot_guard(0);cache.set_per_layer_admission(false);
 if(!cache.open_logical_sized(std::vector<int64_t>(selected_offsets.size()-1,2764816),48,512,error))throw std::runtime_error(error);
 std::vector<int32_t>own(original_global_R.begin(),original_global_R.end());
 for(int k=0;k<48*512;++k)if(k/(24*512)!=home)own[size_t(k)]=-1;
 if(!cache.install_logical_selection(own,home*24,(home+1)*24,error))throw std::runtime_error(error);
 need(cache.logical_only()&&cache.allocated_payload_bytes()==0&&cache.bytes()==int64_t(selected_virtual_bytes),"V8 logical allocation/extent mismatch");
 for(size_t s=0;s<selected_offsets.size();++s)need(cache.slot_offset(int64_t(s))==selected_offsets[s],"V8 logical offset changed");
}
inline strata::core::VerifyHits logical_hits(const strata::core::ExpertCache&cache,const int32_t*d_res,const int32_t*h_res){
 need(cache.logical_only()&&cache.resident()==cache.slots()&&d_res&&h_res,"V8 unready logical verifier view");
 strata::core::VerifyHits hits;hits.d_res=d_res;hits.h_res=h_res;hits.blob=2764816;hits.n_slots=cache.slots();
 hits.logical_only_v8=true; // NO fake cache_base/offset pointer
 return hits;
}
}
