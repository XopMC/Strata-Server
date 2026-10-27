#pragma once
#include "strata/ds41/native_experts.hpp"
#include "strata/ds41/native_prefill.hpp"
#include <array>
#include <map>
struct CUstream_st;
namespace strata::ds41 {
// Single serial decode caller. Exact packed weights; GPU FFN arithmetic is an
// explicit candidate and must be included in the final quality measurements.
class NativeGpuExperts {
 struct Entry {uint8_t*storage=nullptr;size_t bytes=0;int arena=0;NativeDeviceProjection projections[3];};
 struct Arena {int device=0;size_t budget=0,used=0;void*workspace=nullptr;uint16_t*x=nullptr,*output=nullptr,*host_output=nullptr;float*alpha=nullptr;CUstream_st* stream=nullptr;uint64_t hits=0;};
 std::shared_ptr<const BorrowedWeights>weights_;
 std::map<std::pair<int,int>,Entry>entries_;
 std::vector<Arena>arenas_;
 size_t budget_=0,used_=0,workspace_bytes_=0;void*workspace_=nullptr;
 
 uint64_t hits_=0,misses_=0;
 std::array<uint64_t,4> output16_counts_v252_{}; // entries,H2Dbytes,originalrowbytes,maxdevtemp

 void load(int layer,int expert,int preferred);
 void release()noexcept;
public:
 NativeGpuExperts(std::shared_ptr<const BorrowedWeights>,size_t budget,const std::vector<std::array<uint64_t,384>>&frequencies);
 ~NativeGpuExperts();
 std::vector<uint16_t>run(NativeExperts&,int layer,const std::vector<uint16_t>&input,const std::vector<int64_t>&ids,const std::vector<float>&alpha);
 std::vector<uint16_t>run_many(NativeExperts&,int layer,const std::vector<uint16_t>&input,const std::vector<int64_t>&ids,const std::vector<float>&alpha);
 std::array<uint64_t,5>statistics()const{return{hits_,misses_,uint64_t(entries_.size()),used_,budget_};}
 std::vector<std::array<uint64_t,4>>device_statistics()const;
 std::array<uint64_t,4>output16_info_v252()const{return output16_counts_v252_;}
};
// Explicit owner used by a serial server to retain expert weights across fresh
// context engines. KV, attention state and history are never stored here.
struct NativeExpertCachePool {
 std::shared_ptr<NativeGpuExperts>cache;
 std::weak_ptr<const BorrowedWeights>weights;
 size_t requested=0;
 std::string devices;
};
}
