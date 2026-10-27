#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "strata/core/expert_source.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace model_ep_v403 {
constexpr int Max=80,Qrow=2880;
struct DevicePlan {
 int32_t counts[4],start[81],dst[80],tok[80],padding;
 unsigned long long ptr[80];
};
struct alignas(64) Frame {
 uint32_t ready=0,abort=0;
 uint64_t current_window=0,plan_window=0;
 int32_t T=0,layer=-1,error=0,reserved=0;
 uint32_t completed[2]{},status[2]{};
 DevicePlan bank[2]{};
};
static_assert(sizeof(DevicePlan)==1624);
struct Home {
 uint8_t* base=nullptr;
 uint64_t bytes=0,payload=0;
 std::vector<uint64_t> offsets;
};
class Bridge {
public:
 struct Impl;
private:
 std::unique_ptr<Impl> p_;
public:
 Bridge();~Bridge();
 Bridge(const Bridge&)=delete;Bridge&operator=(const Bridge&)=delete;
 bool enabled()const noexcept;
 bool configure(std::array<Home,2>,std::array<void*,2> verifiers,
                std::array<strata::core::GpuPlanSink*,2>,std::string&);
 bool early_root_publish(strata::core::ExpertSource&, const std::vector<int32_t>&, std::string&);
 bool rehome(strata::core::ExpertSource&, const std::vector<int32_t>&, std::string&); // private ONE-GEN: after original PP/refill, before original prompt_ms snapshot
 bool begin_request(std::string&);
 bool defer_identity_refill(int bank,int slot,bool&defer,std::string&);
 bool prompt_identity(strata::core::ExpertSource&,const std::vector<int32_t>&,std::string&);
 // Original synchronous adaptive policy; immutable logical->physical map.
 bool copy_adapt(int logical_home,int slot,const uint8_t*source,size_t bytes,std::string&);
 bool join_adapt(int logical_home,cudaStream_t original_home_stream,std::string&);
 bool quiesce(std::string&);
 bool begin_window(int T,std::string&);
 void before_dispatch(strata::core::ExpertDispatch&,const int32_t*,int64_t,int64_t,int64_t) noexcept;
 void after_dispatch(strata::core::ExpertDispatch&)noexcept;
 bool finish_request(std::string&);
};
class ExitGuard {Bridge*bridge_;public:explicit ExitGuard(Bridge*b):bridge_(b){}~ExitGuard();};
bool requested();
bool owns(const void*)noexcept;
bool captured(const void*,int T)noexcept;
bool prepare(const void*,int T,int lb,int le,
             const std::vector<strata::kernels::NativeExpertLayout>&,
             const uint8_t* q8,void* local_scratch,float* owner_hits,std::string&);
bool fork_peer(const void*,int layer,int T,cudaStream_t,std::string&);
bool join_peer(const void*,int layer,int T,cudaStream_t,std::string&);
bool enqueue(const void*,int layer,int T,const strata::kernels::NativeExpertLayout&,
             const uint8_t* q8,void* local_scratch,float* owner_hits,cudaStream_t,std::string&);
bool completed(const void*,std::string&);
void release(const void*)noexcept;

cudaError_t ingest(Frame*,DevicePlan*,int bank,int T,int layer,int ring,cudaStream_t);
cudaError_t copy_q8_if_needed(const DevicePlan*,const uint8_t*,uint8_t*,int T,cudaStream_t);
cudaError_t mark_completed(Frame*,int bank,int ring,cudaStream_t);
}
