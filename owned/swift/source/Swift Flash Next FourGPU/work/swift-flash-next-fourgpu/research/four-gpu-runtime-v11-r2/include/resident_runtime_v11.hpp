#pragma once
#include "../../four-gpu-physical-banks-v3/physical_types_v3.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
namespace resident_runtime_v11 {
struct StageView {
 const void*self=nullptr;
 int device=-1,lb=0,le=0,max_t=0;
 const float*original_x=nullptr;
 const void*native_q8=nullptr;
 const int32_t*ids=nullptr;
 float*parts=nullptr,*native_hits=nullptr;
 const int32_t*logical_res=nullptr;
 cudaStream_t stream=nullptr;
};
struct Config {
 std::array<four_banks_v3::PhysicalDeviceView,4>physical_views{};
 std::shared_ptr<const void>physical_owner;
 const int32_t*host_res=nullptr;
 size_t host_res_count=0;
 uint64_t logical_epoch=1;
 int adapt_swaps=0;
};
struct Rows {const int32_t*dst=nullptr;const int32_t*count=nullptr;const uint32_t*error=nullptr;};
// Throw on refused admission; no fallback. Registration is metadata-only.
void register_stage(const StageView&);
// Called AFTER this Verifier's parent graphs are destroyed, BEFORE its stream
// and buffers disappear; positively retires only that stage's retained sources.
void unregister_stage(const void*self);
// Both stages must be registered. Allocates/captures all source bodies before
// READY and requires actual current CPU-worker + process numerical admission.
void prepare(const Config&);
void activate(); // after checked resident PP; before first seed T1
void deactivate(); // forbids new frames; does not free captured resources
void close(); // throwing; only after both per-stage unregister calls
bool prepared()noexcept;
bool active(const void*self)noexcept;
bool active_layer(int layer)noexcept;
void begin_stage(const void*self,int T);
Rows capture_pre(const void*self,int layer,int T);
Rows capture_post(const void*self,int layer,int T);
void capture_cpu_rows(const void*self,int layer,int T);
bool finish_stage(const void*self,std::string&error);
void abort_stage(const void*self);
// Accounting callback validates exact host R binding and current live stage;
// call before updating original counters, never waits for GPU completion.
void require_accounting(int layer,int T,int K,const int32_t*host_res,const int32_t*ids);
void fail_accounting(const char*reason)noexcept;
uint64_t planned_device_bytes(int device); // pure maxT8 buffer bound, excludes Verifier/driver
uint64_t device_bytes(int device);
}
