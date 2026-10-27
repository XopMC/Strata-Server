#pragma once
#include "../cpu-resident-grouped-compact-v8/compact_v8.hpp"
#include "../two-dense-helper-metadata-locality-v74/partial_plan_v26.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace four_dispatch_v10 {
constexpr int Devices=4,Capacity=80,H=2560,F=640,Qrow=2880;
enum class CpuPolicy { Base, Compact };
struct NativeMask {int32_t rows[Capacity];int32_t count;uint32_t error;uint64_t sequence;};
struct BankStatus {uint32_t error=0,native_rows=0;cpu_rows_v5::ResultStamp stamp{};};
struct BankBuffers {
 cpu_grouped_v6::Images cpu;
 cpu_compact_v8::WorkList*compact=nullptr;
 uint8_t*native_q8=nullptr; // [8,2880] original native Q8 opaque bytes
 swift_ep_v26::Plan*native_plan=nullptr;
 void*native_scratch=nullptr;size_t native_scratch_bytes=0;
 float*native_rows=nullptr; // [80,H], only original native rows written
 BankStatus*status=nullptr;
 int persistent_ctas=0;
 cudaStream_t capture_stream=nullptr;
};
struct LayerReceipt {uint64_t sequence=0;uint32_t error=0,T=0,layer=0,done=0;};
struct OwnerView {
 int device=-1,layer=-1,T=0;
 cudaStream_t stream=nullptr;
 const cpu_rows_v5::Inputs*inputs=nullptr; // immutable per drained frame
 const float*original_x=nullptr;
 const void*original_native_q8=nullptr;
 float*parts=nullptr;float*native_hits=nullptr;
 NativeMask*mask=nullptr;
 LayerReceipt*receipt=nullptr; // unique per captured layer, retired with parent
 uint32_t*stage_error=nullptr; // caller resets ONCE per whole stage frame
};
struct Rows {const int32_t*dst;const int32_t*count;const uint32_t*error;};
// Source bodies borrow every buffer. This token must own/retain physical banks,
// uploaded views, scratch and original Verifier resources through parent retirement.
class Dispatcher {
 struct State;std::shared_ptr<State>p_;
public:
 Dispatcher(OwnerView,const std::array<BankBuffers,4>&,std::shared_ptr<const void>lease,CpuPolicy=CpuPolicy::Base);
 ~Dispatcher();Dispatcher(const Dispatcher&)=delete;Dispatcher&operator=(const Dispatcher&)=delete;
 // Require active parent capture on owner0/1. Store retain() alongside the
 // parent executable until that executable has been destroyed and drained.
 std::shared_ptr<const void>retain()const;
 Rows capture_pre(); // after original FP32/nativeQ8/router/logicalR producer
 Rows capture_post(); // full four-bank fork/join then native raw-row gather
 void capture_cpu_rows(); // AFTER old host wait/mapped copy, BEFORE combine
 // Positive parent-stream completion, then actual accumulated stage error read.
 void begin(uint64_t expected_sequence);
 bool finish(std::string&error);
 static bool finish_all(const std::vector<Dispatcher*>&,std::string&error);
 void abort();
 // Must follow destroyed parent graphs and release of their retain() tokens.
 void close();
};
cudaError_t reset_stage_error(uint32_t*,cudaStream_t);
}
