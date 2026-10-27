#pragma once
#include "../cpu-resident-row-adapter-v5-r2/row_adapter_v5.cuh"
#include "../cpu-semantics-cuda-leaf-v86/cpu_semantics_v86.hpp"
namespace cpu_grouped_v6 {
constexpr int Capacity=cpu_rows_v5::Capacity,MaxT=8,H=2560,F=640;
struct Status {
 uint32_t error=0,input_tokens=0,jobs=0,output_rows=0;
 cpu_rows_v5::ResultStamp stamp{};
};
// Every span is caller-owned and lives through the final owner consumer.
// output/raw DN [80,H], other float intermediates [80,F], hiddenQ [80,680],
// inputQ [8,2720], input [8,H]. No expanded/requantized/cached weight storage.
struct Images {
 const cpu_rows_v5::Plan*plan;
 const uint32_t*plan_error;
 const four_banks_v3::PhysicalDeviceView*physical;
 const float*input;
 uint8_t*input_q;
 float *gate_raw,*up_raw,*gate_scaled,*up_scaled,*hidden;
 uint8_t*hidden_q;
 float *down_raw,*output;
 Status*status;
};
// Five numerical kernels + status clear + one explicit publication kernel.
// Constant per-bank launch count; jobs are device-predicated, no CPU readback.
// root must qualify actual CPU workers/libm/archive before model admission.
cudaError_t chain(const Images&,int bank,cudaStream_t);
}
