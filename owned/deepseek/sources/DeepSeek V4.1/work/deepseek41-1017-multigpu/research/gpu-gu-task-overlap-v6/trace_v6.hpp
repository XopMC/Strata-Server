#pragma once
#include <cstdint>
struct DsTraceV6 {
 uint32_t struct_size,version,rows,topk,dim,inter;
 const float*activation,*gate,*down;
 const uint16_t*hidden_bf16,*output;
 const int8_t*hidden_codes;const float*hidden_scales;const int32_t*hidden_sums;
 uint64_t flags[2],served;int32_t error,reserved;
};
extern "C" int ds41_moe_trace_v6(void*task,DsTraceV6*out);
