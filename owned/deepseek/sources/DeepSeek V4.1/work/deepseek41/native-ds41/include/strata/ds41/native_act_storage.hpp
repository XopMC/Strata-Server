#pragma once
#include <cstddef>
#include <cstdint>
struct CUstream_st;
namespace strata::ds41 {
struct NativeActStorageLayout {size_t codes_bytes,scales_bytes;};
NativeActStorageLayout native_act_storage_layout(int rows,int dim);
// The same group32 FP8/E8M0 activation quantizer already used by experts.
// This is not an extra quantization level. Gather restores its BF16 output.
void native_act_storage_pack(const uint16_t*x,int rows,int dim,uint8_t*codes,uint8_t*scales,CUstream_st* stream=0);
void native_act_storage_gather(const uint8_t*codes,const uint8_t*scales,const int32_t*ids,int rows,int capacity,int dim,uint16_t*out,CUstream_st* stream=0);
}
