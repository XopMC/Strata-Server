#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::ds41 {
struct NativeDeviceProjection {const uint8_t* weights;const uint8_t* scales;int bits;};
struct NativePrefillLayout {size_t weight,x,gate,up,hidden,tmp,bytes;};
void native_prefill_gather(const uint16_t*source,const int32_t*ids,int rows,int source_rows,int dim,uint16_t*out);
void native_prefill_scatter(const uint16_t*source,const int32_t*ids,int rows,int output_rows,int dim,float*out);
NativePrefillLayout native_prefill_layout(int rows,int dim,int inter);
size_t native_prefill_workspace_bytes(int rows,int dim,int inter);
// One expert's complete routed FFN for already gathered, unique token rows.
// BF16 x/output, FP32 routing weight applied before hidden BF16/FP8 rounding.
// Default CUDA stream only, matching the fork's batched-prefill operations.
void native_expert_prefill(const uint16_t*x,const float*routing,int rows,int dim,int inter,float limit,
                          NativeDeviceProjection gate,NativeDeviceProjection up,NativeDeviceProjection down,
                          uint16_t*out,void*workspace,size_t workspace_bytes);
// Exactly three immutable BF16 matrices, decoded once for one selected expert.
void native_expert_decode_weights(int dim,int inter,NativeDeviceProjection gate,NativeDeviceProjection up,
                                 NativeDeviceProjection down,uint16_t*out,size_t elements);
void native_expert_prefill_decoded(const uint16_t*x,const float*routing,int rows,int dim,int inter,float limit,
                                 const uint16_t*decoded,size_t elements,uint16_t*out,void*workspace,size_t bytes);
}
