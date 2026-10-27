#pragma once
#include "strata/ds41/native_prefill.hpp"
struct CUstream_st;
namespace strata::ds41 {
struct NativeDeviceExpert {NativeDeviceProjection projection[3];};
void native_expert_cached_many(const uint16_t*x,const NativeDeviceExpert*experts,const float*alpha,int count,float limit,uint16_t*out,void*workspace,size_t bytes,CUstream_st* stream=0);
void native_expert_cached_gemv(const uint16_t*x,const float*alpha,float limit,const NativeDeviceProjection&gate,const NativeDeviceProjection&up,const NativeDeviceProjection&down,uint16_t*out,void*workspace,size_t bytes,CUstream_st* stream=0);
}
