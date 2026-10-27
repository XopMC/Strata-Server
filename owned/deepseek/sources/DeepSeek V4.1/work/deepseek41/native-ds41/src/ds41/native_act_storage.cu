#include "strata/ds41/native_act_storage.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>
namespace strata::ds41 {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("native activation storage: ")+cudaGetErrorString(e));}
__global__ void pack(const __nv_bfloat16*x,size_t count,uint8_t*codes,uint8_t*scales){
    const size_t group=size_t(blockIdx.x)*8+(threadIdx.x>>5);const int lane=threadIdx.x&31;
    const size_t at=group*32+lane;if(at>=count)return;
    const float value=__bfloat162float(x[at]);float maximum=fabsf(value);
    for(int shift=16;shift>0;shift>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,shift));
    int exponent;const float mantissa=frexpf(fmaxf(maximum,1e-4f)*(1.f/448.f),&exponent);
    const int e=mantissa==.5f?exponent-1:exponent;const float scale=ldexpf(1.f,e);
    const __nv_fp8_e4m3 quantized(fminf(fmaxf(value/scale,-448.f),448.f));
    codes[at]=quantized.__x;if(!lane)scales[group]=uint8_t(e+127);
}
__global__ void gather(const uint8_t*codes,const uint8_t*scales,const int32_t*ids,int rows,int capacity,int dim,__nv_bfloat16*out){
    const size_t at=size_t(blockIdx.x)*256+threadIdx.x;if(at>=size_t(rows)*dim)return;
    const int row=ids[at/dim];assert(row>=0&&row<capacity);const size_t source=size_t(row)*dim+at%dim;
    __nv_fp8_e4m3 value;value.__x=codes[source];const uint8_t exponent=scales[source/32];assert(exponent!=255);
    out[at]=__float2bfloat16_rn(float(value)*ldexpf(1.f,int(exponent)-127));
}
}
void native_act_storage_pack(const uint16_t*x,int rows,int dim,uint8_t*codes,uint8_t*scales,cudaStream_t stream){
    const auto layout=native_act_storage_layout(rows,dim);if(!x||!codes||!scales)throw std::invalid_argument("native activation storage: missing buffer");
    pack<<<(layout.codes_bytes+255)/256,256,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(x),layout.codes_bytes,codes,scales);check(cudaGetLastError());
}
void native_act_storage_gather(const uint8_t*codes,const uint8_t*scales,const int32_t*ids,int rows,int capacity,int dim,uint16_t*out,cudaStream_t stream){
    native_act_storage_layout(capacity,dim);
    if(!codes||!scales||!ids||!out||rows<1||rows>16384)throw std::invalid_argument("native activation storage: invalid bounded gather");
    gather<<<(size_t(rows)*dim+255)/256,256,0,stream>>>(codes,scales,ids,rows,capacity,dim,reinterpret_cast<__nv_bfloat16*>(out));check(cudaGetLastError());
}
}
