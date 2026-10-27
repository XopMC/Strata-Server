// Private paired139 address reader; source_bf16_value and row arithmetic are
// literal selected244 bodies. No native Engine/class/source modification.
#include "pair_fp8_v531.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cstdio>
namespace {
thread_local char error531[256]={};
constexpr size_t MatrixBytes=6635520;
bool optin(){static const bool v=[](){const char*s=std::getenv("DS1017_PREFILL_DIRECT_FP8_V531");return s&&std::strcmp(s,"1")==0;}();return v;}
bool interval(const void*p,size_t n,uintptr_t&lo,uintptr_t&hi){lo=reinterpret_cast<uintptr_t>(p);if(!p||!n||n>std::numeric_limits<uintptr_t>::max()-lo)return false;hi=lo+n;return true;}
bool device_range(const void*p,size_t n,int expected){
 cudaPointerAttributes a{};if(cudaPointerGetAttributes(&a,p)!=cudaSuccess||a.type!=cudaMemoryTypeDevice||a.device!=expected)return false;
 CUdeviceptr base=0;size_t capacity=0;if(cuMemGetAddressRange(&base,&capacity,CUdeviceptr(reinterpret_cast<uintptr_t>(p)))!=CUDA_SUCCESS)return false;
 const uintptr_t at=reinterpret_cast<uintptr_t>(p);return at>=uintptr_t(base)&&size_t(at-uintptr_t(base))<=capacity&&n<=capacity-size_t(at-uintptr_t(base));
}
// Derivation: old nibble index ((row%16)%2)*4+col%4 is interleaved
// with its sibling tile into pair-byte lane (row%16)*4+col%4.
template<int Bits,int Width>
__device__ __forceinline__ unsigned compact_code(const uint8_t*blocked,int row,int col){
 static_assert(Bits==0,"paired139 contains exact original E2M1 codes only");
 const uint8_t*block=blocked+(size_t(row/32)*(Width/32)+col/32)*576;
 const unsigned packed=block[((col%32)/4)*64+(row%16)*4+col%4];
 return (packed>>(4*((row/16)&1)))&15u;
}
template<int Bits,int Width>
__device__ __forceinline__ uint16_t compact_scale(const uint8_t*blocked,int row,int col){
 static_assert(Bits==0,"paired139 contains original zeroextended E8 footer only");
 const uint8_t*scale=blocked+(size_t(row/32)*(Width/32)+col/32)*576+512+2*(row%32);
 return uint16_t(scale[0])|(uint16_t(scale[1])<<8);
}
template<int Bits, int Width, bool Compact>
__device__ __forceinline__ float source_bf16_value(const uint8_t* w, const uint8_t* s, int row, int col) {
    float value;
    if constexpr (Bits == 0) {
        const int code = Compact ? int(compact_code<Bits, Width>(w, row, col))
            : (w[size_t(row) * (Width / 2) + col / 2] >> (4 * (col % 2))) & 15;
        const int magnitude = code & 7;
        const uint32_t raw = magnitude < 2 ? (magnitude ? 0x3f000000u : 0u)
            : (uint32_t((magnitude >> 1) + 126) << 23) | (uint32_t(magnitude & 1) << 22);
        const float decoded = __uint_as_float(raw | ((code & 8) ? 0x80000000u : 0u));
        const uint8_t scale = Compact ? uint8_t(compact_scale<Bits, Width>(w, row, col))
            : s[size_t(row) * (Width / 32) + col / 32];
        const float factor = scale == 255 ? nanf("")
            : __uint_as_float(scale ? uint32_t(scale) << 23 : 0x00400000u);
        value = decoded * factor;
    } else {
        int code; float scale;
        if constexpr (Compact) {
            code = int(compact_code<Bits, Width>(w, row, col));
            scale = __uint_as_float(uint32_t(compact_scale<Bits, Width>(w, row, col)) << 16);
        } else {
            constexpr size_t stride = size_t(Width) * Bits / 8 + size_t(Width / 32) * 2;
            const uint8_t* base = w + size_t(row) * stride;
            const int at = col * Bits, byte = at / 8;
            code = ((int(base[byte]) | (int(base[byte + 1]) << 8)) >> (at % 8)) & ((1 << Bits) - 1);
            const uint8_t* ss = base + size_t(Width) * Bits / 8 + (col / 32) * 2;
            scale = __uint_as_float(uint32_t(uint16_t(ss[0]) | (uint16_t(ss[1]) << 8)) << 16);
        }
        value = (code == (1 << Bits) - 1 || !isfinite(scale) || scale < 0) ? nanf("")
            : float(code - ((1 << (Bits - 1)) - 1)) * scale;
    }
    return __bfloat162float(__float2bfloat16_rn(value));
}

template<int Bits, int Width, bool Compact>
__global__ void encode_rows531(const uint8_t* w, const uint8_t* s, uint8_t* q, float* scales) {
    constexpr int Iterations = (Width + 255) / 256;
    const int row = blockIdx.x;
    float values[Iterations], maximum = 1e-4f;
    __shared__ float shared[256];
#pragma unroll
    for (int it = 0; it < Iterations; ++it) {
        const int c = threadIdx.x + it * 256;
        values[it] = c < Width ? source_bf16_value<Bits, Width, Compact>(w, s, row, c) : 0.f;
        if (c < Width) maximum = fmaxf(maximum, fabsf(values[it]));
    }
    shared[threadIdx.x] = maximum; __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (threadIdx.x < d) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + d]);
        __syncthreads();
    }
    const float raw = shared[0] * (1.f / 448.f);
    const uint32_t bits = __float_as_uint(raw);
    const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
    const float scale = ldexpf(1.f, exponent);
    if (threadIdx.x == 0) scales[row] = scale;
#pragma unroll
    for (int it = 0; it < Iterations; ++it) {
        const int c = threadIdx.x + it * 256;
        if (c < Width) {
            const __nv_fp8_e4m3 value(values[it] / scale);
            q[size_t(row) * Width + c] = value.__x;
        }
    }
}

int refuse(int code,const char*message){std::snprintf(error531,sizeof(error531),"%s",message);return code;}
}
extern "C" const char*ds531_pair32_fp8_error(){return error531;}
extern "C" int ds531_pair32_fp8_encode(const Ds531Projection*projections,size_t count,int expected,uint64_t stream){
 error531[0]=0;if(!optin())return refuse(DS531_POLICY,"explicit direct531 startup optin required");
 if(!projections||(count!=1&&count!=3)||(expected!=0&&expected!=1))return refuse(DS531_ARGUMENT,"one projection or gate/up/down triple on explicit logical0/1 required");
 // Copy caller POD before validation; it is never retained by queued kernels.
 std::array<Ds531Projection,3>p{};std::array<std::array<uintptr_t,2>,9>ranges{};
 for(size_t i=0;i<count;i++){
  p[i]=projections[i];const auto&v=p[i];
  if(v.struct_bytes!=sizeof(v)||v.layout_tag!=139||v.source_bits||v.source_bytes!=MatrixBytes||
     !((v.rows==2304&&v.columns==5120)||(v.rows==5120&&v.columns==2304))||
     (count==3&&(v.rows!=(i==2?5120:2304)||v.columns!=(i==2?2304:5120)))||
     v.fp8_bytes!=uint64_t(v.rows)*uint64_t(v.columns)||v.row_scale_bytes!=uint64_t(v.rows)*sizeof(float)||(reinterpret_cast<uintptr_t>(v.row_scales)%alignof(float)))return refuse(DS531_ARGUMENT,"typed139 exact dimensions/byte lengths differ");
  const void*ptrs[3]={v.source,v.fp8,v.row_scales};const size_t bytes[3]={size_t(v.source_bytes),size_t(v.fp8_bytes),size_t(v.row_scale_bytes)};
  for(int j=0;j<3;j++)if(!interval(ptrs[j],bytes[j],ranges[i*3+j][0],ranges[i*3+j][1]))return refuse(DS531_ARGUMENT,"null/overflow/empty pointer range");
 }
 for(size_t i=0;i<count*3;i++)for(size_t j=i+1;j<count*3;j++)if(ranges[i][0]<ranges[j][1]&&ranges[j][0]<ranges[i][1])return refuse(DS531_ALIAS,"source/output ranges must all be disjoint");
 auto st=reinterpret_cast<cudaStream_t>(stream);cudaStreamCaptureStatus capture{};
 if(cudaStreamIsCapturing(st,&capture)!=cudaSuccess)return refuse(DS531_CUDA_GUARD,"stream capture query failed");
 if(capture!=cudaStreamCaptureStatusNone)return refuse(DS531_CAPTURE,"direct531 PP encoder is not a capture-time entry");
 int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=expected)return refuse(DS531_DEVICE,"caller must retain explicit logical0/1 current");
 if(st){int d=-1;if(cudaStreamGetDevice(st,&d)!=cudaSuccess||d!=current)return refuse(DS531_DEVICE,"stream belongs to a different GPU");}
 for(size_t i=0;i<count;i++)if(!device_range(p[i].source,p[i].source_bytes,current)||!device_range(p[i].fp8,p[i].fp8_bytes,current)||!device_range(p[i].row_scales,p[i].row_scale_bytes,current))return refuse(DS531_POINTER,"actual cudaMalloc bounds/device required");
 // No partial launch on metadata/alias/device/capture/pointer validation failure.
 // A CUDA enqueue failure is returned directly, never hidden by last-error reset.
 for(size_t i=0;i<count;i++){
  const auto*w=static_cast<const uint8_t*>(p[i].source);const uint8_t*s=nullptr;
  auto*q=static_cast<uint8_t*>(p[i].fp8);auto*scales=static_cast<float*>(p[i].row_scales);
  void*args[]={&w,&s,&q,&scales};
  const void*fn=p[i].columns==5120?reinterpret_cast<const void*>(encode_rows531<0,5120,true>):reinterpret_cast<const void*>(encode_rows531<0,2304,true>);
  const auto code=cudaLaunchKernel(fn,dim3(p[i].rows),dim3(256),args,0,st);if(code!=cudaSuccess)return refuse(int(code),"direct531 kernel enqueue failed");
 }
 return 0;
}
