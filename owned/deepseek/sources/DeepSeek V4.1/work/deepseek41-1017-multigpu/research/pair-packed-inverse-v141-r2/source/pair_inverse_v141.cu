#include "../../../../../../../include/strata_ds/fresh_diagnostic_visibility.hpp"
// Pure byte permutation. Original GPU0 FP8 PP and GPU1 GU13 math untouched.
#include "pair_inverse_v141.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <limits>
namespace {
constexpr size_t Bytes=6635520;
bool interval(const void*p,size_t n,uintptr_t&lo,uintptr_t&hi){
 lo=reinterpret_cast<uintptr_t>(p);if(!p||!n||n>std::numeric_limits<uintptr_t>::max()-lo)return false;hi=lo+n;return true;
}
bool device_range(const void*p,size_t n,int expected){
 cudaPointerAttributes a{};
 if(cudaPointerGetAttributes(&a,p)!=cudaSuccess||a.type!=cudaMemoryTypeDevice||a.device!=expected)return false;
 CUdeviceptr base=0;size_t capacity=0;
 if(cuMemGetAddressRange(&base,&capacity,CUdeviceptr(reinterpret_cast<uintptr_t>(p)))!=CUDA_SUCCESS)return false;
 const uintptr_t at=reinterpret_cast<uintptr_t>(p);
 return at>=uintptr_t(base)&&size_t(at-uintptr_t(base))<=capacity&&n<=capacity-size_t(at-uintptr_t(base));
}
template<int Groups>
__global__ void inverse141(const uint8_t*source,uint8_t*destination){
 for(size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;at<Bytes;at+=size_t(blockDim.x)*gridDim.x){
  const int tile=int(at/(size_t(Groups)*288));
  const int group=int(at/288%Groups),byte=int(at%288),half=tile&1;
  const auto*pair=source+(size_t(tile/2)*Groups+group)*576;
  uint8_t value;
  if(byte>=256)value=pair[512+half*32+byte-256];
  else{
   const int segment=byte/32,lane=2*(byte%32),shift=4*half;
   const uint8_t low=uint8_t((pair[segment*64+lane]>>shift)&15);
   const uint8_t high=uint8_t((pair[segment*64+lane+1]>>shift)&15);
   value=uint8_t(low|(high<<4));
  }
  destination[at]=value;
 }
}
}
extern "C" int ds141_pair32_to_output16(const void*source,size_t bytes,int rows,int columns,
 uint32_t tag,void*destination,size_t capacity,int expected,uint64_t stream){
 const char*flag=std::getenv("DS1017_PAIR32_BYTE_INVERSE_V141");
 if(!flag||std::strcmp(flag,"1"))return DS141_POLICY;
 if((expected!=0&&expected!=1)||tag!=139||bytes!=Bytes||capacity!=Bytes||
    !((rows==2304&&columns==5120)||(rows==5120&&columns==2304)))return DS141_ARGUMENT;
 uintptr_t sl,sh,dl,dh;
 if(!interval(source,bytes,sl,sh)||!interval(destination,capacity,dl,dh))return DS141_ARGUMENT;
 if(sl<dh&&dl<sh)return DS141_OVERLAP;
 auto st=reinterpret_cast<cudaStream_t>(stream);cudaStreamCaptureStatus capture;
 if(cudaStreamIsCapturing(st,&capture)!=cudaSuccess)return DS141_CUDA_GUARD;
 if(capture!=cudaStreamCaptureStatusNone)return DS141_CAPTURE;
 int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=expected)return DS141_DEVICE;
 strata::ds41::publication::DiagnosticSnapshot fresh{};if(!strata::ds41::publication::diagnostic_snapshot(2,fresh))return DS141_DEVICE;
 CUuuid uuid{};if(cuDeviceGetUuid(&uuid,current)!=CUDA_SUCCESS||std::memcmp(uuid.bytes,fresh[expected].data(),16))return DS141_DEVICE;
 if(st){int stream_device=-1;if(cudaStreamGetDevice(st,&stream_device)!=cudaSuccess||stream_device!=current)return DS141_DEVICE;}
 if(!device_range(source,bytes,current)||!device_range(destination,capacity,current))return DS141_POINTER;
 // cudaLaunchKernel reports this enqueue directly; do not clear or hide a
 // caller's prior asynchronous error through cudaGetLastError here.
 const auto*src=static_cast<const uint8_t*>(source);auto*dst=static_cast<uint8_t*>(destination);
 void*args[]={&src,&dst};
 const void*fn=columns==5120?reinterpret_cast<const void*>(inverse141<160>):reinterpret_cast<const void*>(inverse141<72>);
 return int(cudaLaunchKernel(fn,dim3(1024),dim3(256),args,0,st));
}
