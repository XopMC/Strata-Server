#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <stdexcept>
namespace qsa_vmm451 {
inline void cu(CUresult e){if(e!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(e,&s);throw std::runtime_error(s?s:"VMM driver failure");}}
inline void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
class Pair {
 std::array<CUdeviceptr,2>reserved_{};std::array<std::array<CUmemGenericAllocationHandle,2>,2>h_{};
 CUdeviceptr temporary_{};cudaStream_t migration_{};bool foreign_=false;
 void access(CUdeviceptr p,size_t n){std::array<CUmemAccessDesc,2>a{};for(int d=0;d<2;++d){a[d].location.type=CU_MEM_LOCATION_TYPE_DEVICE;a[d].location.id=d;a[d].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;}cu(cuMemSetAccess(p,n,a.data(),2));}
public:
 size_t gran=0,bytes=0,physical=0,lower=0;std::array<CUdeviceptr,2>va{};
 explicit Pair(size_t logical,size_t exact_boundary):bytes(logical){
  cu(cuInit(0));for(int d=0;d<2;++d){CUdevice dev;cu(cuDeviceGet(&dev,d));int support=0;cu(cuDeviceGetAttribute(&support,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));if(!support)throw std::runtime_error("VMM mandatory support absent");CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=d;size_t g=0;cu(cuMemGetAllocationGranularity(&g,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));gran=std::max(gran,g);}
  physical=(bytes+gran-1)/gran*gran;lower=exact_boundary/gran*gran;if(!lower||lower>=physical)throw std::runtime_error("VMM partition cannot be represented");
  for(int d=0;d<2;++d){ck(cudaSetDevice(d));cu(cuMemAddressReserve(&reserved_[d],physical+2*gran,gran,0,0));va[d]=reserved_[d]+gran;for(int k=0;k<2;++k){const size_t offset=k?lower:0,len=k?physical-lower:lower;CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=d;cu(cuMemCreate(&h_[d][k],len,&prop,0));cu(cuMemMap(va[d]+offset,len,0,h_[d][k],0));access(va[d]+offset,len);}}
  ck(cudaSetDevice(0));ck(cudaStreamCreateWithFlags(&migration_,cudaStreamNonBlocking));ck(cudaMalloc(reinterpret_cast<void**>(&temporary_),gran));
 }
 float* data(int d)const{return reinterpret_cast<float*>(va.at(d));}
 bool foreign()const{return foreign_;}
 void exchange(){
  // Caller must first positively drain every graph/consumer of both pointers.
  ck(cudaSetDevice(0));for(size_t off=0;off<lower;off+=gran){cu(cuMemcpyDtoDAsync(temporary_,va[0]+off,gran,migration_));cu(cuMemcpyDtoDAsync(va[0]+off,va[1]+off,gran,migration_));cu(cuMemcpyDtoDAsync(va[1]+off,temporary_,gran,migration_));}ck(cudaStreamSynchronize(migration_));
  cu(cuMemUnmap(va[0],lower));cu(cuMemUnmap(va[1],lower));std::swap(h_[0][0],h_[1][0]);for(int d=0;d<2;++d){cu(cuMemMap(va[d],lower,0,h_[d][0],0));access(va[d],lower);}foreign_=!foreign_;
 }
 ~Pair(){
  try{ck(cudaSetDevice(0));ck(cudaStreamSynchronize(migration_));for(int d=0;d<2;++d){cu(cuMemUnmap(va[d],lower));cu(cuMemUnmap(va[d]+lower,physical-lower));for(auto h:h_[d])cu(cuMemRelease(h));cu(cuMemAddressFree(reserved_[d],physical+2*gran));}ck(cudaFree(reinterpret_cast<void*>(temporary_)));ck(cudaStreamDestroy(migration_));}catch(...){std::abort();}
 }
};
}
