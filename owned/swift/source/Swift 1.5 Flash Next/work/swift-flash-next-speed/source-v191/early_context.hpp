#pragma once
#include "../source-v190-r3/partial_plan.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <cstdio>
#include <string>
#include <memory>
#include <mutex>
namespace swift191 {
cudaError_t prepare_check();
cudaError_t check_plan(const int32_t*,const int32_t*,int,int,const uint64_t*,uint32_t,uint32_t*,cudaStream_t);
inline std::atomic<int> pcie_num{0};
struct Context {
 uint64_t* host{};uint64_t* device{};uint64_t* offsets{};uint32_t* status{};
 bool fast=true,check=false;uint32_t mask=0;uint64_t windows=0,layers=0,checked=0;
 ~Context(){if(status)cudaFree(status);if(offsets)cudaFree(offsets);if(host)cudaFreeHost(host);}
 void reset(){for(int g=0;g<2;++g)std::atomic_ref<uint64_t>(host[g]).store(0,std::memory_order_seq_cst);}
 void release(){for(int g=0;g<2;++g)std::atomic_ref<uint64_t>(host[g]).store(UINT64_MAX&~1ull,std::memory_order_seq_cst);}
};
inline std::mutex mutex;
inline std::map<const void*,std::unique_ptr<Context>> contexts;
inline Context* get(const void* key){std::lock_guard lock(mutex);auto it=contexts.find(key);return it==contexts.end()?nullptr:it->second.get();}
inline void erase(const void* key){std::lock_guard lock(mutex);auto it=contexts.find(key);if(it!=contexts.end()){auto* c=it->second.get();std::fprintf(stderr,"Swift EARLY191: windows=%llu layers=%llu checked=%llu ready128B ABI_unchanged=1\n",(unsigned long long)c->windows,(unsigned long long)c->layers,(unsigned long long)c->checked);contexts.erase(it);}}
inline bool create(const void* key,const uint64_t* offsets,int64_t slots,std::string& err){
 const char* flag=std::getenv("STRATA_SWIFT_EARLY_HITS191");if(!flag||std::strcmp(flag,"1"))return true;
 auto c=std::make_unique<Context>();const char* check=std::getenv("STRATA_SWIFT_EARLY_HITS191_CHECK");c->check=check&&std::strcmp(check,"1")==0;
 cudaError_t e=swift190::prepare();if(e==cudaSuccess)e=prepare_check();
 if(e==cudaSuccess)e=cudaHostAlloc(&c->host,128,cudaHostAllocMapped);
 if(e==cudaSuccess)e=cudaHostGetDevicePointer(&c->device,c->host,0);
 if(e==cudaSuccess)e=cudaMalloc(&c->status,2*sizeof(uint32_t));
 if(e==cudaSuccess)e=cudaMemset(c->status,0,2*sizeof(uint32_t));
 if(e==cudaSuccess&&offsets){e=cudaMalloc(&c->offsets,slots*sizeof(uint64_t));if(e==cudaSuccess)e=cudaMemcpy(c->offsets,offsets,slots*sizeof(uint64_t),cudaMemcpyHostToDevice);}
 if(e!=cudaSuccess){err=std::string("EARLY191 initialization: ")+cudaGetErrorString(e);return false;}
 c->fast=pcie_num.load()==0;c->reset();std::lock_guard lock(mutex);if(contexts.contains(key)){err="EARLY191 duplicate context owner";return false;}contexts.emplace(key,std::move(c));return true;
}
}
extern "C" void strata_swift_early191_set_pcie(int n){swift191::pcie_num.store(n,std::memory_order_release);}
