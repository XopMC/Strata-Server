#pragma once
// Native device boundaries only. Root's default P2P admission runs before any
// allocation/capture. This helper neither enables peers nor offers host staging.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace peer_transport_v17 {
constexpr int ModelDevices=3,DecodeRows=8,PrefillRows=32768;
struct Boundary {float* pointer=nullptr;size_t bytes=0;int producer=-1,consumer=-1;};
struct Registry {std::mutex mutex;std::vector<Boundary> entries;};
inline Registry& registry(){static Registry* value=new Registry;return *value;}
[[noreturn]]inline void fatal(const char* text,cudaError_t error=cudaErrorInvalidValue)noexcept{
    std::fprintf(stderr,"V17 native transport retirement: %s: %s\n",text,cudaGetErrorString(error));
    std::fflush(stderr);std::_Exit(2);
}
inline void check(cudaError_t error,const char* text){
    if(error!=cudaSuccess)throw std::runtime_error(std::string("V17 ")+text+": "+cudaGetErrorString(error));
}
struct Restore {
    int previous=-1;
    Restore(){check(cudaGetDevice(&previous),"save owning device");}
    ~Restore(){if(previous>=0&&cudaSetDevice(previous)!=cudaSuccess)fatal("restore owning device");}
};
// Declare BEFORE all GpuStage/Verifier owners. Destruction follows their graph,
// stream and borrower retirement; it uses context drains, never stale streams.
class BoundaryPool {
    std::vector<Boundary> owned_;
public:
    BoundaryPool()=default;
    BoundaryPool(const BoundaryPool&)=delete;BoundaryPool&operator=(const BoundaryPool&)=delete;
    float* allocate(int producer,int consumer,size_t bytes){
        if(producer<0||producer>=ModelDevices-1||consumer!=producer+1||!bytes)
            throw std::runtime_error("V17 expected directed current-model producer/successor boundary");
        Restore restore;check(cudaSetDevice(producer),"boundary allocation owner");
        int accessible=0;check(cudaDeviceCanAccessPeer(&accessible,consumer,producer),"successor native peer capability");
        if(!accessible)throw std::runtime_error("V17 native successor access absent; no host fallback");
        Boundary b;b.producer=producer;b.consumer=consumer;b.bytes=bytes;
        check(cudaMalloc(reinterpret_cast<void**>(&b.pointer),bytes),"small producer boundary allocation");
        try{
            check(cudaMemset(b.pointer,0,bytes),"initial boundary contents");
            owned_.push_back(b);
            auto& r=registry();std::lock_guard<std::mutex>lock(r.mutex);r.entries.push_back(b);
        }catch(...){
            const auto found=std::find_if(owned_.begin(),owned_.end(),[&](const Boundary&x){return x.pointer==b.pointer;});
            if(found!=owned_.end())owned_.erase(found);
            if(cudaDeviceSynchronize()!=cudaSuccess||cudaFree(b.pointer)!=cudaSuccess)fatal("failed boundary allocation retirement");
            throw;
        }
        return b.pointer;
    }
    ~BoundaryPool()noexcept{
        if(owned_.empty())return;
        int saved=-1;if(cudaGetDevice(&saved)!=cudaSuccess)fatal("save pool retirement device");
        // Drain EVERY producer/consumer before ANY boundary free. Consumers may
        // have issued the final reads even after their producer was idle.
        bool used[ModelDevices]={};for(const auto&b:owned_){used[b.producer]=true;used[b.consumer]=true;}
        for(int d=0;d<ModelDevices;++d)if(used[d]){
            if(cudaSetDevice(d)!=cudaSuccess)fatal("select borrower retirement device");
            const auto e=cudaDeviceSynchronize();if(e!=cudaSuccess)fatal("positive borrower context drain",e);
        }
        auto&r=registry();std::lock_guard<std::mutex>lock(r.mutex);
        for(auto&b:owned_){
            if(cudaSetDevice(b.producer)!=cudaSuccess)fatal("select boundary free owner");
            const auto e=cudaFree(b.pointer);if(e!=cudaSuccess)fatal("boundary free",e);
            r.entries.erase(std::remove_if(r.entries.begin(),r.entries.end(),[&](const Boundary&x){return x.pointer==b.pointer;}),r.entries.end());
        }
        if(cudaSetDevice(saved)!=cudaSuccess)fatal("restore pool retirement device");
    }
};
// Initialization only, before capture; validates exact pool-owned base/capacity.
inline bool admit(const float* pointer,int producer,int consumer,size_t bytes,std::string&error){
    auto&r=registry();std::lock_guard<std::mutex>lock(r.mutex);
    const auto it=std::find_if(r.entries.begin(),r.entries.end(),[&](const Boundary&b){
        return b.pointer==pointer&&b.producer==producer&&b.consumer==consumer&&b.bytes==bytes;
    });
    if(it==r.entries.end()){error="V17 unknown/foreign/nonexact native boundary lease";return false;}
    return true;
}
// Capturable UVA device-to-device copy. The stream belongs to the currently
// owning stage. Root's enabled native peer matrix is a mandatory prerequisite.
inline bool copy(void* destination,const void* source,size_t bytes,cudaStream_t stream,std::string&error){
    const auto e=cudaMemcpyAsync(destination,source,bytes,cudaMemcpyDeviceToDevice,stream);
    if(e!=cudaSuccess){error=std::string("V17 native device boundary copy: ")+cudaGetErrorString(e);return false;}
    return true;
}
// Failure only: cover separate-stream callback consumers before returning an
// error (for example MTP reads of the last R). This does not affect normal flow.
inline void retire_failed_model_work()noexcept{
    int saved=-1;if(cudaGetDevice(&saved)!=cudaSuccess)fatal("save failed-run owner");
    for(int device=0;device<ModelDevices;++device){
        if(cudaSetDevice(device)!=cudaSuccess)fatal("select failed-run borrower");
        const auto e=cudaDeviceSynchronize();if(e!=cudaSuccess)fatal("uncertain failed-run borrower completion",e);
    }
    if(cudaSetDevice(saved)!=cudaSuccess)fatal("restore failed-run owner");
}
inline bool flag(const char* name){const char*v=std::getenv(name);return v&&std::atoi(v)!=0;}
inline bool inactive(const char* name){const char*v=std::getenv(name);return !v||!*v||!std::strcmp(v,"0");}
} // namespace peer_transport_v17
