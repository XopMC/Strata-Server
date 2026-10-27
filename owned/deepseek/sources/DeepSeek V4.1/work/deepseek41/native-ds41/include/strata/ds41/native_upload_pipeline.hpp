#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
namespace strata::ds41 {
// Two bounded pinned slots; upload n+1 can overlap default-stream compute n.
class NativeUploadPipeline {
    static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
    uint8_t*host_[2]{};uint8_t*device_[2]{};
    cudaEvent_t copied_[2]{},used_[2]{};cudaStream_t stream_{};
    bool submitted_[2]{};size_t bytes_;
    void release()noexcept{
        if(stream_)cudaStreamSynchronize(stream_);
        cudaStreamSynchronize(0);
        for(int i=0;i<2;i++){if(copied_[i])cudaEventDestroy(copied_[i]);if(used_[i])cudaEventDestroy(used_[i]);if(device_[i])cudaFree(device_[i]);if(host_[i])cudaFreeHost(host_[i]);}
        if(stream_)cudaStreamDestroy(stream_);
    }
public:
    explicit NativeUploadPipeline(size_t bytes):bytes_(bytes){
        try{
            check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
            for(int i=0;i<2;i++){check(cudaHostAlloc(reinterpret_cast<void**>(&host_[i]),bytes,cudaHostAllocDefault));check(cudaMalloc(reinterpret_cast<void**>(&device_[i]),bytes));check(cudaEventCreateWithFlags(&copied_[i],cudaEventDisableTiming));check(cudaEventCreateWithFlags(&used_[i],cudaEventDisableTiming));}
        }catch(...){release();throw;}
    }
    ~NativeUploadPipeline(){release();}
    NativeUploadPipeline(const NativeUploadPipeline&)=delete;
    NativeUploadPipeline&operator=(const NativeUploadPipeline&)=delete;
    uint8_t*begin(int slot){if(submitted_[slot])check(cudaEventSynchronize(copied_[slot]));return host_[slot];}
    uint8_t*device(int slot){return device_[slot];}
    void upload(int slot,size_t bytes){
        if(bytes>bytes_)throw std::runtime_error("native upload staging overflow");
        if(submitted_[slot])check(cudaStreamWaitEvent(stream_,used_[slot],0));
        check(cudaMemcpyAsync(device_[slot],host_[slot],bytes,cudaMemcpyHostToDevice,stream_));
        check(cudaEventRecord(copied_[slot],stream_));check(cudaStreamWaitEvent(0,copied_[slot],0));
        submitted_[slot]=true;
    }
    void used(int slot){check(cudaEventRecord(used_[slot],0));}
};
}
