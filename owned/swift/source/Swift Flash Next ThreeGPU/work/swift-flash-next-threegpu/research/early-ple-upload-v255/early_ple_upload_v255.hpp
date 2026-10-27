#pragma once
// Private per-run host transport. No new representation, numerical leaf or public layout.
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
namespace strata::prefill::early_ple_v255 {
inline bool requested() {
    static const bool on=[] {const char*p=std::getenv("STRATA_PREFILL_EARLY_PLE_FP32_V255");return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool disjoint(const void*a,uint64_t na,const void*b,uint64_t nb) {
    const auto x=(uintptr_t)a,y=(uintptr_t)b;
    return a&&b&&na&&nb&&na<=UINTPTR_MAX-x&&nb<=UINTPTR_MAX-y&&(x+na<=y||y+nb<=x);
}
// The object and captured destination are scoped to ONE run_impl/borrowed layout.
// No caller may relayout/refill/free that layout until its checked drain returns.
class Owner {
    cudaStream_t stream_=nullptr,compute_=nullptr,experts_=nullptr;
    cudaEvent_t consumed_event_=nullptr;
    const void* destination_=nullptr; uint64_t capacity_=0;
    std::mutex mutex_;std::condition_variable cv_;
    uint64_t consumed_=0;
    std::atomic<uint64_t> ready_{0};
    std::atomic<bool> failed_{false},cancelled_{false};
    bool active_=false,drained_=false;
    bool error(std::string& e,const char* what,cudaError_t rc) {
        {std::lock_guard<std::mutex> lock(mutex_);failed_.store(true,std::memory_order_release);}cv_.notify_all();
        e=std::string("V255 ")+what+": "+cudaGetErrorString(rc);return false;
    }
public:
    Owner()=default;Owner(const Owner&)=delete;Owner&operator=(const Owner&)=delete;
    bool init(void* destination,uint64_t capacity,cudaStream_t compute,cudaStream_t experts,std::string& e) {
        if(active_||!destination||!capacity||!compute||!experts){e="V255 invalid owning layout/streams";return false;}
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        cudaError_t rc=cudaStreamIsCapturing(compute,&capture);
        if(rc!=cudaSuccess)return error(e,"capture query",rc);
        if(capture!=cudaStreamCaptureStatusNone){e="V255 Prefill capture unsupported";return false;}
        rc=cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking);
        if(rc!=cudaSuccess)return error(e,"copy stream",rc);
        rc=cudaEventCreateWithFlags(&consumed_event_,cudaEventDisableTiming);
        if(rc!=cudaSuccess)return error(e,"consumed event",rc);
        destination_=destination;capacity_=capacity;compute_=compute;experts_=experts;active_=true;
        return true;
    }
    void fail() noexcept {{std::lock_guard<std::mutex> lock(mutex_);failed_.store(true,std::memory_order_release);}cv_.notify_all();}
    void cancel() noexcept {{std::lock_guard<std::mutex> lock(mutex_);cancelled_.store(true,std::memory_order_release);}cv_.notify_all();}
    bool ready(uint64_t generation)const noexcept{return ready_.load(std::memory_order_acquire)>=generation;}
    // A chunk's issuer can always finish layer0. Only future-layer credits wait here.
    bool wait_ready(uint64_t generation,const std::atomic<bool>& stop) {
        std::unique_lock<std::mutex> lock(mutex_);
        while(!ready(generation)&&!failed_.load(std::memory_order_acquire)&&!cancelled_.load(std::memory_order_acquire)&&!stop.load(std::memory_order_acquire))
            cv_.wait_for(lock,std::chrono::milliseconds(5)); // stop is owned by the existing chunk IssuerJoin
        // A later prefetched chunk may fail while this already-ready chunk runs.
        // Never strand its existing ring consumers by retracting an earned credit.
        return ready(generation)&&!cancelled_.load(std::memory_order_acquire)&&!stop.load(std::memory_order_acquire);
    }
    bool upload(uint64_t generation,int buffer,const void* source,uint64_t bytes,cudaEvent_t copied,std::string& e) {
        if(!active_||generation==0||buffer<0||buffer>1||!source||!copied||!bytes||bytes>capacity_){fail();e="V255 upload extent/epoch";return false;}
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock,[&]{return consumed_>=generation-1||failed_.load(std::memory_order_acquire)||cancelled_.load(std::memory_order_acquire);});
            if(failed_.load(std::memory_order_acquire)||cancelled_.load(std::memory_order_acquire)){e="V255 upload cancelled/failed";return false;}
            if(ready_.load(std::memory_order_acquire)!=generation-1){failed_.store(true,std::memory_order_release);cv_.notify_all();e="V255 nonsequential upload";return false;}
        }
        // consumed(g-1) was recorded BEFORE its host publication. The next main
        // consumed(g) cannot be recorded until this future's upload completes.
        cudaError_t rc=cudaSuccess;
        if(generation>1)rc=cudaStreamWaitEvent(stream_,consumed_event_,0);
        if(rc==cudaSuccess)rc=cudaMemcpyAsync(const_cast<void*>(destination_),source,(size_t)bytes,cudaMemcpyHostToDevice,stream_);
        if(rc==cudaSuccess)rc=cudaEventRecord(copied,stream_);
        if(rc!=cudaSuccess)return error(e,"upload/record",rc);
        // Host source reuse AND future-layer credits require DEVICE completion.
        rc=cudaEventSynchronize(copied);
        if(rc!=cudaSuccess)return error(e,"device ready",rc);
        {std::lock_guard<std::mutex> lock(mutex_);ready_.store(generation,std::memory_order_release);}cv_.notify_all();
        std::fprintf(stderr,"V255 READY CUDA0 generation=%llu buffer=%d FP32_bytes=%llu device_complete=1\n",(unsigned long long)generation,buffer,(unsigned long long)bytes);
        return true;
    }
    bool land(uint64_t generation,cudaEvent_t copied,std::string& e) {
        if(!active_||!ready(generation)||failed_.load(std::memory_order_acquire)){e="V255 land before ready/failed";return false;}
        const auto rc=cudaStreamWaitEvent(compute_,copied,0);
        return rc==cudaSuccess||error(e,"compute ready wait",rc);
    }
    bool consumed(uint64_t generation,std::string& e) {
        if(!active_||!ready(generation)||failed_.load(std::memory_order_acquire)){e="V255 consumed before ready/failed";return false;}
        // Record after ALL PLE consumers, before publishing the single destination.
        const auto rc=cudaEventRecord(consumed_event_,compute_);
        if(rc!=cudaSuccess)return error(e,"consumed record",rc);
        {std::lock_guard<std::mutex> lock(mutex_);if(consumed_+1!=generation){failed_.store(true,std::memory_order_release);cv_.notify_all();e="V255 nonsequential consumed";return false;}consumed_=generation;}
        cv_.notify_all();return true;
    }
    // Called only AFTER cancellation and joining the one gather/upload future.
    // Chunk issuer has already stopped/joined, so no new expert DMA can appear.
    void drain_or_die() noexcept {
        if(!active_||drained_)return;
        if(cudaStreamSynchronize(stream_)!=cudaSuccess||cudaStreamSynchronize(compute_)!=cudaSuccess||cudaStreamSynchronize(experts_)!=cudaSuccess){
            std::fprintf(stderr,"V255 FATAL: cannot drain owning PLE/layout streams before loan release\n");std::_Exit(2);
        }
        drained_=true;
    }
    ~Owner() noexcept {
        cancel();drain_or_die();
        if(consumed_event_)cudaEventDestroy(consumed_event_);
        if(stream_)cudaStreamDestroy(stream_);
    }
};
} // namespace strata::prefill::early_ple_v255
