#pragma once
#include "fastpath_counters_v534.hpp"
#include "strata/ds41/native_upload_pipeline.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
namespace strata::ds41::fast534 {
class UploadLease {
 uint8_t*host_[2]{};uint8_t*device_[2]{};
 cudaEvent_t copied_[2]{},used_[2]{};cudaStream_t stream_{};
 bool submitted_[2]{},closed_=false,registered_=false;
 size_t bytes_;int device_id_;uint64_t epoch_,generation_,format_;
 std::thread::id issuer_=std::this_thread::get_id();
 static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
 void current()const{int d=-1;check(cudaGetDevice(&d));if(closed_||std::this_thread::get_id()!=issuer_||d!=device_id_)throw std::logic_error("534 lease issuer/device/closed mismatch");}
 void slot(int s)const{if(s<0||s>1)throw std::out_of_range("534 two-slot lease");}
 void drain(){if(stream_)check(cudaStreamSynchronize(stream_));check(cudaStreamSynchronize(0));}
 void release()noexcept{
  // Never free events or DMA sources after an unproved drain.
  try{current();drain();}catch(...){add(21);std::terminate();}
  for(int i=0;i<2;i++){
   if(copied_[i]&&cudaEventDestroy(copied_[i])!=cudaSuccess)std::terminate();
   if(used_[i]&&cudaEventDestroy(used_[i])!=cudaSuccess)std::terminate();
   if(device_[i]&&cudaFree(device_[i])!=cudaSuccess)std::terminate();
   if(host_[i]&&cudaFreeHost(host_[i])!=cudaSuccess)std::terminate();
  }
  if(stream_&&cudaStreamDestroy(stream_)!=cudaSuccess)std::terminate();
  closed_=true;
  if(registered_){add(19);add(20);words[18].fetch_sub(1);registered_=false;}
 }
public:
 UploadLease(size_t bytes,int device,uint64_t epoch,uint64_t generation,uint64_t format)
  :bytes_(bytes),device_id_(device),epoch_(epoch),generation_(generation),format_(format){
  if(!bytes||device!=0||!epoch||!generation||!format)throw std::invalid_argument("534 typed headful lease required");
  current();
  try{check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
   for(int i=0;i<2;i++){check(cudaHostAlloc(reinterpret_cast<void**>(&host_[i]),bytes,cudaHostAllocDefault));check(cudaMalloc(reinterpret_cast<void**>(&device_[i]),bytes));check(cudaEventCreateWithFlags(&copied_[i],cudaEventDisableTiming));check(cudaEventCreateWithFlags(&used_[i],cudaEventDisableTiming));}
  }catch(...){release();throw;}
  registered_=true;add(9);add(18);set(22,generation);set(23,format);set(24,device);
 }
 ~UploadLease(){if(!closed_)release();}
 UploadLease(const UploadLease&)=delete;UploadLease&operator=(const UploadLease&)=delete;
 void validate(size_t bytes,int device,uint64_t epoch,uint64_t generation,uint64_t format){current();if(bytes!=bytes_||device!=device_id_||epoch!=epoch_||generation!=generation_||format!=format_)throw std::logic_error("534 lease binding changed");const auto old=words[11].fetch_add(1);if(old)add(10);}
 uint8_t*begin(int s){current();slot(s);if(submitted_[s])check(cudaEventSynchronize(copied_[s]));return host_[s];}
 uint8_t*device(int s){current();slot(s);return device_[s];}
 void upload(int s,size_t bytes){current();slot(s);if(!bytes||bytes>bytes_)throw std::invalid_argument("534 upload span");if(submitted_[s])check(cudaStreamWaitEvent(stream_,used_[s],0));check(cudaMemcpyAsync(device_[s],host_[s],bytes,cudaMemcpyHostToDevice,stream_));check(cudaEventRecord(copied_[s],stream_));check(cudaStreamWaitEvent(0,copied_[s],0));submitted_[s]=true;add(12);add(13,bytes);}
 void used(int s){current();slot(s);if(!submitted_[s])throw std::logic_error("534 use without upload");check(cudaEventRecord(used_[s],0));}
 void close(){if(closed_)return;current();release();}
};
// One protocol, separate old/fast owners. Old modes retain the original class.
class PipelineView {
 std::unique_ptr<NativeUploadPipeline> old_;
 UploadLease*fast_=nullptr;
public:
 PipelineView(UploadLease*fast,bool old_enabled,size_t bytes):fast_(fast){if(!fast_&&old_enabled)old_=std::make_unique<NativeUploadPipeline>(bytes);}
 explicit operator bool()const noexcept{return fast_||bool(old_);}
 PipelineView*operator->()noexcept{return this;}
 uint8_t*begin(int s){return fast_?fast_->begin(s):old_->begin(s);}
 uint8_t*device(int s){return fast_?fast_->device(s):old_->device(s);}
 void upload(int s,size_t b){if(fast_)fast_->upload(s,b);else old_->upload(s,b);}
 void used(int s){if(fast_)fast_->used(s);else old_->used(s);}
};
}
