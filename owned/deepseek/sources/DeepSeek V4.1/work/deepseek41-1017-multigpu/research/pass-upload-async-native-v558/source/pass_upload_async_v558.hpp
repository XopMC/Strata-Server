#pragma once
#include "pass_upload_lease_v554.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
namespace strata::ds41::pass558 {
inline constexpr int InfoWords=28;
inline std::array<std::atomic<uint64_t>,InfoWords> stats{};
inline bool enabled(){
 const char*v=std::getenv("DS1017_PASS_UPLOAD_ASYNC_V558");
 // This derivative has no flag0 legacy-drain route. Both contracts are explicit.
 if(!v||std::strcmp(v,"1")||!pass554::enabled())
  throw std::invalid_argument("558 requires literal async558=1 and pass554=1");
 return true;
}
inline void add(int i,uint64_t n=1){stats[size_t(i)].fetch_add(n,std::memory_order_relaxed);}
class Lease final {
 uint8_t*host_[2]{};uint8_t*device_[2]{};
 cudaEvent_t copied_[2]{},used_[2]{};cudaStream_t stream_{};
 bool submitted_[2]{},closed_=false,registered_=false;
 size_t bytes_;int device_id_,next_layer_=0,open_layer_=-1;
 uint64_t request_epoch_,pass_epoch_,generation_,format_;
 const std::thread::id issuer_=std::this_thread::get_id();
 static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
 void current()const{
  int d=-1;check(cudaGetDevice(&d));
  if(closed_||issuer_!=std::this_thread::get_id()||d!=device_id_)throw std::logic_error("558 lease issuer/device/closed mismatch");
 }
 void slot(int s)const{if(s<0||s>1)throw std::out_of_range("558 exactly two slots");}
 void drain(){if(stream_)check(cudaStreamSynchronize(stream_));check(cudaStreamSynchronize(0));}
 void release()noexcept{
  // A failed/unknown drain cannot release DMA storage, even while unwinding.
  try{current();if(open_layer_!=-1)throw std::logic_error("558 borrowed layer on close");drain();add(24+device_id_);}
  catch(...){add(12+device_id_);std::terminate();}
  for(int i=0;i<2;i++){
   if(copied_[i]&&cudaEventDestroy(copied_[i])!=cudaSuccess)std::terminate();
   if(used_[i]&&cudaEventDestroy(used_[i])!=cudaSuccess)std::terminate();
   if(device_[i]&&cudaFree(device_[i])!=cudaSuccess)std::terminate();
   if(host_[i]&&cudaFreeHost(host_[i])!=cudaSuccess)std::terminate();
  }
  if(stream_&&cudaStreamDestroy(stream_)!=cudaSuccess)std::terminate();
  closed_=true;
  if(registered_){
   add(8+device_id_);stats[size_t(10+device_id_)].fetch_sub(1);
   stats[size_t(18+device_id_)].fetch_sub(2*bytes_);stats[size_t(20+device_id_)].fetch_sub(2*bytes_);
   registered_=false;
  }
 }
public:
 Lease(size_t bytes,int device,uint64_t request_epoch,uint64_t pass_epoch,uint64_t generation,uint64_t format)
  :bytes_(bytes),device_id_(device),request_epoch_(request_epoch),pass_epoch_(pass_epoch),generation_(generation),format_(format){
  if(!bytes||bytes>SIZE_MAX/2||(device!=0&&device!=1)||!request_epoch||!pass_epoch||!generation||!format)
   throw std::invalid_argument("558 typed rank/pass/source binding required");
  current();
  try{
   check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
   for(int i=0;i<2;i++){
    check(cudaHostAlloc(reinterpret_cast<void**>(&host_[i]),bytes,cudaHostAllocDefault));
    check(cudaMalloc(reinterpret_cast<void**>(&device_[i]),bytes));
    check(cudaEventCreateWithFlags(&copied_[i],cudaEventDisableTiming));check(cudaEventCreateWithFlags(&used_[i],cudaEventDisableTiming));
   }
  }catch(...){add(12+device_id_);release();throw;}
  registered_=true;add(2+device_id_);add(10+device_id_);add(18+device_id_,2*bytes_);add(20+device_id_,2*bytes_);
  stats[size_t(22+device_id_)].store(request_epoch_,std::memory_order_relaxed);
 }
 ~Lease(){if(!closed_)release();}
 Lease(const Lease&)=delete;Lease&operator=(const Lease&)=delete;
 void validate(size_t bytes,int device,uint64_t request_epoch,uint64_t pass_epoch,uint64_t generation,uint64_t format){
  current();
  if(bytes!=bytes_||device!=device_id_||request_epoch!=request_epoch_||pass_epoch!=pass_epoch_||generation!=generation_||format!=format_)
   throw std::logic_error("558 immutable pass lease changed");
 }
 void begin_layer(int layer){
  current();if(open_layer_!=-1||layer!=next_layer_||layer<0||layer>=40)throw std::logic_error("558 ordered layer borrow required");
  open_layer_=layer;
 }
 void end_layer(bool success)noexcept{
  try{
   current();if(open_layer_!=next_layer_)throw std::logic_error("558 matching borrowed layer required");
   // Success releases only the HOST enqueue borrow. Slot events and the
   // pass owner retain all DMA/consumer lifetimes; no GPU completion claim.
   if(success){add(4+device_id_);++next_layer_;}
   else{
    // Failure may occur after upload but before used() was recorded. Drain
    // BOTH streams while every owner is alive; do not await a missing event.
    drain();add(6+device_id_);add(26+device_id_);add(12+device_id_);
   }
   open_layer_=-1;
  }catch(...){add(12+device_id_);std::terminate();}
 }
 uint8_t*begin(int s){current();slot(s);if(open_layer_<0)throw std::logic_error("558 slot outside layer");if(submitted_[s])check(cudaEventSynchronize(copied_[s]));return host_[s];}
 uint8_t*device(int s){current();slot(s);if(open_layer_<0)throw std::logic_error("558 device slot outside layer");return device_[s];}
 void upload(int s,size_t bytes){
  current();slot(s);if(open_layer_<0||!bytes||bytes>bytes_)throw std::invalid_argument("558 bounded upload in borrowed layer required");
  if(submitted_[s])check(cudaStreamWaitEvent(stream_,used_[s],0));
  check(cudaMemcpyAsync(device_[s],host_[s],bytes,cudaMemcpyHostToDevice,stream_));
  check(cudaEventRecord(copied_[s],stream_));check(cudaStreamWaitEvent(0,copied_[s],0));
  submitted_[s]=true;add(14+device_id_);add(16+device_id_,bytes);
 }
 void used(int s){current();slot(s);if(open_layer_<0||!submitted_[s])throw std::logic_error("558 use without owned upload");check(cudaEventRecord(used_[s],0));}
 void close(){if(!closed_){current();release();}}
};
// Original mode4 and flagOFF delegate literally to the existing534 view.
// A317 pass owns Lease; successful view exit releases only its host enqueue borrow.
class PipelineView final {
 Lease*pass_;fast534::PipelineView original_;int exceptions_;
public:
 PipelineView(Lease*pass,fast534::UploadLease*fast,bool old_enabled,size_t bytes,int layer)
  :pass_(pass),original_(fast,pass?false:old_enabled,bytes),exceptions_(pass?std::uncaught_exceptions():0){
  if(pass_){if(fast||!old_enabled)throw std::logic_error("558 pass cannot alias mode4/disabled pipeline");pass_->begin_layer(layer);}
 }
 ~PipelineView(){if(pass_)pass_->end_layer(std::uncaught_exceptions()==exceptions_);}
 PipelineView(const PipelineView&)=delete;PipelineView&operator=(const PipelineView&)=delete;
 explicit operator bool()const noexcept{return pass_||bool(original_);}
 PipelineView*operator->()noexcept{return this;}
 uint8_t*begin(int s){return pass_?pass_->begin(s):original_->begin(s);}
 uint8_t*device(int s){return pass_?pass_->device(s):original_->device(s);}
 void upload(int s,size_t b){if(pass_)pass_->upload(s,b);else original_->upload(s,b);}
 void used(int s){if(pass_)pass_->used(s);else original_->used(s);}
};
}
extern "C" __attribute__((visibility("default"))) int ds558_pass_upload_info(uint64_t*out,int words)noexcept{
 if(!out||words!=strata::ds41::pass558::InfoWords)return 1;
 try{
  out[0]=558;out[1]=strata::ds41::pass558::InfoWords;
  for(int i=2;i<words;i++)out[i]=strata::ds41::pass558::stats[size_t(i)].load(std::memory_order_relaxed);
  return 0;
 }catch(...){return 5;}
}
