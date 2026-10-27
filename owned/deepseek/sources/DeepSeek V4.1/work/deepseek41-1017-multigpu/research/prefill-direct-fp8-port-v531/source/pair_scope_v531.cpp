#include "pair_scope_v531.hpp"
#include "pair_fp8_v531.h"
#include <atomic>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace strata::ds41::pair531 {
namespace {
thread_local Scope*current=nullptr;
std::atomic<const void*>active_instance[2]{};
std::array<std::array<std::atomic<uint64_t>,16>,2>counts{};
[[noreturn]]void bad(const char*s){throw std::runtime_error(std::string("pair531 typed scope: ")+s);}
}
bool enabled(){static const bool value=[](){const char*v=std::getenv("DS1017_PREFILL_DIRECT_FP8_V531");return v&&!std::strcmp(v,"1");}();return value;}
Scope::Scope(const void*instance,const Inputs&inputs):instance_(instance),inputs_(inputs),pid_(getpid()){
 const char*flag=std::getenv("DS1017_PREFILL_DIRECT_FP8_V531");
 if(!enabled()||!flag||std::strcmp(flag,"1")||!instance_||inputs_.layout_tag!=139||inputs_.reserved||!inputs_.owner_generation||!inputs_.bank_format_epoch||current||(inputs_.expected_device!=0&&inputs_.expected_device!=1))bad("explicit exact139 N1 descriptor/instance/optin required");
 for(const auto&s:inputs_.source)if(!s.data||s.bytes!=6635520||s.bits)bad("exact139 bits0 source triple required");
 const void*expected=nullptr;
 if(!active_instance[inputs_.expected_device].compare_exchange_strong(expected,instance_,std::memory_order_acq_rel))bad("another typed call scope is active");
 current=this;counts[inputs_.expected_device][4]++;counts[inputs_.expected_device][12]=inputs_.owner_generation;counts[inputs_.expected_device][13]=inputs_.bank_format_epoch;
}
Scope::~Scope()noexcept{
 // Same host call stack owns this POD until all kernel enqueues return. No GPU
 // pointers or bank references survive the scope. No CUDA or allocator action.
 if(current!=this||pid_!=getpid())std::terminate();current=nullptr;
 const void*expected=instance_;if(!active_instance[inputs_.expected_device].compare_exchange_strong(expected,nullptr,std::memory_order_acq_rel))std::terminate();
}
void Scope::require_consumed()const{if(current!=this||pid_!=getpid()||!consumed_)bad("linked encoder did not consume exact scoped triple");}
bool encode_if_scoped(const void*instance,const std::array<Source,3>&source,const std::array<Destination,3>&dest){
 auto*s=current;
 if(!s){for(int d=0;d<2;d++)if(active_instance[d].load(std::memory_order_acquire)==instance&&instance)bad("same typed instance called from wrong host thread");return false;}
 if(!instance||s->instance_!=instance||s->pid_!=getpid()||s->consumed_)bad("wrong instance/process or duplicate scope consumption");
 std::array<Ds531Projection,3>p{};
 for(int i=0;i<3;i++){
  const auto&a=source[i];const auto&b=s->inputs_.source[i];const auto&d=dest[i];
  if(a.data!=b.data||a.bytes!=b.bytes||a.bits!=b.bits||a.bits||d.rows!=(i==2?5120:2304)||d.columns!=(i==2?2304:5120)||!d.fp8||!d.row_scales)bad("method source/destination metadata differs from explicit139 scope");
  p[i]={sizeof(Ds531Projection),139,0,a.data,a.bytes,d.fp8,uint64_t(d.rows)*uint64_t(d.columns),d.row_scales,uint64_t(d.rows)*sizeof(float),d.rows,d.columns};
 }
 if(ds531_pair32_fp8_encode(p.data(),p.size(),s->inputs_.expected_device,s->inputs_.stream)){counts[s->inputs_.expected_device][10]++;bad(ds531_pair32_fp8_error());}
 auto&n=counts[s->inputs_.expected_device];n[5]++;n[6]+=3;n[7]+=19906560;n[8]+=35389440;n[9]+=38912;
 s->consumed_=true;return true;
}
}

extern "C" int ds531_prefill_direct_info(int device,uint64_t*out,size_t words){
 using namespace strata::ds41::pair531;
 if((device!=0&&device!=1)||!out||words!=16)return -1;
 for(size_t i=0;i<16;i++)out[i]=counts[device][i].load(std::memory_order_relaxed);
 out[0]=531;out[1]=16;out[2]=device;out[3]=enabled()?1:0;out[11]=active_instance[device].load(std::memory_order_acquire)?1:0;return 0;
}
