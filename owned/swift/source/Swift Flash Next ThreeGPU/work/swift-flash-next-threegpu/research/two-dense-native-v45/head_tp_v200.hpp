#pragma once
// Private source ABI only. No existing SDK/class layout changes.
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
namespace strata::core {class Verifier;class NativeHead;}
namespace head_tp_v200 {
struct View {
 const strata::core::Verifier* verifier=nullptr;const strata::core::NativeHead* head=nullptr;
 int T=0,capacity=0,device=-1;bool batch=false,split=false,profile=false,prelaunched=false,nan=false;
 float*mixed=nullptr;void*q8=nullptr;void*il=nullptr;float*logits=nullptr;
 void*arg=nullptr;int32_t*out=nullptr;uint32_t*nf=nullptr;cudaStream_t stream=nullptr;
};
bool requested();
class Owner {
 struct Impl;std::unique_ptr<Impl>p_;
 explicit Owner(std::unique_ptr<Impl>);
public:
 ~Owner();Owner(const Owner&)=delete;Owner&operator=(const Owner&)=delete;
 // Setup only after all normal cache/startup sizing, no windows/captures in flight.
 static std::unique_ptr<Owner> create(const strata::core::Verifier*,const strata::core::NativeHead*,cudaStream_t,std::string&);
 bool prepare(const View&,std::string&);bool cut(int T,bool ar_off,bool batch,bool split,bool profile);
 bool finish(const View&,bool ar_off,std::string&);
};
// No-op for unbound stages/flag0. Binding remains immutable through all captures.
bool prepare(const View&,std::string&);
bool cut(const strata::core::Verifier*,int T,bool ar_off,bool batch,bool split,bool profile);
bool finish(const View&,bool ar_off,std::string&);
bool reject_nonordinary(const strata::core::Verifier*,const char*api,std::string&);
}
