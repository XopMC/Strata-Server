#pragma once
// Private PAIR-only host admission. Tokens are supplied afresh by Python's
// current NVIDIA-index0/1 resolver; no UUID table or fixed PCI address exists.
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
namespace ds501 {
using Uuid = std::array<unsigned char,16>;
struct Pair {std::array<Uuid,2> uuid;};
inline void need(bool ok,const char*message){if(!ok)throw std::invalid_argument(message);}
inline int hex(char c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;}
inline Uuid parse_uuid(const std::string& value){
 need(value.size()==40&&value.compare(0,4,"GPU-")==0,"501 canonical transient GPU identifier required");
 Uuid out{};unsigned bytes=0;int high=-1;
 for(unsigned i=4;i<40;i++){
  const unsigned j=i-4;
  if(j==8||j==13||j==18||j==23){need(value[i]=='-',"501 malformed UUID separator");continue;}
  const int n=hex(value[i]);need(n>=0,"501 malformed UUID digit");
  if(high<0)high=n;else{need(bytes<16,"501 UUID overflow");out[bytes++]=static_cast<unsigned char>((high<<4)|n);high=-1;}
 }
 need(bytes==16&&high<0,"501 incomplete UUID");return out;
}
inline Pair require_environment(){
 const char*expected=std::getenv("DS41_EXPECTED_PAIR_UUIDS");
 const char*visible=std::getenv("CUDA_VISIBLE_DEVICES");
 const char*pair=std::getenv("DS41_NATIVE_PAIR_MASK");
 const char*stage=std::getenv("DEEPSEEK41_GPU_STAGE");
 const char*order=std::getenv("CUDA_DEVICE_ORDER");
 need(pair&&!std::strcmp(pair,"1")&&stage&&!std::strcmp(stage,"0"),"501 private pair-only mode required");
 need(order&&!std::strcmp(order,"PCI_BUS_ID"),"501 explicit CUDA device order required");
 need(expected&&visible&&!std::strcmp(expected,visible),"501 visibility differs from current index0/1 snapshot");
 const std::string text(expected);need(text.size()==81&&text[40]==',',"501 exactly two explicit transient identifiers required");
 Pair p{{parse_uuid(text.substr(0,40)),parse_uuid(text.substr(41,40))}};
 need(p.uuid[0]!=p.uuid[1],"501 two distinct current GPUs required");return p;
}
inline void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
inline void validate_device(int ordinal){
 const auto expected=require_environment();need(ordinal==0||ordinal==1,"501 ordinal outside admitted pair");
 int count=0;check(cudaGetDeviceCount(&count));need(count==2,"501 exactly two CUDA-visible devices required");
 cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,ordinal));
 need(p.major==8&&p.minor==9,"501 selected numerical kernels require SM89");
 need(!std::memcmp(p.uuid.bytes,expected.uuid[ordinal].data(),16),"501 CUDA ordinal differs from current NVIDIA index snapshot");
}
inline void validate_all(){require_environment();validate_device(0);validate_device(1);}
} // namespace ds501
