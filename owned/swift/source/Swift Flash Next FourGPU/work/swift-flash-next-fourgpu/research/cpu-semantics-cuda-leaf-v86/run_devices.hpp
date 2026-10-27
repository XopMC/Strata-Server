#pragma once
// V86 finite admission only: explicit current0/1 or current0/1/2/3.
// No3device compatibility mode, inferred visible count or default-all scope.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
namespace v86_scope {
inline int count(){
 static const int n=[](){const char*s=std::getenv("STRATA_RUN_DEVICE_COUNT");
  if(s&&!std::strcmp(s,"2"))return 2;
  if(s&&!std::strcmp(s,"4"))return 4;
  throw std::runtime_error("V86 requires explicit device count2or4; no default/3/all-device mode");
 }();return n;
}
inline bool hardware(std::string&err){try{
 const int admitted=count();const char*mask=std::getenv("CUDA_VISIBLE_DEVICES");const char*snap=std::getenv("STRATA_RUN_GPU_UUIDS");
 if(!mask||!snap||std::strcmp(mask,snap)||std::strlen(mask)!=size_t(admitted*41-1))throw std::runtime_error("V86 requires fresh restricted current-index UUID snapshot");
 int visible=0;if(cudaGetDeviceCount(&visible)!=cudaSuccess||visible!=admitted)throw std::runtime_error("V86 admitted/visible count mismatch");
 for(int d=0;d<admitted;++d){cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,d)!=cudaSuccess||p.major!=8||p.minor!=9)throw std::runtime_error("V86 numerical candidate requires admittedSM89");
  char id[41];const auto*u=reinterpret_cast<const unsigned char*>(p.uuid.bytes);
  std::snprintf(id,sizeof id,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  if(std::string(snap+d*41,40)!=id||(d+1<admitted&&snap[d*41+40]!=','))throw std::runtime_error("V86 CUDA ordinal differs from transient current-index snapshot");
 }return true;
}catch(const std::exception&e){err=e.what();return false;}}
}
