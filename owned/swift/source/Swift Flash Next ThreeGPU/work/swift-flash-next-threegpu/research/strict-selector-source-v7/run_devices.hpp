#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
namespace threegpu_scope {
inline int count(){static const int n=[](){const char*s=std::getenv("STRATA_RUN_DEVICE_COUNT");if(s&&!std::strcmp(s,"2"))return 2;if(s&&!std::strcmp(s,"3"))return 3;std::fprintf(stderr,"threeGPU: explicit current-index device count2/3 required\n");std::abort();}();return n;}
inline int last(){return count()-1;}
inline int span(){return 48/count();}
inline bool mask(){const char*a=std::getenv("CUDA_VISIBLE_DEVICES"),*b=std::getenv("STRATA_RUN_GPU_UUIDS");return a&&b&&!std::strcmp(a,b)&&std::strlen(a)==size_t(count()*41-1);}
inline bool stage(int d,long long lo,long long hi,bool next){return d>=0&&d<count()&&lo==d*span()&&hi==(d+1)*span()&&next==(d<last());}
inline bool hardware(std::string&err){int n=0;if(!mask()||cudaGetDeviceCount(&n)!=cudaSuccess||n!=count()){err="threeGPU: current-run visibility mismatch";return false;}const char*m=std::getenv("STRATA_RUN_GPU_UUIDS");for(int d=0;d<n;++d){cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,d)!=cudaSuccess||p.major!=8||p.minor!=9){err="threeGPU: owning SM89 device required";return false;}char uuid[64];const auto*u=(const unsigned char*)p.uuid.bytes;std::snprintf(uuid,sizeof uuid,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);if(std::string(m+d*41,40)!=uuid){err="threeGPU: transient ordinal snapshot mismatch";return false;}}return true;}
}
