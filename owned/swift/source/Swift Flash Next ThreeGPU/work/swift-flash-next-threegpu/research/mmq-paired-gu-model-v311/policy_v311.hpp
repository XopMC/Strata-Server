#pragma once
#include "strata_owned/device_admission.hpp"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
namespace paired_model_v311 {
struct Policy{bool valid=true,on=false;};
inline const Policy& policy(){static const Policy p=[] {Policy v;const char*s=std::getenv("STRATA_PAIRED_GU_V311");if(!s||!std::strcmp(s,"0"))return v;if(!std::strcmp(s,"1")){v.on=true;return v;}v.valid=false;return v;}();return p;}
inline bool literal_one(const char*name){const char*p=std::getenv(name);return p&&!std::strcmp(p,"1");}
inline bool hardware(std::string&err){
 if(!strata_owned::require_selected_pair(err))return false;
 for(int d=0;d<2;++d){cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,d)!=cudaSuccess||p.major!=8||p.minor!=9){err="V311 selected owning SM89 required";return false;}}
 return true;
}
}
