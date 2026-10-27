#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
// Portable launch policy only: mathematical kernels and placement are unchanged.
// The launcher resolves current system indices into a fresh ordered UUID snapshot.
namespace strata_owned {
inline bool require_selected_pair(std::string& error) {
 const char* mask=std::getenv("CUDA_VISIBLE_DEVICES");
 const char* snapshot=std::getenv("STRATA_RUN_GPU_UUIDS");
 if(!mask||!*mask||!snapshot||std::strcmp(mask,snapshot)) {error="Fresh selected-device snapshot required";return false;}
 const std::string selected(snapshot);const auto comma=selected.find(',');
 if(comma==std::string::npos||selected.find(',',comma+1)!=std::string::npos||comma==0||comma+1==selected.size()) {error="Exactly two ordered selected devices required";return false;}
 const std::string expected[2]={selected.substr(0,comma),selected.substr(comma+1)};
 if(expected[0]==expected[1]) {error="Selected devices must differ";return false;}
 int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess||n!=2) {error="Exactly two CUDA-visible devices required";return false;}
 for(int d=0;d<2;++d) {
  cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,d)!=cudaSuccess) {error="Selected device identity query failed";return false;}
  const auto* u=reinterpret_cast<const unsigned char*>(p.uuid.bytes);char actual[64];
  std::snprintf(actual,sizeof actual,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  if(expected[d]!=actual) {error="Selected CUDA ordering differs from this launch snapshot";return false;}
 }
 return true;
}
}
