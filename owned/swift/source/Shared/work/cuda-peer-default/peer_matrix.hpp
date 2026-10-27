#pragma once
#include "strata/core/required_native_peer.hpp"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
namespace strata_peer_default {
inline bool require_all(int admitted,std::string& error) noexcept {
 try {
  const char* mask=std::getenv("CUDA_VISIBLE_DEVICES");
  const char* fresh=std::getenv("STRATA_RUN_GPU_UUIDS");
  const char* indices=std::getenv("STRATA_RUN_GPU_INDICES");
  if(!mask || !fresh || !indices || std::string(mask)!=fresh) {error="owned P2P requires this-launch visibility and current system-index snapshot";return false;}
  strata::core::RequiredNativePeerOptions options;
  options.admitted_devices=admitted;
  options.binding=strata::core::NativePeerBinding::FreshUuidSnapshot;
  options.before_model_allocation_and_capture=true;
  std::stringstream ids(indices),uuids(fresh);std::string item;
  while(std::getline(ids,item,',')){size_t used=0;int value=std::stoi(item,&used);if(used!=item.size())throw std::invalid_argument("index token");options.current_system_indices.push_back(value);}
  while(std::getline(uuids,item,','))options.transient_cuda_uuids.push_back(item);
  strata::core::RequiredNativePeerReceipt receipt;
  if(!strata::core::require_native_peer(options,receipt,error))return false;
  std::fprintf(stderr,"STRATA_P2P_DEFAULT devices=%d directed_pairs=%d native_required=1 host_fallback=0\n",receipt.devices,receipt.directed_pairs);
  return true;
 }catch(const std::exception& e){error=std::string("owned P2P admission: ")+e.what();return false;}
 catch(...){error="owned P2P admission metadata failure";return false;}
}
}
