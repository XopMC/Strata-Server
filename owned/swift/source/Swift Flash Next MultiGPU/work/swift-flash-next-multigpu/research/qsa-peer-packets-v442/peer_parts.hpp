#pragma once
#include "selector16.hpp"
namespace qsa_peer_v441 {
constexpr int PacketWords=8+2*8*swift_selector_v7_leaf::Keep;
cudaError_t inputs(const float*,const float*,const int32_t*,const float*,float*,float*,int32_t*,float*,int,cudaStream_t);
cudaError_t score(int,const float*,const float*,const float*,const int32_t*,int64_t,float*,int64_t,int64_t,cudaStream_t);
cudaError_t chunks(const float*,const int32_t*,int,int64_t,swift_selector_v7_leaf::Workspace,int,int,cudaStream_t);
cudaError_t gather(const float*,int64_t,swift_selector_v7_leaf::Workspace,uint32_t*,int,cudaStream_t);
cudaError_t publish(float*,int64_t,swift_selector_v7_leaf::Workspace,const uint32_t*,int,cudaStream_t);
}
extern "C" void* swift_score_kernel112(int);
