#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace ds1017_exact_down_v53 {
constexpr int Rows=5120,K=2304,Groups=72,MaxExperts=12,MaxViews=36,MaxCount=6,MaxPairs=36;
constexpr size_t WeightBytes=size_t(Rows/16)*Groups*288;
constexpr size_t MaxOutputWords=size_t(MaxPairs)*Rows;
struct Entry{uint32_t weight,count,output_words,reserved,view[6],reserved_tail[2];};
static_assert(sizeof(Entry)==48);
struct alignas(16) Request{
 uint64_t generation;uint32_t experts,pairs,output_words,reserved[11];
 Entry entry[MaxExperts];int8_t codes[MaxViews*K];
 float scales[MaxViews*Groups];int32_t sums[MaxViews*(K/4)];
};
static_assert(offsetof(Request,entry)==64&&sizeof(Request)==176896);
// Single-projection adaptation only. Host validated immutable weight table,
// request extents/unique routes/per4 sums/finite bounds; allocation outside capture.
// No scratch and no host allocation/synchronization in this launch leaf.
cudaError_t launch(const uint8_t*const*weights,const Request*request,int experts,
 uint32_t output_words,float*out,cudaStream_t stream);
}
