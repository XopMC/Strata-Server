#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
namespace ds1017_gu_batch_v5 {
constexpr int Rows=2304,K=5120,Groups=160,MaxExperts=12,MaxQueries=6,MaxPairs=48;
constexpr size_t WeightBytes=size_t(Rows/16)*Groups*288;
constexpr size_t MaxOutputWords=2ull*MaxPairs*Rows;
constexpr size_t MaxScratchWords=MaxOutputWords*Groups;
struct Entry {uint32_t weight,count,output_words,scratch_words,query[6],reserved[2];};
static_assert(sizeof(Entry)==48);
struct alignas(16) Request {
 uint64_t generation;
 uint32_t experts,pairs,output_words,reserved[11];
 Entry entry[MaxExperts];
 int8_t codes[MaxQueries*K];
 float scales[MaxQueries*Groups];
 int32_t sums[MaxQueries*(K/4)];
};
static_assert(offsetof(Request,entry)==64&&sizeof(Request)==65920);
// Caller validates complete HOST request and immutable table ownership before upload.
// One stream, N1 issuer, persistent disjoint table/request/scratch/output, retained
// through completion. No allocation/sync/status retry. Two kernels total per batch.
cudaError_t launch(const uint8_t*const*weights,const Request*request,int experts,
                   uint32_t output_words,int32_t*scratch,float*out,cudaStream_t stream);
}
