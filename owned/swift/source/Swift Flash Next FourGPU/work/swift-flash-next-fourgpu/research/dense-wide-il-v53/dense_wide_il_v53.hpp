#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
namespace swift_wide53 {
inline constexpr int In=2560,Out=10240,MaxT=8,Type=14;
inline constexpr size_t RowBytes=2100,WeightBytes=Out*RowBytes;
// Exactly the original wide launcher's request-to-effective row mapping.
int effective_rows(int requested);
// Explicit finite experiment; no table, process-wide tune or oracle mutation.
// All spans belong to the caller's current owner and must remain live until
// its original stream positively completes. T5..8 only, native Q6_K rows.
// Caller produces original native Q8 + original native IL before this call.
void launch(const void*weights,const void*il,float*output,int rows,int T,int requested,cudaStream_t stream);
} // namespace swift_wide53
