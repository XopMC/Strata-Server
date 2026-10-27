#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace mapped_moe_v350 {
struct Args {
 const float* cpu=nullptr;const float* hits=nullptr;
 const int32_t* dst=nullptr;const int32_t* count=nullptr;
 const float* weights=nullptr;const float* shared=nullptr;float* output=nullptr;
 float* parts_receipt=nullptr; // optional private proof output, NEVER aliases live inputs
 int width=2560,k=10,tokens=0;cudaStream_t stream=nullptr;
};
bool enabled();
// Declines before enqueue with admitted=false. Once admitted, errors are fatal
// to this arm; caller must drain and reject, never replay the original chain.
cudaError_t launch(const Args&,bool& admitted);
void* function();
}
