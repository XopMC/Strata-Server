// v63: expand the PP-only BF16 wire to the original F32 compute buffer. No weights or reductions.
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::prefill {
namespace {
__global__ void wire_bf16_expand_v63(const uint16_t* __restrict__ wire, float* __restrict__ out, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x * blockDim.x)
        out[i] = __uint_as_float((uint32_t) wire[i] << 16);
}
}
cudaError_t wire_bf16_unpack_v63(const uint16_t* wire, float* out, int64_t n, cudaStream_t stream) {
    if (n < 0 || (n > 0 && (!wire || !out))) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    const int64_t wanted = (n + 255) / 256;
    const unsigned blocks = (unsigned) (wanted < 4096 ? wanted : 4096);
    wire_bf16_expand_v63<<<blocks, 256, 0, stream>>>(wire, out, n);
    return cudaGetLastError();
}
}  // namespace strata::prefill
