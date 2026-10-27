#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Component-only APIs. Device pointers and caller-owned CUDA stream; no model.
// main Mixed25 source_bits0=native E2M1/E8M0 or3=Uniform/BF16group32.
int ds41_fp8_row_encode_raw_v244(const uint8_t* weights, const uint8_t* native_scales,
    int source_bits, int rows, int columns, uint8_t* fp8, float* row_scales, uint64_t stream);
int ds41_fp8_row_encode_output16_v244(const uint8_t* compact, size_t compact_bytes,
    int source_bits, int rows, int columns, uint8_t* fp8, float* row_scales, uint64_t stream);
// Byte-only inverse for GPU cached-FFN initialization. Existing GEMV stays unchanged.
int ds41_output16_to_rowpacked_gpu_v244(const uint8_t* compact, size_t compact_bytes,
    int source_bits, int rows, int columns, uint8_t* row_weights, size_t weight_bytes,
    uint8_t* native_scales, size_t scale_bytes, uint64_t stream);
const char* ds41_fp8_output16_v244_error(void);
#ifdef __cplusplus
}
#endif
