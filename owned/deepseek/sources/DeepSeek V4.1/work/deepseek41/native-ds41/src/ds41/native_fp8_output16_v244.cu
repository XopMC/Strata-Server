// Isolated component for lossless CPUoutput16→GPUfeeding. No engine integration.
// Row-encoder arithmetic copied unchanged from native_fp8lt_prototype.cu v78:
// one256-thread CTA/outputrow, same column-stride256 values and max tree,
// exact sourceBF16 rounding, same pow2 row scale, then E4M3 encode.
#include "strata/ds41/native_fp8_output16_v244.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void validate(int bits, int rows, int columns) {
    if ((bits != 0 && bits != 3) || rows < 16 || rows > 32768 || rows % 16 ||
        (columns != 5120 && columns != 2304))
        throw std::invalid_argument("v244 component requires actual Mixed25 bits0|3/output16/native widths");
}
size_t compact_bytes(int bits, int rows, int columns) {
    return size_t(rows / 16) * (columns / 32) * (64 * (bits ? bits : 4) + 32);
}

template<int Bits, int Width>
__device__ __forceinline__ const uint8_t* compact_block(const uint8_t* blocked, int row, int col) {
    constexpr int width = Bits ? Bits : 4;
    return blocked + (size_t(row / 16) * (Width / 32) + col / 32) * (64 * width + 32);
}
template<int Bits, int Width>
__device__ __forceinline__ unsigned compact_code(const uint8_t* blocked, int row, int col) {
    constexpr int width = Bits ? Bits : 4;
    const uint8_t* block = compact_block<Bits, Width>(blocked, row, col);
    const uint8_t* group = block + ((col % 32) / 4) * (8 * width) + ((row % 16) / 2) * width;
    // width3 groups are not word-aligned. Read exactly width bytes: no
    // unaligned uint32 cast or read beyond a compact allocation.
    unsigned packed = 0;
#pragma unroll
    for (int i = 0; i < width; ++i) packed |= unsigned(group[i]) << (8 * i);
    const int shift = (((row % 16) % 2) * 4 + col % 4) * width;
    return (packed >> shift) & ((1u << width) - 1u);
}
template<int Bits, int Width>
__device__ __forceinline__ uint16_t compact_scale(const uint8_t* blocked, int row, int col) {
    constexpr int width = Bits ? Bits : 4;
    const uint8_t* scale = compact_block<Bits, Width>(blocked, row, col) + 64 * width + 2 * (row % 16);
    return uint16_t(scale[0]) | (uint16_t(scale[1]) << 8);
}

template<int Bits, int Width, bool Compact>
__device__ __forceinline__ float source_bf16_value(const uint8_t* w, const uint8_t* s, int row, int col) {
    float value;
    if constexpr (Bits == 0) {
        const int code = Compact ? int(compact_code<Bits, Width>(w, row, col))
            : (w[size_t(row) * (Width / 2) + col / 2] >> (4 * (col % 2))) & 15;
        const int magnitude = code & 7;
        const uint32_t raw = magnitude < 2 ? (magnitude ? 0x3f000000u : 0u)
            : (uint32_t((magnitude >> 1) + 126) << 23) | (uint32_t(magnitude & 1) << 22);
        const float decoded = __uint_as_float(raw | ((code & 8) ? 0x80000000u : 0u));
        const uint8_t scale = Compact ? uint8_t(compact_scale<Bits, Width>(w, row, col))
            : s[size_t(row) * (Width / 32) + col / 32];
        const float factor = scale == 255 ? nanf("")
            : __uint_as_float(scale ? uint32_t(scale) << 23 : 0x00400000u);
        value = decoded * factor;
    } else {
        int code; float scale;
        if constexpr (Compact) {
            code = int(compact_code<Bits, Width>(w, row, col));
            scale = __uint_as_float(uint32_t(compact_scale<Bits, Width>(w, row, col)) << 16);
        } else {
            constexpr size_t stride = size_t(Width) * Bits / 8 + size_t(Width / 32) * 2;
            const uint8_t* base = w + size_t(row) * stride;
            const int at = col * Bits, byte = at / 8;
            code = ((int(base[byte]) | (int(base[byte + 1]) << 8)) >> (at % 8)) & ((1 << Bits) - 1);
            const uint8_t* ss = base + size_t(Width) * Bits / 8 + (col / 32) * 2;
            scale = __uint_as_float(uint32_t(uint16_t(ss[0]) | (uint16_t(ss[1]) << 8)) << 16);
        }
        value = (code == (1 << Bits) - 1 || !isfinite(scale) || scale < 0) ? nanf("")
            : float(code - ((1 << (Bits - 1)) - 1)) * scale;
    }
    return __bfloat162float(__float2bfloat16_rn(value));
}

template<int Bits, int Width, bool Compact>
__global__ void encode_rows(const uint8_t* w, const uint8_t* s, uint8_t* q, float* scales) {
    constexpr int Iterations = (Width + 255) / 256;
    const int row = blockIdx.x;
    float values[Iterations], maximum = 1e-4f;
    __shared__ float shared[256];
#pragma unroll
    for (int it = 0; it < Iterations; ++it) {
        const int c = threadIdx.x + it * 256;
        values[it] = c < Width ? source_bf16_value<Bits, Width, Compact>(w, s, row, c) : 0.f;
        if (c < Width) maximum = fmaxf(maximum, fabsf(values[it]));
    }
    shared[threadIdx.x] = maximum; __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (threadIdx.x < d) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + d]);
        __syncthreads();
    }
    const float raw = shared[0] * (1.f / 448.f);
    const uint32_t bits = __float_as_uint(raw);
    const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
    const float scale = ldexpf(1.f, exponent);
    if (threadIdx.x == 0) scales[row] = scale;
#pragma unroll
    for (int it = 0; it < Iterations; ++it) {
        const int c = threadIdx.x + it * 256;
        if (c < Width) {
            const __nv_fp8_e4m3 value(values[it] / scale);
            q[size_t(row) * Width + c] = value.__x;
        }
    }
}

template<int Bits, int Width>
__global__ void inverse_to_original_rows(const uint8_t* compact, int rows, uint8_t* w, uint8_t* s) {
    constexpr int code_bytes = Width * (Bits ? Bits : 4) / 8;
    constexpr int groups = Width / 32;
    constexpr int stride = Bits ? code_bytes + groups * 2 : code_bytes;
    const size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= size_t(rows) * stride) return;
    const int row = at / stride, byte = at % stride;
    if (byte < code_bytes) {
        constexpr int width = Bits ? Bits : 4;
        const int first_col = (byte * 8) / width, shift = (byte * 8) % width;
        unsigned word = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (first_col + i < Width) word |= compact_code<Bits, Width>(compact, row, first_col + i) << (i * width);
        }
        w[at] = uint8_t(word >> shift);
    } else {
        const int group = (byte - code_bytes) / 2;
        const auto value = compact_scale<Bits, Width>(compact, row, group * 32);
        w[at] = uint8_t(value >> (8 * ((byte - code_bytes) % 2)));
    }
    if constexpr (Bits == 0) {
        if (byte < groups) s[size_t(row) * groups + byte] = uint8_t(compact_scale<Bits, Width>(compact, row, byte * 32));
    }
}

template<bool Compact>
void encode_dispatch(const uint8_t* w, const uint8_t* s, int bits, int rows, int k,
                     uint8_t* q, float* scales, cudaStream_t stream) {
    if (bits == 0 && k == 5120) encode_rows<0, 5120, Compact><<<rows, 256, 0, stream>>>(w, s, q, scales);
    else if (bits == 0) encode_rows<0, 2304, Compact><<<rows, 256, 0, stream>>>(w, s, q, scales);
    else if (k == 5120) encode_rows<3, 5120, Compact><<<rows, 256, 0, stream>>>(w, s, q, scales);
    else encode_rows<3, 2304, Compact><<<rows, 256, 0, stream>>>(w, s, q, scales);
    check(cudaGetLastError());
}
}

extern "C" const char* ds41_fp8_output16_v244_error() { return error.c_str(); }
extern "C" int ds41_fp8_row_encode_raw_v244(const uint8_t* w, const uint8_t* s, int bits,
    int rows, int k, uint8_t* q, float* scales, uint64_t stream) {
    try { error.clear(); validate(bits, rows, k);
        if (!w || !q || !scales || (!bits && !s)) throw std::invalid_argument("raw row encode pointers");
        encode_dispatch<false>(w, s, bits, rows, k, q, scales, reinterpret_cast<cudaStream_t>(stream)); return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
extern "C" int ds41_fp8_row_encode_output16_v244(const uint8_t* compact, size_t bytes, int bits,
    int rows, int k, uint8_t* q, float* scales, uint64_t stream) {
    try { error.clear(); validate(bits, rows, k);
        if (!compact || !q || !scales || bytes != compact_bytes(bits, rows, k)) throw std::invalid_argument("compact row encode pointers/bytes");
        encode_dispatch<true>(compact, nullptr, bits, rows, k, q, scales, reinterpret_cast<cudaStream_t>(stream)); return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
extern "C" int ds41_output16_to_rowpacked_gpu_v244(const uint8_t* compact, size_t bytes, int bits,
    int rows, int k, uint8_t* w, size_t wbytes, uint8_t* s, size_t sbytes, uint64_t stream) {
    try { error.clear(); validate(bits, rows, k);
        const size_t expected_w = size_t(rows) * (bits ? size_t(k) * bits / 8 + size_t(k / 32) * 2 : size_t(k / 2));
        const size_t expected_s = bits ? 0 : size_t(rows) * (k / 32);
        if (!compact || !w || bytes != compact_bytes(bits, rows, k) || wbytes != expected_w || sbytes != expected_s || (bits ? s != nullptr : s == nullptr))
            throw std::invalid_argument("compact inverse output pointers/byte lengths");
        const unsigned blocks = unsigned((expected_w + 255) / 256);
        auto st = reinterpret_cast<cudaStream_t>(stream);
        if (bits == 0 && k == 5120) inverse_to_original_rows<0, 5120><<<blocks, 256, 0, st>>>(compact, rows, w, s);
        else if (bits == 0) inverse_to_original_rows<0, 2304><<<blocks, 256, 0, st>>>(compact, rows, w, s);
        else if (k == 5120) inverse_to_original_rows<3, 5120><<<blocks, 256, 0, st>>>(compact, rows, w, s);
        else inverse_to_original_rows<3, 2304><<<blocks, 256, 0, st>>>(compact, rows, w, s);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
