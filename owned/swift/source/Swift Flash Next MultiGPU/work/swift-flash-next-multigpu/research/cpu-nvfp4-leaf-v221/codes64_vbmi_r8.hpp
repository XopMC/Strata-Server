#pragma once
#include <immintrin.h>
#include <cstdint>

// Proposed exact replacement for codes64() only. NOT compiled or measured.
// Requires AVX512F + AVX512VBMI, both advertised by the project's EPYC9754.
// All other NVFP4/scale/VNNI/FMA/reduction operations stay unchanged.
static inline __m512i codes64_vbmi_r8(const uint8_t* qs) {
    const __m256i packed = _mm256_loadu_si256((const __m256i*)qs);
    const __m512i dup_index = _mm512_setr_epi64(0,0,1,1,2,2,3,3);
    const __m512i duplicated = _mm512_permutexvar_epi64(
        dup_index, _mm512_castsi256_si512(packed));
    const __m512i shifts = _mm512_setr_epi64(
        0x3830282018100800LL, 0x3c342c241c140c04LL,
        0x3830282018100800LL, 0x3c342c241c140c04LL,
        0x3830282018100800LL, 0x3c342c241c140c04LL,
        0x3830282018100800LL, 0x3c342c241c140c04LL);
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(shifts, duplicated),
                            _mm512_set1_epi8(15));
}

// Pure scalar specification for the requested exact parity test.
static inline void codes64_scalar_r8(const uint8_t* qs, uint8_t* out) {
    for (int s=0;s<4;++s) for(int j=0;j<8;++j) {
        out[16*s+j] = qs[8*s+j]&15;
        out[16*s+8+j] = qs[8*s+j]>>4;
    }
}
