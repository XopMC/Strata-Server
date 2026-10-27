// Exact current modern source helper extraction; ggml/Strata MIT provenance retained in README. Component ONLY.
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
#if defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (vecdotq.cuh) - v_perm_b32 takes 3-bit byte indices, so the low and high halves
    // of the table are looked up and the index MSB picks between them: 4 perms per 8 values.
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(table);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}
__device__ __forceinline__ float ue4m3_to_f32_half(uint8_t x) {
    // The same value as ldexpf(1 + man/8, exp - 7) * 0.5 (ldexpf(man, -9) * 0.5 for exp 0), built from its bits: every
    // such value is exact in FP32, so this is equal for all 256 inputs - without the ldexpf and the branches it cost
    // in the decode experts' inner loop.  Normal: 2^(exp-8) * (1 + man/8) = exponent field exp + 119, mantissa man<<20.
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t exp = (x >> 3) & 0xF, man = x & 0x7;
    return exp == 0 ? (float) man * 0.0009765625f : __uint_as_float(((exp + 119u) << 23) | (man << 20));
}
template<int> struct Fmt; template<> struct Fmt<40>{static constexpr int qk=64,ipb=2,step=4;};
template<int> struct Split;
template<> struct Split<40> {   // NVFP4: vec_dot_nvfp4_q8_1's two 16-value sub-blocks, weights decoded once per call
    struct W { int2 v0[2], v1[2]; float sc[2]; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_nvfp4* bq4 = (const block_nvfp4*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int iqs0 = iqs + 2 * i, iqs1 = iqs0 + 1, is = iqs0 >> 1;
            r.v0[i] = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_fp4);
            r.v1[i] = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_fp4);
            r.sc[i] = ue4m3_to_f32_half(bq4->d[is]);
        }
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        float sum = 0.0f;                              // the same operations, in the same order, as the dot
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int is = (iqs + 2 * i) >> 1;
            const block_q8_1* bq8 = bq8_1 + (is >> 1);
            const int i8 = (is & 1) << 2;
            int sumi = ggml_cuda_dp4a(r.v0[i].x, get_int_b4(bq8->qs, i8 + 0), 0);
            sumi = ggml_cuda_dp4a(r.v0[i].y, get_int_b4(bq8->qs, i8 + 2), sumi);
            sumi = ggml_cuda_dp4a(r.v1[i].x, get_int_b4(bq8->qs, i8 + 1), sumi);
            sumi = ggml_cuda_dp4a(r.v1[i].y, get_int_b4(bq8->qs, i8 + 3), sumi);
            sum += r.sc[i] * __low2float(bq8->ds) * (float) sumi;
        }
        return sum;
    }
};
template<int TY, int NC, bool EXACT_N = false>
__device__ __forceinline__ void row_dot_20_sub8(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                int t, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    {
        const int k = t, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s8[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s8[c] = 0.0f;
    {
        const int k = t + 8, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s8[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(__fadd_rn(s[c], s16[c]), __fadd_rn(s8[c], 0.0f));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_80_sub16(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                 int t, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = t; k < 80; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    for (int k = t + 16; k < 64; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}
