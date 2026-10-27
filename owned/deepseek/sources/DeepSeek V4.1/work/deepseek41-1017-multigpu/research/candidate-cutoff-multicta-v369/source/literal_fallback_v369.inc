template <bool Candidate>
__global__ void fallback369(Values<Candidate> values, int64_t n,
                                       int k, int32_t offset, int32_t* out,
                                       uint8_t* cand, const int* pos, int ratio, bool small_only, const Ds369State* control) {
    if(control->mode==3)return;
    if (pos) {
        values.positions = live_length(pos, ratio, values.positions);
        if (values.positions == 0) return;
        n = Candidate ? (values.positions - 1) / values.block_size + 1 : values.positions;
        k = int(n < k ? n : k);
        if (small_only && !Candidate && n >= kRadixItems && k > 0 && k < n) return;
        if (k <= 0) {
            if constexpr (Candidate) {
                for (int64_t j = threadIdx.x; j < values.positions; j += blockDim.x) cand[j] = 0;
            }
            return;
        }
    }
    __shared__ Count bins[256];
    __shared__ Selection state;
    __shared__ uint32_t greater_prefix[32], equal_prefix[32];
    __shared__ uint32_t batch_greater, batch_equal;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    uint32_t prefix = 0, prefix_mask = 0;
    constexpr int first_shift = Candidate ? 24 : 8;
    for (int shift = first_shift; shift >= 0; shift -= 8) {
        bins[tid] = 0;
        __syncthreads();
        for (int64_t base = 0; base < n; base += kThreads) {
            const int64_t j = base + tid;
            const uint32_t key = j < n ? key_of<!Candidate>(values.get(j)) : 0;
            const bool valid = j < n && (key & prefix_mask) == prefix;
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const uint32_t bin = (key >> shift) & 255u;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, Count(__popc(peers)));
            }
        }
        __syncthreads();
        if (tid == 0) {
            Count remaining = shift == first_shift ? Count(k) : state.remaining;
            for (int bin = 255; bin >= 0; --bin) {
                if (bins[bin] >= remaining) {
                    state.prefix = prefix | (uint32_t(bin) << shift);
                    state.remaining = remaining;
                    break;
                }
                remaining -= bins[bin];
            }
        }
        __syncthreads();
        prefix = state.prefix;
        prefix_mask |= 255u << shift;
    }
    const Count quota = state.remaining;
    Count greater_before = 0, equal_before = 0;
    // Four consecutive 256-position parts reduce barrier overhead. The first
    // warp scans all 32 (part, warp) totals in positional order. Its exclusive
    // prefixes then give both the tie quota and the ascending output address.
    for (int64_t base = 0; base < n; base += kSelectItems) {
        unsigned greater_lanes[4], equal_lanes[4];
        uint32_t keys[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kThreads + tid;
            const uint32_t key = j < n ? key_of<!Candidate>(values.get(j)) : 0;
            keys[part] = key;
            greater_lanes[part] = __ballot_sync(kFullWarp, j < n && key > prefix);
            equal_lanes[part] = __ballot_sync(kFullWarp, j < n && key == prefix);
            if (lane == 0) {
                greater_prefix[part * 8 + warp] = __popc(greater_lanes[part]);
                equal_prefix[part * 8 + warp] = __popc(equal_lanes[part]);
            }
        }
        __syncthreads();
        if (warp == 0) {
            const uint32_t own_greater = greater_prefix[lane];
            const uint32_t own_equal = equal_prefix[lane];
            uint32_t greater = own_greater, equal = own_equal;
#pragma unroll
            for (int delta = 1; delta < 32; delta <<= 1) {
                const uint32_t previous_greater = __shfl_up_sync(kFullWarp, greater, delta);
                const uint32_t previous_equal = __shfl_up_sync(kFullWarp, equal, delta);
                if (lane >= delta) { greater += previous_greater; equal += previous_equal; }
            }
            greater_prefix[lane] = greater - own_greater;
            equal_prefix[lane] = equal - own_equal;
            if (lane == 31) { batch_greater = greater; batch_equal = equal; }
        }
        __syncthreads();
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kThreads + tid;
            const Count preceding_greater = greater_before + greater_prefix[part * 8 + warp]
                + __popc(greater_lanes[part] & lane_mask);
            const Count preceding_equal = equal_before + equal_prefix[part * 8 + warp]
                + __popc(equal_lanes[part] & lane_mask);
            const bool selected = j < n && (keys[part] > prefix ||
                (keys[part] == prefix && preceding_equal < quota));
            if constexpr (Candidate) {
                if (j < n) {
                    const uint8_t keep = selected && keys[part] != key_of<false>(-CUDART_INF_F);
                    const int64_t begin = j * values.block_size;
                    for (int d = 0; d < values.block_size && begin + d < values.positions; ++d)
                        cand[begin + d] = keep;
                }
            } else if (selected) {
                out[preceding_greater + (preceding_equal < quota ? preceding_equal : quota)] = int32_t(j) + offset;
            }
        }
        greater_before += batch_greater;
        equal_before += batch_equal;
        __syncthreads();
    }
}

