#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
// K14-02: head-streamed 16-query x 128-key BF16 tensor-core score tiles.
// All scratch belongs to the caller; no allocation, host sync, or per-query launch.
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"

#include <mma.h>
#include <math_constants.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
using Count = unsigned long long;
constexpr int kQueryTile = 16;
constexpr int kKeyTile = 128;
constexpr int kHeadGroup = 8;
constexpr int kInputStride = 136;
constexpr int kScoreThreads = 512;
constexpr int kSelectThreads = 256;
constexpr int kBatchRows = 256;
constexpr size_t kAlignment = 256;
constexpr unsigned kFullWarp = 0xffffffffu;

size_t checked_product(size_t a, size_t b) {
    // Pointer differences must also remain representable, not just size_t.
    constexpr size_t limit = size_t(std::numeric_limits<ptrdiff_t>::max());
    if (a && b > limit / a) throw std::invalid_argument("K14: size overflow");
    return a * b;
}

size_t score_bytes(int m, int64_t t_max) {
    if (m < 1 || m > 16384 || t_max < 0)
        throw std::invalid_argument("K14: invalid workspace shape");
    if (uint64_t(t_max) > uint64_t(std::numeric_limits<ptrdiff_t>::max()))
        throw std::invalid_argument("K14: size overflow");
    return checked_product(checked_product(size_t(std::min(m, kBatchRows)), size_t(t_max)), sizeof(bf16));
}

void check_launch(const char* operation) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__device__ __forceinline__ float rounded(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Stage eight heads from every query, compute them, then reuse the same
// storage for their FP32 dots. Keys remain resident through all four groups.
// The +8 BF16 input stride rotates consecutive rows across shared banks.
union __align__(32) QueryDots {
    bf16 queries[kQueryTile * kHeadGroup * kInputStride];
    float dots[kQueryTile * kHeadGroup * kKeyTile];
};
struct __align__(32) ScoreTile {
    QueryDots group;
    bf16 keys[kKeyTile * kInputStride];
    bf16 weights[kQueryTile * kHeadGroup];
};
static_assert(sizeof(QueryDots) == 64 * 1024, "query/dot storage size");
static_assert(sizeof(ScoreTile) == 100608, "score tile storage size");
static_assert(sizeof(ScoreTile) <= 99 * 1024, "consumer shared-memory budget");
static_assert(kQueryTile * kKeyTile % kScoreThreads == 0, "whole thread-owned scores");
constexpr int kThreadScores = kQueryTile * kKeyTile / kScoreThreads;

__global__ __launch_bounds__(kScoreThreads) void batched_scores(
        const bf16* q, const bf16* keys, const bf16* w,
        int first, int rows, int pos0, int ratio,
        const uint8_t* cand, int64_t cand_stride,
        bf16* scores, int64_t score_stride) {
    extern __shared__ __align__(32) unsigned char shared[];
    ScoreTile& tile = *reinterpret_cast<ScoreTile*>(shared);
    const int tid = threadIdx.x;
    const int tile_row = int(blockIdx.y) * kQueryTile;
    const int tile_rows = min(kQueryTile, rows - tile_row);
    const int64_t key_base = int64_t(blockIdx.x) * kKeyTile;
    const int64_t tile_end = (int64_t(pos0) + first + tile_row + tile_rows) / ratio;
    if (key_base >= tile_end) return;  // Uniform, before any barrier.

    for (int item = tid; item < kKeyTile * 128; item += kScoreThreads) {
        const int key_row = item / 128, d = item % 128;
        const int64_t j = key_base + key_row;
        tile.keys[key_row * kInputStride + d] = j < tile_end
            ? keys[size_t(j) * 128 + d] : __float2bfloat16_rn(0.0f);
    }
    float total[kThreadScores] = {};
    const int warp = tid >> 5;
    const int query_pair = warp / 2;
    const int key_parity = warp % 2;
    namespace wm = nvcuda::wmma;

    // Ascending groups and ascending heads within each group preserve the
    // reference's FP32 head-sum order, including the group boundaries.
#pragma unroll 1
    for (int head_base = 0; head_base < 32; head_base += kHeadGroup) {
        for (int item = tid; item < kQueryTile * kHeadGroup * 128; item += kScoreThreads) {
            const int qh = item / 128, d = item % 128;
            const int local_row = qh / kHeadGroup, head = qh % kHeadGroup;
            tile.group.queries[qh * kInputStride + d] = local_row < tile_rows
                ? q[(size_t(first + tile_row + local_row) * 32 + head_base + head) * 128 + d]
                : __float2bfloat16_rn(0.0f);
        }
        if (tid < kQueryTile * kHeadGroup) {
            const int local_row = tid / kHeadGroup, head = tid % kHeadGroup;
            tile.weights[tid] = local_row < tile_rows
                ? w[size_t(first + tile_row + local_row) * 32 + head_base + head]
                : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();  // Queries, weights, and the retained keys are ready.

        // Each warp computes two queries x eight heads against four disjoint
        // 16-key groups. All 16 warps together cover the 128x128 dot matrix.
        wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major> a;
        wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major> b;
        wm::fragment<wm::accumulator, 16, 16, 16, float> dots[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) wm::fill_fragment(dots[part], 0.0f);
#pragma unroll
        for (int d = 0; d < 128; d += 16) {
            wm::load_matrix_sync(a, tile.group.queries + query_pair * 16 * kInputStride + d, kInputStride);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int key_group = key_parity + part * 2;
                wm::load_matrix_sync(b, tile.keys + key_group * 16 * kInputStride + d, kInputStride);
                wm::mma_sync(dots[part], a, b, dots[part]);
            }
        }
        __syncthreads();  // Retire every query reader before the overlay stores.
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int key_group = key_parity + part * 2;
            wm::store_matrix_sync(tile.group.dots + query_pair * 16 * kKeyTile + key_group * 16,
                                  dots[part], kKeyTile, wm::mem_row_major);
        }
        __syncthreads();  // Every dot is now visible to its owning score thread.
#pragma unroll
        for (int part = 0; part < kThreadScores; ++part) {
            const int output = tid + part * kScoreThreads;
            const int local_row = output / kKeyTile, key_col = output % kKeyTile;
#pragma unroll
            for (int head = 0; head < kHeadGroup; ++head) {
                const float dot = rounded(tile.group.dots[(local_row * kHeadGroup + head) * kKeyTile + key_col]);
                total[part] += rounded(fmaxf(dot, 0.0f)
                    * __bfloat162float(tile.weights[local_row * kHeadGroup + head]));
            }
        }
        __syncthreads();  // Retire dot/weight readers before staging the next group.
    }
#pragma unroll
    for (int part = 0; part < kThreadScores; ++part) {
        const int output = tid + part * kScoreThreads;
        const int local_row = output / kKeyTile, key_col = output % kKeyTile;
        const int row = first + tile_row + local_row;
        const int64_t j = key_base + key_col;
        const int64_t n = (int64_t(pos0) + row + 1) / ratio;
        if (local_row < tile_rows && j < n) {
            const float value = !cand || cand[size_t(row) * size_t(cand_stride) + size_t(j)]
                ? total[part] : -CUDART_INF_F;
            scores[size_t(tile_row + local_row) * size_t(score_stride) + size_t(j)] = __float2bfloat16_rn(value);
        }
    }
}

#include "register_score_v334.inc"

// Exact selector copied without functional changes from reviewed K14-01,
// commit 1e8f6d22166620116d76d81c1c00851f4b100285, production SHA-256:
// d061923b0df41325df17949a730cfb694b184f13e314f2a7a4def368d818a11e.
// All stored scores and block maxima are BF16 values (or infinity). Float-flip
// therefore needs only two radix bytes. Signed zeros compare equal by contract.
__device__ __forceinline__ uint32_t key_of(float value) {
    const uint32_t bits = value == 0.0f ? 0u : __float_as_uint(value);
    return (bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u)) >> 16;
}

template <bool Candidate>
struct Values {
    const bf16* scores;
    int64_t positions;
    int block_size;
    __device__ __forceinline__ float get(int64_t j) const {
        if constexpr (!Candidate) {
            return __bfloat162float(scores[j]);
        } else {
            if (j == (positions - 1) / block_size) return CUDART_INF_F;
            float value = -CUDART_INF_F;
            const int64_t begin = j * int64_t(block_size);
            for (int64_t d = 0; d < block_size && begin + d < positions; ++d)
                value = fmaxf(value, __bfloat162float(scores[begin + d]));
            return value;
        }
    }
};

// Candidate-only score tile: every query owns its selected key list, so
// stage all32 heads of ONE query against64 selected keys. Preserve K14's
// BF16 dot/product rounding and ascending FP32 head reduction exactly.
union __align__(32) CompactQueryDots {
    bf16 q[32*136];
    float dots[32*64];
};
struct __align__(32) CompactScoreTile {
    CompactQueryDots query;
    bf16 keys[64*136];
    bf16 weights[32];
    int32_t positions[64];
};
static_assert(sizeof(CompactScoreTile)<32*1024,"compact indexer bounded tile");
__global__ __launch_bounds__(128) void compact_scores(
    const bf16*q,const bf16*keys,const bf16*w,const int32_t*blocks,int first,int pos0,int ratio,bf16*scores){
    __shared__ CompactScoreTile tile;
    const int tid=threadIdx.x,warp=tid/32,row=first+blockIdx.y,base=blockIdx.x*64;
    const int64_t available=(int64_t(pos0)+row+1)/ratio;
    if(tid<64){
        const int block=blocks[size_t(row)*2048+(base+tid)/8];
        const int64_t position=int64_t(block)*8+(base+tid)%8;
        tile.positions[tid]=block>=0&&position<available?int32_t(position):-1;
    }
    if(tid<32)tile.weights[tid]=w[size_t(row)*32+tid];
    for(int item=tid;item<32*128;item+=128)tile.query.q[(item/128)*136+item%128]=q[(size_t(row)*32+item/128)*128+item%128];
    __syncthreads();
    for(int item=tid;item<64*128;item+=128){
        const int j=tile.positions[item/128];
        tile.keys[(item/128)*136+item%128]=j>=0?keys[size_t(j)*128+item%128]:__float2bfloat16_rn(0.f);
    }
    __syncthreads();
    namespace wm=nvcuda::wmma;
    wm::fragment<wm::accumulator,16,16,16,float> dots[2];
    wm::fill_fragment(dots[0],0.f);wm::fill_fragment(dots[1],0.f);
    const int head=(warp/2)*16,col=(warp%2)*32;
    for(int d=0;d<128;d+=16){
        wm::fragment<wm::matrix_a,16,16,16,bf16,wm::row_major>a;
        wm::load_matrix_sync(a,tile.query.q+head*136+d,136);
        for(int part=0;part<2;part++){
            wm::fragment<wm::matrix_b,16,16,16,bf16,wm::col_major>b;
            wm::load_matrix_sync(b,tile.keys+(col+part*16)*136+d,136);
            wm::mma_sync(dots[part],a,b,dots[part]);
        }
    }
    __syncthreads();
    for(int part=0;part<2;part++)wm::store_matrix_sync(tile.query.dots+head*64+col+part*16,dots[part],64,wm::mem_row_major);
    __syncthreads();
    if(tid<64){
        float sum=0.f;
        for(int h=0;h<32;h++)sum+=rounded(fmaxf(rounded(tile.query.dots[h*64+tid]),0.f)*__bfloat162float(tile.weights[h]));
        scores[size_t(blockIdx.y)*16384+base+tid]=__float2bfloat16_rn(tile.positions[tid]>=0?sum:-CUDART_INF_F);
    }
}
__global__ void map_compact_topk(int32_t*indices,const int32_t*blocks,int first,int m,int pos0,int ratio){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=m*512)return;
    const int row=first+i/512,local=indices[size_t(row)*512+i%512];
    int position=-1;
    if(local>=0){const int block=blocks[size_t(row)*2048+local/8];const int64_t p=int64_t(block)*8+local%8;
        if(block>=0&&p<(int64_t(pos0)+row+1)/ratio)position=int(p);}
    indices[size_t(row)*512+i%512]=position;
}

struct Selection { uint32_t prefix; Count remaining; };

// K5's ordered cutoff/emission, parallelized over query rows. One CTA owns each
// complete row, so there is no global counter, inter-CTA scan, or hidden scratch.
// Warp-private histogram counters are bounded by ceil(t_i/8)+32 < 2^32 because
// pos0 is an int and m<=16384; their reduction and all positional counts use u64.
template <bool Candidate, bool Blocks = false, bool FixedPositions = false>
__global__ void select_rows(const bf16* scores, int64_t score_stride, int first,
                            int pos0, int ratio, int k, int32_t offset,
                            int topk_blocks, int block_size, int32_t* out_idx,
                            uint8_t* cand_out, int64_t cand_stride, int32_t* block_ids = nullptr) {
    __shared__ uint32_t warp_bins[8][256];
    __shared__ Count bins[256];
    __shared__ Selection state;
    __shared__ uint32_t greater_prefix[32], equal_prefix[32], batch_greater, batch_equal;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row = first + int(blockIdx.x);
    const int64_t positions = FixedPositions ? score_stride : (int64_t(pos0) + row + 1) / ratio;
    const Values<Candidate> values{scores + size_t(blockIdx.x) * size_t(score_stride), positions, block_size};
    int32_t* out = out_idx + size_t(row) * size_t(k);
    uint8_t* candidate = nullptr;
    if constexpr (Candidate && !Blocks) candidate = cand_out + size_t(row) * size_t(cand_stride);
    int32_t* compact = nullptr;
    if constexpr (Blocks) {
        compact = block_ids + size_t(row) * size_t(topk_blocks);
        for (int j = tid; j < topk_blocks; j += kSelectThreads) compact[j] = -1;
        __syncthreads();
    }
    if constexpr (!Candidate) {
        for (int64_t j = min(int64_t(k), positions) + tid; j < k; j += kSelectThreads) out[j] = -1;
    }
    if (!positions) return;
    const int64_t n = Candidate ? (positions - 1) / block_size + 1 : positions;
    const int64_t wanted = Candidate ? min(int64_t(topk_blocks), n) : min(int64_t(k), n);
    if constexpr (Candidate) {
        if (wanted <= 0) {
            if constexpr (!Blocks) for (int64_t j = tid; j < positions; j += kSelectThreads) candidate[j] = 0;
            return;
        }
    }
    if (wanted == n) {
        for (int64_t j = tid; j < n; j += kSelectThreads) {
            if constexpr (Candidate) {
                const uint8_t keep = values.get(j) != -CUDART_INF_F;
                const int64_t begin = j * int64_t(block_size);
                if constexpr (Blocks) { if (keep) compact[j] = int32_t(j); }
                else for (int64_t d = 0; d < block_size && begin + d < positions; ++d) candidate[begin + d] = keep;
            } else {
                out[j] = int32_t(j + int64_t(offset));
            }
        }
        return;
    }

    uint32_t prefix = 0, prefix_mask = 0;
    for (int shift = 8; shift >= 0; shift -= 8) {
#pragma unroll
        for (int w = 0; w < 8; ++w) warp_bins[w][tid] = 0;
        __syncthreads();
        for (int64_t base = 0; base < n; base += kSelectThreads) {
            const int64_t j = base + tid;
            const uint32_t key = j < n ? key_of(values.get(j)) : 0;
            const bool valid = j < n && (key & prefix_mask) == prefix;
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const uint32_t bin = (key >> shift) & 255u;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1) atomicAdd(&warp_bins[warp][bin], uint32_t(__popc(peers)));
            }
        }
        __syncthreads();
        Count count = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) count += Count(warp_bins[w][tid]);
        bins[tid] = count;
        __syncthreads();
        if (tid == 0) {
            Count remaining = shift == 8 ? Count(wanted) : state.remaining;
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
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    Count greater_before = 0, equal_before = 0;
    for (int64_t base = 0; base < n; base += 4 * kSelectThreads) {
        unsigned greater_lanes[4], equal_lanes[4];
        uint32_t keys[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kSelectThreads + tid;
            const uint32_t key = j < n ? key_of(values.get(j)) : 0;
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
            const uint32_t own_greater = greater_prefix[lane], own_equal = equal_prefix[lane];
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
            const int64_t j = base + part * kSelectThreads + tid;
            const Count preceding_greater = greater_before + greater_prefix[part * 8 + warp]
                + __popc(greater_lanes[part] & lane_mask);
            const Count preceding_equal = equal_before + equal_prefix[part * 8 + warp]
                + __popc(equal_lanes[part] & lane_mask);
            const bool selected = j < n && (keys[part] > prefix || (keys[part] == prefix && preceding_equal < quota));
            if constexpr (Candidate) {
                if (j < n) {
                    const uint8_t keep = selected && keys[part] != key_of(-CUDART_INF_F);
                    const int64_t begin = j * int64_t(block_size);
                    if constexpr (Blocks) {
                        if (keep) compact[preceding_greater + (preceding_equal < quota ? preceding_equal : quota)] = int32_t(j);
                    } else for (int64_t d = 0; d < block_size && begin + d < positions; ++d) candidate[begin + d] = keep;
                }
            } else if (selected) {
                out[preceding_greater + (preceding_equal < quota ? preceding_equal : quota)] = int32_t(j + int64_t(offset));
            }
        }
        greater_before += batch_greater;
        equal_before += batch_equal;
        __syncthreads();
    }
}
}  // namespace

size_t indexer_topk_prefill_workspace_bytes(int m, int64_t t_max) {
    const size_t bytes = score_bytes(m, t_max);
    size_t extra=0;
    const char*s8=std::getenv("DS41_INDEXER_S8_R1");
    if(s8&&std::strcmp(s8,"1")==0&&m>=256&&t_max>=32LL*m){
        const size_t a=checked_product(size_t(t_max),133),b=checked_product(size_t(m),4256);
        const size_t limit=size_t(std::numeric_limits<ptrdiff_t>::max());
        if(a>limit-b-7*kAlignment)throw std::invalid_argument("K14 S8 workspace overflow");
        extra=a+b+7*kAlignment;
    }

    if (!bytes) return 0;
    if (bytes > size_t(std::numeric_limits<ptrdiff_t>::max()) - (kAlignment - 1) || extra > size_t(std::numeric_limits<ptrdiff_t>::max()) - (kAlignment - 1) - bytes)
        throw std::invalid_argument("K14: size overflow");
    return bytes + extra + kAlignment - 1;
}

bool try_s8_full_r1(const bf16*,const bf16*,const bf16*,int,int,int,const uint8_t*,uint8_t*,int64_t,int,int32_t,int,int,int32_t*,void*,size_t,cudaStream_t,int32_t*);
void indexer_topk_prefill_impl(const bf16* q, const bf16* keys, const bf16* w, int m, int pos0,
                          int ratio, const uint8_t* cand, uint8_t* cand_out, int64_t cand_stride,
                          int k, int32_t offset, int topk_blocks, int block, int32_t* out_idx,
                          void* workspace, size_t workspace_bytes, cudaStream_t stream, int32_t* block_ids) {
    if (m < 1 || m > 16384 || ratio < 1 || k < 1 || pos0 < 0 || !out_idx || (cand_out && block < 1))
        throw std::invalid_argument("K14: invalid arguments");
    const int64_t t_max = (int64_t(pos0) + m) / ratio;
    const size_t required = indexer_topk_prefill_workspace_bytes(m, t_max);
    if (workspace_bytes < required || (required && !workspace))
        throw std::invalid_argument("K14: insufficient workspace");
    if (t_max && (!q || !keys || !w)) throw std::invalid_argument("K14: null score input");
    if (t_max && t_max - 1 + int64_t(offset) > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("K14: output index overflow");
    checked_product(size_t(m), checked_product(size_t(k), sizeof(int32_t)));
    checked_product(size_t(t_max), 128 * sizeof(bf16));
    if (cand || cand_out) {
        if (cand_stride < t_max) throw std::invalid_argument("K14: candidate stride is too small");
        const size_t before_last = checked_product(size_t(m - 1), size_t(cand_stride));
        if (before_last > size_t(std::numeric_limits<ptrdiff_t>::max()) - size_t(t_max))
            throw std::invalid_argument("K14: candidate size overflow");
    }
    if(block_ids && (topk_blocks < 1 || topk_blocks > 2048 || block < 1))
        throw std::invalid_argument("K14: invalid compact candidate geometry");
    if(block_ids && !t_max) {
        const auto status=cudaMemsetAsync(block_ids,0xff,size_t(m)*topk_blocks*sizeof(int32_t),stream);
        if(status!=cudaSuccess)throw std::runtime_error("K14 compact padding failed");
    }
    if (!t_max) {
        const cudaError_t error = cudaMemsetAsync(out_idx, 0xff, size_t(m) * size_t(k) * sizeof(int32_t), stream);
        if (error != cudaSuccess) throw std::runtime_error("K14: output padding failed");
        return;
    }
    if(try_s8_full_r1(q,keys,w,m,pos0,ratio,cand,cand_out,cand_stride,k,offset,topk_blocks,block,out_idx,workspace,workspace_bytes,stream,block_ids))return;
    // Opt in for this function on the current device, without allocating
    // scratch, querying device properties, or synchronizing a stream. Repeating
    // the same function attribute avoids process-global per-device cache state.
    const bool on334=score_flag334();
    const bool selected334=on334&&ratio==2&&!cand&&!cand_out&&!block_ids;
    score_stats334[2].fetch_add(1,std::memory_order_relaxed);
    if(selected334){score_stats334[3].fetch_add(1,std::memory_order_relaxed);score_stats334[4].fetch_add(m,std::memory_order_relaxed);}
    else score_stats334[on334?6:5].fetch_add(1,std::memory_order_relaxed);
    const cudaError_t attribute_error = selected334
        ? cudaFuncSetAttribute(batched_scores334,cudaFuncAttributeMaxDynamicSharedMemorySize,int(sizeof(ScoreTile334)))
        : cudaFuncSetAttribute(batched_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,int(sizeof(ScoreTile)));
    if (attribute_error != cudaSuccess)
        throw std::runtime_error(std::string("K14 shared-memory opt-in: ") + cudaGetErrorString(attribute_error));
    // The API includes alignment slack, so even a byte-aligned workspace works.
    const size_t adjustment = (kAlignment - (reinterpret_cast<uintptr_t>(workspace) % kAlignment)) % kAlignment;
    bf16* scores = reinterpret_cast<bf16*>(static_cast<unsigned char*>(workspace) + adjustment);
    for (int first = 0; first < m; first += kBatchRows) {
        const int rows = std::min(kBatchRows, m - first);
        const int64_t batch_end = (int64_t(pos0) + first + rows) / ratio;
        if (batch_end) {
            const dim3 grid(unsigned((batch_end - 1) / kKeyTile + 1), unsigned((rows + kQueryTile - 1) / kQueryTile));
            if(selected334){
                batched_scores334<<<grid,512,sizeof(ScoreTile334),stream>>>(q,keys,w,first,rows,pos0,ratio,scores,t_max);
                record_launch334(rows,uint64_t(grid.x)*grid.y,uint64_t(batch_end));
            }else batched_scores<<<grid,kScoreThreads,sizeof(ScoreTile),stream>>>(q,keys,w,first,rows,pos0,ratio,
                                                              cand,cand_stride,scores,t_max);
            check_launch("K14 batched scores");
        }
        select_rows<false><<<rows, kSelectThreads, 0, stream>>>(scores, t_max, first, pos0, ratio, k, offset,
                                                               topk_blocks, block, out_idx, nullptr, cand_stride);
        check_launch("K14 top-k rows");
        if(block_ids) {
            select_rows<true,true><<<rows,kSelectThreads,0,stream>>>(scores,t_max,first,pos0,ratio,k,offset,
                topk_blocks,block,out_idx,nullptr,0,block_ids);
            check_launch("K14 compact candidate blocks");
        }
        if (cand_out) {
            select_rows<true><<<rows, kSelectThreads, 0, stream>>>(scores, t_max, first, pos0, ratio, k, offset,
                                                                  topk_blocks, block, out_idx, cand_out, cand_stride);
            check_launch("K14 candidate rows");
        }
    }
}
void indexer_topk_prefill(const bf16* q,const bf16* keys,const bf16* w,int m,int pos0,
    int ratio,const uint8_t* cand,uint8_t* cand_out,int64_t cand_stride,int k,int32_t offset,
    int topk_blocks,int block,int32_t* out_idx,void* workspace,size_t bytes,cudaStream_t stream) {
    indexer_topk_prefill_impl(q,keys,w,m,pos0,ratio,cand,cand_out,cand_stride,k,offset,topk_blocks,block,out_idx,workspace,bytes,stream,nullptr);
}
void indexer_topk_prefill_blocks(const bf16* q,const bf16* keys,const bf16* w,int m,int pos0,
    int ratio,int k,int topk_blocks,int block,int32_t* out_idx,int32_t* block_ids,void* workspace,size_t bytes,cudaStream_t stream) {
    if(!block_ids)throw std::invalid_argument("K14 missing compact candidate output");
    indexer_topk_prefill_impl(q,keys,w,m,pos0,ratio,nullptr,nullptr,0,k,0,topk_blocks,block,out_idx,workspace,bytes,stream,block_ids);
}
void indexer_topk_prefill_from_blocks(const bf16*q,const bf16*keys,const bf16*w,const int32_t*blocks,
    int m,int pos0,int ratio,int32_t*out_idx,void*workspace,size_t bytes,cudaStream_t stream){
    if(m<1||m>16384||pos0<0||pos0>1048576-m||ratio<1||ratio>2||!q||!keys||!w||!blocks||!out_idx)
        throw std::invalid_argument("K14 invalid compact-score input");
    const size_t needed=indexer_topk_prefill_workspace_bytes(m,16384);
    if(!workspace||bytes<needed)throw std::invalid_argument("K14 compact-score workspace too small");
    const size_t adjustment=(kAlignment-reinterpret_cast<uintptr_t>(workspace)%kAlignment)%kAlignment;
    auto*scores=reinterpret_cast<bf16*>(static_cast<uint8_t*>(workspace)+adjustment);
    for(int first=0;first<m;first+=kBatchRows){
        const int rows=std::min(kBatchRows,m-first);
        compact_scores<<<dim3(256,rows),128,0,stream>>>(q,keys,w,blocks,first,pos0,ratio,scores);
        check_launch("K14 compact selected scores");
        select_rows<false,false,true><<<rows,kSelectThreads,0,stream>>>(scores,16384,first,pos0,ratio,512,0,2048,8,out_idx,nullptr,0);
        check_launch("K14 compact selected topk");
        map_compact_topk<<<(rows*512+255)/256,256,0,stream>>>(out_idx,blocks,first,rows,pos0,ratio);
        check_launch("K14 compact mapped positions");
    }
}
}  // namespace strata::ds41::kernels

namespace strata::ds41::kernels {namespace {
__global__ void pack_lossless_rows(const bf16*input,int rows,int8_t*output,float*scale,uint8_t*valid){
    int lane=threadIdx.x%32,row=int(blockIdx.x)*4+threadIdx.x/32;if(row>=rows)return;
    float v[4],maxabs=0.f;
    for(int j=0;j<4;j++){v[j]=__bfloat162float(input[size_t(row)*128+lane+j*32]);maxabs=fmaxf(maxabs,fabsf(v[j]));}
    for(int d=16;d;d/=2)maxabs=fmaxf(maxabs,__shfl_xor_sync(0xffffffffu,maxabs,d));
    int e=maxabs==0.f?0:int((__float_as_uint(maxabs)>>23)&255)-127-6;
    bool ok=e>=-50&&e<=50&&isfinite(maxabs);float s=ldexpf(1.f,ok?e:0);
    for(int j=0;j<4;j++){
        int code=__float2int_rn(v[j]/s);ok=ok&&code>=-127&&code<=127&&isfinite(v[j])&&float(code)*s==v[j];
        output[size_t(row)*128+lane+j*32]=int8_t(code);
    }
    bool exact=__all_sync(0xffffffffu,ok);
    if(lane==0){scale[row]=s;valid[row]=uint8_t(exact);}
}
__device__ unsigned long long bf16_first_tile_r1=0,s8_first_tile_r1=0;
__global__ __launch_bounds__(kScoreThreads) void guarded_bf16_scores(
        const bf16* q, const bf16* keys, const bf16* w,
        int first, int rows, int pos0, int ratio,
        const uint8_t* cand, int64_t cand_stride,
        bf16* scores, int64_t score_stride,const uint8_t*qvalid,const uint8_t*kvalid) {
    extern __shared__ __align__(32) unsigned char shared[];
    ScoreTile& tile = *reinterpret_cast<ScoreTile*>(shared);
    const int tid = threadIdx.x;
    const int tile_row = int(blockIdx.y) * kQueryTile;
    const int tile_rows = min(kQueryTile, rows - tile_row);
    const int64_t key_base = int64_t(blockIdx.x) * kKeyTile;
    const int64_t tile_end = (int64_t(pos0) + first + tile_row + tile_rows) / ratio;
    if (key_base >= tile_end) return;  // Uniform, before any barrier.

    bool valid=tid>=512||tid/32>=tile_rows||qvalid[size_t(first+tile_row+tid/32)*32+tid%32];
    if(tid<128&&key_base+tid<tile_end)valid=valid&&kvalid[key_base+tid];
    if(__syncthreads_and(valid))return;
    if(blockIdx.x==0&&blockIdx.y==0&&tid==0)atomicAdd(&bf16_first_tile_r1,1ULL);
    for (int item = tid; item < kKeyTile * 128; item += kScoreThreads) {
        const int key_row = item / 128, d = item % 128;
        const int64_t j = key_base + key_row;
        tile.keys[key_row * kInputStride + d] = j < tile_end
            ? keys[size_t(j) * 128 + d] : __float2bfloat16_rn(0.0f);
    }
    float total[kThreadScores] = {};
    const int warp = tid >> 5;
    const int query_pair = warp / 2;
    const int key_parity = warp % 2;
    namespace wm = nvcuda::wmma;

    // Ascending groups and ascending heads within each group preserve the
    // reference's FP32 head-sum order, including the group boundaries.
#pragma unroll 1
    for (int head_base = 0; head_base < 32; head_base += kHeadGroup) {
        for (int item = tid; item < kQueryTile * kHeadGroup * 128; item += kScoreThreads) {
            const int qh = item / 128, d = item % 128;
            const int local_row = qh / kHeadGroup, head = qh % kHeadGroup;
            tile.group.queries[qh * kInputStride + d] = local_row < tile_rows
                ? q[(size_t(first + tile_row + local_row) * 32 + head_base + head) * 128 + d]
                : __float2bfloat16_rn(0.0f);
        }
        if (tid < kQueryTile * kHeadGroup) {
            const int local_row = tid / kHeadGroup, head = tid % kHeadGroup;
            tile.weights[tid] = local_row < tile_rows
                ? w[size_t(first + tile_row + local_row) * 32 + head_base + head]
                : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();  // Queries, weights, and the retained keys are ready.

        // Each warp computes two queries x eight heads against four disjoint
        // 16-key groups. All 16 warps together cover the 128x128 dot matrix.
        wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major> a;
        wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major> b;
        wm::fragment<wm::accumulator, 16, 16, 16, float> dots[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) wm::fill_fragment(dots[part], 0.0f);
#pragma unroll
        for (int d = 0; d < 128; d += 16) {
            wm::load_matrix_sync(a, tile.group.queries + query_pair * 16 * kInputStride + d, kInputStride);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int key_group = key_parity + part * 2;
                wm::load_matrix_sync(b, tile.keys + key_group * 16 * kInputStride + d, kInputStride);
                wm::mma_sync(dots[part], a, b, dots[part]);
            }
        }
        __syncthreads();  // Retire every query reader before the overlay stores.
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int key_group = key_parity + part * 2;
            wm::store_matrix_sync(tile.group.dots + query_pair * 16 * kKeyTile + key_group * 16,
                                  dots[part], kKeyTile, wm::mem_row_major);
        }
        __syncthreads();  // Every dot is now visible to its owning score thread.
#pragma unroll
        for (int part = 0; part < kThreadScores; ++part) {
            const int output = tid + part * kScoreThreads;
            const int local_row = output / kKeyTile, key_col = output % kKeyTile;
#pragma unroll
            for (int head = 0; head < kHeadGroup; ++head) {
                const float dot = rounded(tile.group.dots[(local_row * kHeadGroup + head) * kKeyTile + key_col]);
                total[part] += rounded(fmaxf(dot, 0.0f)
                    * __bfloat162float(tile.weights[local_row * kHeadGroup + head]));
            }
        }
        __syncthreads();  // Retire dot/weight readers before staging the next group.
    }
#pragma unroll
    for (int part = 0; part < kThreadScores; ++part) {
        const int output = tid + part * kScoreThreads;
        const int local_row = output / kKeyTile, key_col = output % kKeyTile;
        const int row = first + tile_row + local_row;
        const int64_t j = key_base + key_col;
        const int64_t n = (int64_t(pos0) + row + 1) / ratio;
        if (local_row < tile_rows && j < n) {
            const float value = !cand || cand[size_t(row) * size_t(cand_stride) + size_t(j)]
                ? total[part] : -CUDART_INF_F;
            scores[size_t(tile_row + local_row) * size_t(score_stride) + size_t(j)] = __float2bfloat16_rn(value);
        }
    }
}


}}

namespace strata::ds41::kernels { namespace {
constexpr int pure_stride=144;
union __align__(32) PureQuery {int8_t q[16*8*pure_stride];int32_t dots[16*8*128];};
struct __align__(32) PureTile {PureQuery group;int8_t keys[128*pure_stride];bf16 weights[16*8];};
static_assert(sizeof(PureTile)==84224,"bounded pure S8 shared");
__device__ __forceinline__ float& pure_qscale(PureTile&t,int i){return *reinterpret_cast<float*>(&t.keys[(i/4)*pure_stride+128+(i%4)*4]);}
__device__ __forceinline__ float& pure_kscale(PureTile&t,int i){return *reinterpret_cast<float*>(&t.keys[(32+i/4)*pure_stride+128+(i%4)*4]);}
__global__ __launch_bounds__(512) void pure_scores(const bf16*w,const int8_t*q,const int8_t*keys,const float*qs,const float*ks,
    const uint8_t*qvalid,const uint8_t*kvalid,int m,int pos,int ratio,bf16*scores,int64_t stride){
    extern __shared__ __align__(32) unsigned char memory[];auto&tile=*reinterpret_cast<PureTile*>(memory);
    int tid=threadIdx.x,warp=tid/32,lane=tid%32,qr=int(blockIdx.y)*16,rows=min(16,m-qr);int64_t kb=int64_t(blockIdx.x)*128,end=(int64_t(pos)+qr+rows)/ratio;
    if(kb>=end)return;
    bool valid=tid>=512||tid/32>=rows||qvalid[size_t(qr+tid/32)*32+tid%32];
    if(tid<128&&kb+tid<end)valid=valid&&kvalid[kb+tid];
    if(!__syncthreads_and(valid))return;
    if(blockIdx.x==0&&blockIdx.y==0&&tid==0)atomicAdd(&s8_first_tile_r1,1ULL);
    for(int item=tid;item<128*128;item+=512){int r=item/128,d=item%128;tile.keys[r*pure_stride+d]=kb+r<end?keys[size_t(kb+r)*128+d]:int8_t(0);}
    if(tid<128)pure_kscale(tile,tid)=kb+tid<end?ks[kb+tid]:1.f;
    float total[4]={};int qp=warp/2,parity=warp%2;
    for(int hb=0;hb<32;hb+=8){
        if(tid<128){int r=tid/8,h=tid%8;tile.weights[tid]=r<rows?w[size_t(qr+r)*32+hb+h]:__float2bfloat16_rn(0.f);pure_qscale(tile,tid)=r<rows?qs[size_t(qr+r)*32+hb+h]:1.f;}
        for(int item=tid;item<16*8*128;item+=512){int qh=item/128,d=item%128,r=qh/8,h=qh%8;tile.group.q[qh*pure_stride+d]=r<rows?q[(size_t(qr+r)*32+hb+h)*128+d]:int8_t(0);}
        __syncthreads();int dots[8][4]={{0}};
        for(int d=0;d<128;d+=32){
            unsigned a[4],aa=unsigned(__cvta_generic_to_shared(tile.group.q+qp*16*pure_stride+(lane%16)*pure_stride+(lane/16)*16+d));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(aa));
            for(int n=0;n<8;n++){
                int col=(parity+(n/2)*2)*16+(n%2)*8;unsigned b[2],ba=unsigned(__cvta_generic_to_shared(tile.keys+(col+lane%8)*pure_stride+((lane/8)%2)*16+d));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(b[0]),"=r"(b[1]):"r"(ba));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    :"+r"(dots[n][0]),"+r"(dots[n][1]),"+r"(dots[n][2]),"+r"(dots[n][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
            }
        }
        __syncthreads();
        for(int n=0;n<8;n++){int col=(parity+(n/2)*2)*16+(n%2)*8+(lane%4)*2;for(int r=0;r<4;r++)tile.group.dots[(qp*16+lane/4+(r/2)*8)*128+col+r%2]=dots[n][r];}
        __syncthreads();
        for(int p=0;p<4;p++){int output=tid+p*512,r=output/128,col=output%128;
            for(int h=0;h<8;h++){
                float dot=float(tile.group.dots[(r*8+h)*128+col])*(pure_qscale(tile,r*8+h)*pure_kscale(tile,col));
                total[p]+=rounded(fmaxf(rounded(dot),0.f)*__bfloat162float(tile.weights[r*8+h]));
            }
        }
        __syncthreads();
    }
    for(int p=0;p<4;p++){int output=tid+p*512,r=output/128,col=output%128;int64_t j=kb+col,n=(int64_t(pos)+qr+r+1)/ratio;if(r<rows&&j<n)scores[size_t(qr+r)*stride+j]=__float2bfloat16_rn(total[p]);}
}
}}
extern "C" int s8_pack(const void*in,int rows,void*out,void*scale,void*valid){strata::ds41::kernels::pack_lossless_rows<<<(rows+3)/4,128>>>((const __nv_bfloat16*)in,rows,(int8_t*)out,(float*)scale,(uint8_t*)valid);return int(cudaGetLastError());}
extern "C" int s8_info(int*out){using namespace strata::ds41::kernels;auto e=cudaFuncSetAttribute(pure_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PureTile));if(e!=cudaSuccess)return int(e);cudaFuncAttributes a{};e=cudaFuncGetAttributes(&a,pure_scores);if(e!=cudaSuccess)return int(e);int active=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,pure_scores,512,sizeof(PureTile));out[0]=sizeof(PureTile);out[1]=a.numRegs;out[2]=active;return int(e);}
extern "C" int s8_scores(const void*q,const void*k,const void*w,const void*qi,const void*ki,const void*qs,const void*ks,const void*qv,const void*kv,int m,int pos,int ratio,void*scores){
    using namespace strata::ds41::kernels;int64_t t=(int64_t(pos)+m)/ratio;dim3 grid((t+127)/128,(m+15)/16);
    auto e=cudaFuncSetAttribute(pure_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PureTile));if(e!=cudaSuccess)return int(e);
    e=cudaFuncSetAttribute(guarded_bf16_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ScoreTile));if(e!=cudaSuccess)return int(e);
    guarded_bf16_scores<<<grid,512,sizeof(ScoreTile)>>>((const bf16*)q,(const bf16*)k,(const bf16*)w,0,m,pos,ratio,nullptr,0,(bf16*)scores,t,(const uint8_t*)qv,(const uint8_t*)kv);
    pure_scores<<<grid,512,sizeof(PureTile)>>>((const bf16*)w,(const int8_t*)qi,(const int8_t*)ki,(const float*)qs,(const float*)ks,(const uint8_t*)qv,(const uint8_t*)kv,m,pos,ratio,(bf16*)scores,t);return int(cudaGetLastError());
}


namespace strata::ds41::kernels {
static std::atomic<uint64_t>s8_dispatch_r1{0};
bool try_s8_full_r1(const bf16*q,const bf16*keys,const bf16*w,int m,int pos,int ratio,const uint8_t*cand,uint8_t*cand_out,int64_t cand_stride,
    int k,int32_t offset,int blocks,int block,int32_t*out,void*workspace,size_t bytes,cudaStream_t stream,int32_t*block_ids){
    const char*enabled=std::getenv("DS41_INDEXER_S8_R1");
    if(!enabled||std::strcmp(enabled,"1")||ratio!=1||cand||cand_out||m<256)return false;
    const int64_t t=(int64_t(pos)+m)/ratio;if(t<32LL*m)return false;
    const size_t adjustment=(256-reinterpret_cast<uintptr_t>(workspace)%256)%256;
    size_t used=adjustment;
    auto take=[&](size_t n){used=adjustment+((used-adjustment+255)&~size_t(255));size_t before=used;used+=n;return before;};
    const size_t qbytes=size_t(m)*32*128,kbytes=size_t(t)*128;
    const size_t qi_off=take(qbytes),ki_off=take(kbytes),qs_off=take(size_t(m)*32*4),ks_off=take(size_t(t)*4),qv_off=take(size_t(m)*32),kv_off=take(size_t(t)),score_off=take(size_t(256)*t*2);
    if(used>bytes)return false;  // Caller geometry/storage remains unchanged.
    auto*base=static_cast<uint8_t*>(workspace);auto*qi=reinterpret_cast<int8_t*>(base+qi_off);auto*ki=reinterpret_cast<int8_t*>(base+ki_off);
    auto*qs=reinterpret_cast<float*>(base+qs_off);auto*ks=reinterpret_cast<float*>(base+ks_off);auto*qv=base+qv_off;auto*kv=base+kv_off;auto*scores=reinterpret_cast<bf16*>(base+score_off);
    auto e=cudaFuncSetAttribute(pure_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PureTile));if(e!=cudaSuccess)throw std::runtime_error("S8 shared optin");
    e=cudaFuncSetAttribute(guarded_bf16_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ScoreTile));if(e!=cudaSuccess)throw std::runtime_error("S8 fallback shared optin");
    pack_lossless_rows<<<(m*32+3)/4,128,0,stream>>>(q,m*32,qi,qs,qv);check_launch("S8 Q exact encoding");
    pack_lossless_rows<<<(t+3)/4,128,0,stream>>>(keys,int(t),ki,ks,kv);check_launch("S8 K exact encoding");
    for(int first=0;first<m;first+=256){
        const int rows=std::min(256,m-first);int64_t end=(int64_t(pos)+first+rows)/ratio;dim3 grid((end+127)/128,(rows+15)/16);
        const auto*qrow=q+size_t(first)*32*128;auto*wrow=w+size_t(first)*32;auto*qvr=qv+size_t(first)*32;
        guarded_bf16_scores<<<grid,512,sizeof(ScoreTile),stream>>>(qrow,keys,wrow,0,rows,pos+first,ratio,nullptr,0,scores,t,qvr,kv);check_launch("S8 exact BF16 fallback scores");
        pure_scores<<<grid,512,sizeof(PureTile),stream>>>(wrow,qi+size_t(first)*32*128,ki,qs+size_t(first)*32,ks,qvr,kv,rows,pos+first,ratio,scores,t);check_launch("S8 pure scores");
        select_rows<false><<<rows,kSelectThreads,0,stream>>>(scores,t,first,pos,ratio,k,offset,blocks,block,out,nullptr,cand_stride);check_launch("S8 unchanged topk");
        if(block_ids){select_rows<true,true><<<rows,kSelectThreads,0,stream>>>(scores,t,first,pos,ratio,k,offset,blocks,block,out,nullptr,0,block_ids);check_launch("S8 unchanged candidate blocks");}
    }
    s8_dispatch_r1.fetch_add(1,std::memory_order_relaxed);return true;
}
extern "C" int ds41_s8_indexer_info_r1(uint64_t*out,int n){
    if(!out||n!=3)return -1;out[0]=s8_dispatch_r1.load(std::memory_order_relaxed);
    auto e=cudaMemcpyFromSymbol(out+1,s8_first_tile_r1,sizeof(uint64_t));if(e!=cudaSuccess)return int(e);
    e=cudaMemcpyFromSymbol(out+2,bf16_first_tile_r1,sizeof(uint64_t));return int(e);
}
}

#include <cstdio>

extern "C" size_t tile_workspace(int m,int64_t t){return strata::ds41::kernels::indexer_topk_prefill_workspace_bytes(m,t);}
extern "C" int tile_top(const void*q,const void*k,const void*w,int m,int pos,int ratio,int mode,const void*mask,
    void*indices,void*blocks,void*ws,size_t bytes){
    try{
        using namespace strata::ds41::kernels;
        if(mode==1)indexer_topk_prefill_blocks((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)w,m,pos,ratio,512,2048,8,(int32_t*)indices,(int32_t*)blocks,ws,bytes,0);
        else indexer_topk_prefill((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)w,m,pos,ratio,(const uint8_t*)mask,nullptr,(pos+m)/ratio,512,0,2048,8,(int32_t*)indices,ws,bytes,0);
        return int(cudaGetLastError());
    }catch(const std::exception&e){fprintf(stderr,"tile_top: %s\n",e.what());return -1;}
}

#include "score_cabi_v334.inc"
