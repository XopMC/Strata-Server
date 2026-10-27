#include "../../../prefill-direct-fp8-port-v531/source/pair_scope_v531.hpp"
#define DS516_CONTEXT 524288
#include "../upload_lease_v534.hpp"
#include "../pass_upload_async_v558.hpp"
#include "../serving_policy_v516.hpp"
#include "../../../index-native-admission-v501/source/index_pair_admission_v501.hpp"
// src/ds41/engine.cu - DeepSeek V4.1 Flash decode, one token at a time. See engine.hpp.
//
// Each block of code below restates one method of DeepSeek's model.py (Block, Attention, Compressor, Indexer,
// MoE, Engram, ParallelHead) for one token. The GPU work is ordered on the default stream. The dense FP8 GEMVs
// (K1), the hyper-connection mixes (K7), the sparse attention (K3), routing (K8) and the indexer top-k (K5) go
// through the task kernels. The routed experts run on a CPU thread that meets the
// stream once per layer through an ExpertDoorbell (upstream's doorbell), so the host thread only enqueues work
// and waits once, for the logits. The engram rows of both engram layers are read at the start of the step.
#include "strata/ds41/engine_causal_batch_v310.hpp"
#include "candidate_cutoff_v369.h"
#include "native_adaptive_lease_v387.hpp"
#include <pthread.h>
#include <sched.h>
#include "strata/ds41/native_main.hpp"
#include "strata/ds41/native_chunk.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/doorbell.hpp"
#include "strata/ds41/engram_rows.hpp"
#include "strata/ds41/expert_stream.hpp"
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/lookahead.hpp"
#include "strata/ds41/fp8_gemv.hpp"
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"
#include "pair_indexer_v1.hpp"
#include "pair_registry_v145.hpp"
#include "pair_inverse_v141.h"
#include "strata/ds41/kernels/k15_hc_prefill.hpp"
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"
#include "strata/ds41/kernels/k3_sparse_attn.hpp"
#include "strata/ds41/kernels/k3_causal_batch_v309.hpp"
#include "strata/ds41/kernels/k5_indexer.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/ops.hpp"
#include "attention_woa_1017.hpp"
#include "strata/ds41/prefill_ops.hpp"
#include "strata/ds41/vram_experts.hpp"
#include "strata/ds41/wo_a_fp8_stream_dense_v191.hpp"
#include "strata/ds41/native_experts.hpp"
#include "strata/ds41/native_gpu_experts_compact_v252.hpp"
#include "strata/ds41/native_upload_pipeline.hpp"
#include "strata/ds41/native_stage_copy_pool_v2.hpp"
#include "strata/ds41/native_dense_prefill.hpp"
#include "strata/ds41/native_fp8_experts_compact_v252.hpp"
#include "strata/ds41/native_prefill.hpp"
#include "strata/ds41/native_routes.hpp"
#include "strata/ds41/native_candidate_pack.hpp"
#include "strata/ds41/native_act_storage.hpp"
#include "strata/ds41/native_host_scatter.hpp"
#include "strata/ds41/native_host_prefill_layout.hpp"
#include "strata/ds41/native_vision_plan.hpp"
#include <fstream>
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/wavefront_credits_v325.hpp"
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/early_boundary_protocol_v337.hpp"
#include <functional>

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <limits>
#include <map>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "strata/ds41/native_bf16_many.hpp"
extern "C" void ds41_recompute_norm_r1(const uint16_t*,const uint16_t*,uint16_t*,float*,int);
extern "C" void ds41_recompute_gather_r1(const uint16_t*,const float*,const float*,const uint16_t*,const int32_t*,int,int,uint16_t*);

namespace strata::ds41 {
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/woa_verify_stats_v449.inc"
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/prefill_pass_stats_v317.inc"
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/wavefront_stats_v325.inc"
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/rank_copy_stats_v331.inc"
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/early_attention_stats_v337.inc"
static std::atomic<uint64_t>eng_parallel_calls_r1{0},eng_parallel_rows_r1{0},eng_parallel_workers_r1{0};
static std::atomic<uint64_t>eng_parallel_nodes_r1[2]{},eng_parallel_cpus_r1[4]{};
static NativeStageCopyPoolV2*stage_copy_pool_instance_v2=nullptr;
static NativeStageCopyPoolV2&stage_copy_pool_v2(){static NativeStageCopyPoolV2 pool(4);stage_copy_pool_instance_v2=&pool;return pool;}
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/wavefront_stage_copy_v325.inc"
extern "C" int ds41_native_stage_copy_info_v2(uint64_t*out,int n){
    if(!out||n!=3)return -1;
    if(wavefront325::stats[1].load()){for(int i=0;i<3;i++)out[i]=stage_copy_counts325[i].load();return 0;}
    out[0]=stage_copy_pool_instance_v2?stage_copy_pool_instance_v2->calls():0;
    out[1]=stage_copy_pool_instance_v2?stage_copy_pool_instance_v2->parallel_calls():0;
    out[2]=stage_copy_pool_instance_v2?stage_copy_pool_instance_v2->bytes():0;
    return 0;
}
#include "../serving_counters_v516.inc"
static void native_stage_copy_v1(uint8_t*dst,const uint8_t*src,size_t bytes,bool parallel){
    if(parallel&&stage_copy_rank_mode325){auto&pool=stage_copy_rank_pool325();const auto before=pool.parallel_calls();pool.copy(dst,src,bytes);stage_copy_counts325[0].fetch_add(1);stage_copy_counts325[1].fetch_add(pool.parallel_calls()-before);stage_copy_counts325[2].fetch_add(bytes);}
    else if(parallel)stage_copy_pool_v2().copy(dst,src,bytes);else std::memcpy(dst,src,bytes);
}
// ON-only observer. OFF callers retain the complete original325 function above.
static void native_stage_copy_v331(uint8_t*dst,const uint8_t*src,size_t bytes,bool parallel){
    if(!parallel||!stage_copy_rank_mode325)throw std::logic_error("331 requires actual private325 rank-copy path");
    struct Observe{Observe(){copy331::observing=true;}~Observe(){copy331::observing=false;}}observe;
    NativeStageCopyPoolV2*ptr=nullptr;try{ptr=&stage_copy_rank_pool325();}catch(...){copy331::stats[12].fetch_add(1,std::memory_order_relaxed);throw;}
    auto&pool=*ptr;const int rank=copy331::actual_rank;
    if(rank<0||rank>1){copy331::stats[12].fetch_add(1,std::memory_order_relaxed);throw std::logic_error("331 private pool rank not witnessed");}
    copy331::stats[13+rank].fetch_add(1,std::memory_order_relaxed);
    try{const auto before=pool.parallel_calls();pool.copy(dst,src,bytes);const auto parallel_jobs=pool.parallel_calls()-before;
        stage_copy_counts325[0].fetch_add(1);stage_copy_counts325[1].fetch_add(parallel_jobs);stage_copy_counts325[2].fetch_add(bytes);
        copy331::stats[4+rank].fetch_add(1,std::memory_order_relaxed);copy331::stats[6+rank].fetch_add(parallel_jobs,std::memory_order_relaxed);copy331::stats[8+rank].fetch_add(bytes,std::memory_order_relaxed);
    }catch(...){copy331::stats[10+rank].fetch_add(1,std::memory_order_relaxed);copy331::stats[13+rank].fetch_sub(1,std::memory_order_release);throw;}
    copy331::stats[13+rank].fetch_sub(1,std::memory_order_release);
}
extern "C" int ds41_engram_parallel_info_r1(uint64_t*out,int n){
    if(!out||n!=9)return -1;out[0]=eng_parallel_calls_r1.load();out[1]=eng_parallel_rows_r1.load();out[2]=eng_parallel_workers_r1.load();
    for(int i=0;i<2;i++)out[3+i]=eng_parallel_nodes_r1[i].load();for(int i=0;i<4;i++)out[5+i]=eng_parallel_cpus_r1[i].load();return 0;
}


using ops::bf16;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 engine: ") + what + ": " + cudaGetErrorString(e));
}

template <typename T>
T* dalloc(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    try { ck(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset"); }
    catch (...) { cudaFree(p); throw; }
    return (T*) p;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// precompute_freqs_cis in fp32, as model.py computes it: [seqlen][32] (cos, sin) pairs
std::vector<float> rope_table(int seqlen, bool yarn) {
    const int dim = kRopeDim;
    const double base = yarn ? kCompressRopeTheta : kRopeTheta;
    std::vector<float> freqs(dim / 2);
    for (int i = 0; i < dim / 2; ++i) freqs[i] = 1.0f / std::pow((float) base, (float) (2 * i) / (float) dim);
    if (yarn) {
        auto corrected = [&](double rot) {
            return dim * std::log(kOriginalSeqLen / (rot * 2 * M_PI)) / (2 * std::log(base));
        };
        const int low = std::max((int) std::floor(corrected(kBetaFast)), 0);
        const int high = std::min((int) std::ceil(corrected(kBetaSlow)), dim - 1);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / std::max((float) (high - low), 1e-3f);
            ramp = std::min(std::max(ramp, 0.0f), 1.0f);
            const float smooth = 1.0f - ramp;
            freqs[i] = freqs[i] / (float) kRopeFactor * (1.0f - smooth) + freqs[i] * smooth;
        }
    }
    std::vector<float> t((size_t) seqlen * dim);
    for (int p = 0; p < seqlen; ++p)
        for (int i = 0; i < dim / 2; ++i) {
            const float a = (float) p * freqs[i];
            t[(size_t) p * dim + 2 * i] = std::cos(a);
            t[(size_t) p * dim + 2 * i + 1] = std::sin(a);
        }
    return t;
}

}  // namespace

struct Engine::Impl {
    serving516::Lifetime serving_lifetime516;
    Pack pack;
    EngineOptions opt;
    int max_seq;
    int cpu_threads;
    NativeEngineOptions native_options;
    Engine*outer325=nullptr;
    bool wavefront_latched325=wavefront325::enabled();
    adaptive528::Plan plan528=serving516::snapshot528(max_seq);
    const geometry335::Fixed fixed335=plan528.armed&&(plan528.mode==2||plan528.mode==3)?plan528.g:geometry335::Original512;

    std::unique_ptr<fast534::UploadLease> upload_lease534;
    const bool pass_upload_enabled558=pass558::enabled();
    bool mode4_534()const noexcept{return plan528.armed&&plan528.mode==4;}
    void close_upload534(){if(upload_lease534){upload_lease534->close();upload_lease534.reset();}}
    void append_plan534(int N,int pos,bool loss){
        if(!pos)return;
        if(!serving516::enabled()||!plan528.armed||headless320||verify_pending||active_pass317||upload_lease534||pf.own_scratch||pf.own_ring)
            throw std::logic_error("534 append requires idle ordered headful owner");
        auto candidate=adaptive528::make(max_seq,N,pos,(native_images?adaptive528::Images:0)|(loss?adaptive528::Nll:0));
        if(candidate.mode!=4)throw std::logic_error("534 append must select exact one-span strategy");
        std::lock_guard<std::mutex>lock(serving516::registry);
        if(serving516::engines.load()!=1||serving516::active.load()||prefill317_detail::stats[5].load()||wavefront325::active.load()||fast534::words[18].load()||fast534::words[17].load()||fast534::words[21].load())
            throw std::logic_error("534 append has live/failed request resources");
        candidate.epoch=++serving516::epoch;plan528=candidate;serving516::prepared528=candidate;
        opt.prefill_chunk=candidate.g.cap;opt.prefill_batch=candidate.g.B;native_host_expert_batch=candidate.g.E;
    }

    bool parallel_headless_copy331=copy331::enabled();
    bool early_attention337=early337::enabled();
    static bool latch_candidate375(){
        const char*v=std::getenv("DS1017_CANDIDATE_CUTOFF_V375");
        if(!v)return false;
        if(std::strcmp(v,"0")&&std::strcmp(v,"1"))
            throw std::invalid_argument("375 candidate switch requires explicit0/1");
        return !std::strcmp(v,"1");
    }
    const bool candidate_cutoff375=latch_candidate375();
    bool adaptive_terminal387=false;
    uint8_t*candidate_workspace375=nullptr;
    size_t candidate_workspace_bytes375=0;

    uint64_t wavefront_committed_end325=0;
    bool headless320=false,headless_closed320=false,headless_pf320=false,headless_ack320=false;
    int headless_device320=-1,headless_imported_layer320=-1,headless_observed_first320=0;
    bool headless_frequency_merged320=false;
    HeadlessOptionsV320 headless_options320{};
    uint64_t headless_startup_plan320=0,headless_own_bytes320=0;
    std::thread::id headless_issuer320;uint64_t headless_import_epoch320=0;
    std::vector<int>headless_tokens320;std::vector<uint16_t>headless_final320;
    PrefillTiming headless_timing320{};cudaEvent_t headless_import_event320=nullptr;
    std::shared_ptr<const HeadlessPinnedPrefixV320>headless_import_owner320;
    bool native_math320()const noexcept{return bool(native_cpu)||(headless320&&bool(pack.native()));}
    void headless_current320()const{
        int d=-1;ck(cudaGetDevice(&d),"320 current headless device");
        if(!headless320||d!=1||d!=headless_device320||headless_issuer320!=std::this_thread::get_id())
            throw std::logic_error("320 requires owning rank1 issuer/current device");
    }
    static void headless_guard_device320(){
        ds501::require_environment();
        int d=-1;ck(cudaGetDevice(&d),"320 constructor current device");
        if(d!=1)throw std::invalid_argument("320 headless role requires logical rank1, no device hop");
        ds501::validate_device(1);
    }
    struct PrefillPassState317;PrefillPassState317*active_pass317=nullptr;
    std::unique_ptr<PrefillPassState317>headless_state320;
    bool pass_seam317=prefill317_detail::enabled();uint64_t pass_epoch317=0;
    bool compact_native_prefill_inputs=false;
    bool host_native_prefill=false;
    bool gpu_native_prefill=false,alias_native_hc=false;
    bool decoded_native_expert_cache=false;
    bool direct_fp8_expert_weights=false;uint64_t direct_fp8_projection_encodes=0;
    bool output16_compact_v252=false;
    bool paired_layout145=false; // typed metadata only; old Output16 enum/class ABI stays unchanged.
    uint64_t output16_uploads_v252=0,output16_bytes_v252=0,output16_encodes_v252=0;
    uint64_t output16_capacity_v252=0,output16_max_triple_v252=0,output16_raw_uploads_v252=0;
    int output16_last_cap_v252=0,output16_last_sub_v252=0;

    // Decoder layers own only window state; all global KV/index keys come from encoder20.
    // A conservative4096-token suffix covers128 retained rows +19*127 causal halo.
    bool decoder_suffix_prefill=false;int native_first_prefill_row=0;
    bool recompute_inputs_active_r1=false;
    bool suffix_candidate_storage_r1=false,suffix_candidate_active_r1=false;
    int suffix_candidate_first_r1=0;
    bool compact_candidate_prefill=false;uint64_t compact_candidate_packed_rows=0,compact_candidate_unpacked_rows=0,compact_candidate_saved_peak=0;
    uint64_t decoder_suffix_passes=0,decoder_suffix_layers=0,decoder_suffix_rows_skipped=0;
    bool optimized_native_prefill()const{return host_native_prefill||gpu_native_prefill;}
    size_t native_host_budget=size_t(96)<<30;
    int native_host_expert_batch=4096;
    std::unique_ptr<NativeExperts> native_cpu;
    std::shared_ptr<NativeGpuExperts>native_gpu_experts;
    std::shared_ptr<NativeExpertCachePool>native_gpu_pool;
    bool native_gpu_reused=false;
    size_t native_gpu_budget=0;size_t native_gu_budget_v19=0;
    std::vector<std::array<uint64_t,384>>native_expert_frequency=std::vector<std::array<uint64_t,384>>(40);
    int native_cache_tail_rows=0; // Frozen at Engine creation;0 keeps the original raw counts.
    bool native_cache_tail_active=false;
    int native_cache_tail_begin=0,native_cache_tail_end=0;
    uint64_t native_cache_tail_counted_uses=0;
    int native_prefill_margin_mib=1024; // Frozen at native Engine creation.
    uint64_t native_prefill_minimum_free_bytes=std::numeric_limits<uint64_t>::max();
    uint64_t native_prefill_sampled_passes=0;
    std::atomic<bool> native_failed{false};
    std::vector<std::shared_ptr<void>> owned_device;
    template<class T>T* own_alloc(size_t n){
        T*p=dalloc<T>(n);std::shared_ptr<void>owner(p,[](void*q){cudaFree(q);});
        owned_device.push_back(std::move(owner));if(headless320)headless_own_bytes320+=uint64_t(n)*sizeof(T);return p;
    }
    size_t engram_row_storage()const{return pack.native()?512:264;}
    std::unique_ptr<NativeImagePlan>native_images;
    bf16*native_image_features=nullptr;
    int native_prefill_position=0;
    void native_image_embed(bf16*hidden,int first,int count){
        if(!native_images)return;
        for(auto run:native_images->runs(first,count))if(run.image)for(int hc=0;hc<kHc;hc++)
            ck(cudaMemcpy2DAsync(hidden+size_t(run.first)*kHc*kDim+hc*kDim,kHc*kDim*sizeof(bf16),
                native_image_features+size_t(run.feature_first)*kDim,kDim*sizeof(bf16),kDim*sizeof(bf16),run.count,cudaMemcpyDeviceToDevice,0),"native image feature rows");
    }
    bool native_capture_main=false;
    uint16_t* native_main_ring=nullptr;
    uint16_t* native_main_step=nullptr;

    struct Fp8 { const uint8_t* w; const uint8_t* s; int64_t n, k; };
    struct Layer {
        Fp8 wq_a, wq_b, wkv, wo_b, sh_w1, sh_w2, sh_w3, idx_wq_b, eng_wkv;
        Fp8 wo_a8{};   // wo_a as FP8 with block scales (packs from 2026-10-06); wo_a below is then null
        const bf16 *q_norm, *kv_norm, *wo_a, *attn_norm, *ffn_norm, *gate_w;
        const float *sink, *gate_bias_vl=nullptr, *gate_bias, *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base,
            *hc_ffn_scale;
        const bf16 *c_wkv = nullptr, *c_wgate = nullptr, *c_norm = nullptr;       // compressor (kv sources)
        const bf16 *idx_wp = nullptr, *idx_wk = nullptr, *idx_knorm = nullptr;     // indexer
        const bf16 *eng_qw = nullptr, *eng_kw = nullptr;                          // engram
        int ratio = 0;
        // state
        bf16* window = nullptr;                    // [128][512]
        bf16* comp = nullptr;                      // [max_seq/ratio][512]   (kv sources)
        bf16* idx_keys = nullptr;                  // [max_seq/ratio][128]   (kv sources)
        float *kv_state = nullptr, *score_state = nullptr;   // [ratio][512]   (kv sources, ratio > 1)
        int64_t moe_handle = -1;
    };
    std::vector<Layer> L;
    const bf16 *embed, *head, *final_norm;

    // rope tables on the device: [max_seq][64] floats
    float* rope_plain = nullptr;
    float* rope_yarn = nullptr;

    // engram: rows of both engram layers for the current step, read before the step's GPU work
    std::unique_ptr<EngramRows> eng_rows;   // O_DIRECT reads (upstream DirectFile): the rows bypass the file cache
    int n_eng = 0;                     // engram tables
    std::vector<std::vector<int64_t>> eng_ids;   // [table][kEngRows] the rows of the current step
    std::vector<int32_t> history;      // compressed token ids fed so far
    static constexpr int kEngRows = 24;
    uint8_t* eng_host = nullptr;       // pinned: per engram layer, kEngRows*256 weight bytes then kEngRows*8 scales
    uint8_t* eng_dev = nullptr;        // the same on the device

    // routed experts: the CPU thread and its doorbell (round l+1 = layer l)
    std::unique_ptr<ExpertDoorbell> db;
    std::thread worker;
    std::mutex mu;
    std::condition_variable cv;
    uint64_t go = 0;                   // steps released to the worker
    std::atomic<bool> stop{false};
    std::atomic<int64_t> worker_us{0}; // CPU expert time of the current step
    std::atomic<int> worker_misses{0}; // routed uses outside VRAM (CPU plus zero-copy)
    std::unique_ptr<VramExperts> vram;  // the VRAM tier (null: none)
    std::unique_ptr<HostExperts> host;  // the RAM tier (null: none); the rest is read from the mapped file
    std::atomic<int> worker_ram{0}, worker_file{0}, worker_ssd{0};   // CPU experts of the step by tier
    std::vector<unsigned char> mincore_buf;
    std::unique_ptr<RouterLookahead> lookahead;   // warms the next layer's file-tier experts (DS41_LOOKAHEAD=0: off)
    bool fetch_now = true;             // ask for a layer's missing file pages before computing (DS41_FETCH_NOW=0: off)
    int32_t* gpu_sel = nullptr;        // [6] per-call descriptor indices (-1: CPU)
    std::shared_ptr<int> zc_quota;      // [layers] device values; update only between steps
    std::shared_ptr<void> zc_workspace; // used when there is no VRAM tier
    std::unique_ptr<ExpertStaging> zc_stage;
    std::shared_ptr<ExpertBlob> zc_blobs; // immutable [layers][experts] pack metadata
    int32_t* routes_dev = nullptr;     // [40][6]
    float* weights_dev = nullptr;      // [40][6]
    uint8_t* cand_dev = nullptr;       // [max_seq] candidate mask of the candidate layer

    // scratch
    bf16 *h, *h2, *xa, *xf, *qr, *q, *kvv, *o, *oa, *attn_out, *latent, *ik, *iq, *iw_raw, *iw, *g, *u, *sh_h,
        *sh_out, *ffn_out, *eng_vals, *eng_kv, *final_x;
    float *act, *pre_mix, *pre, *post, *comb, *ffn_pre, *attn_pre, *attn_post, *attn_comb, *ffn_post,
        *ffn_comb, *ckv, *cscore, *scores, *routed, *logits;
    uint16_t* x_half_dev;
    int32_t* idx_dev;                  // attention index list: [0, 128) window, then the compressed top-k

    // DS41_DEBUG=<file>: per step, the intermediates of layers 1 and 2 (bf16 bits), for bisecting a mismatch
    std::FILE* dbg = nullptr;
    bool native_block_debug = false;
    void dbg_write(const bf16* dev, int n) {
        if (!dbg) return;
        ck(cudaStreamSynchronize(st), "debug dump");
        std::vector<uint16_t> b(n);
        ck(cudaMemcpy(b.data(), dev, (size_t) n * 2, cudaMemcpyDeviceToHost), "debug dump");
        if(std::fwrite(b.data(),2,n,dbg)!=size_t(n))throw std::runtime_error("debug probe write failed");
    }

    // decode on its own stream, captured as CUDA graphs (upstream session_capture_token / Verifier): the position
    // dependent values come from fixed pinned staging, copied to the device by the graph's first node
    struct StepParams { int token, pos, t1, t2; };   // t1, t2: compressed lengths at ratio 1 and 2
    cudaStream_t st = nullptr;
    StepParams* hp = nullptr;          // pinned staging of the next replay
    int* dp = nullptr;                 // device copy: dp[0] token, dp[1] pos, dp[2] t1, dp[3] t2
    int* d_next = nullptr;             // device argmax of the logits
    int* hp_next = nullptr;            // pinned: the argmax, the logits and the routes, after the step
    float* lg_pinned = nullptr;
    int32_t* routes_pinned = nullptr;
    float* one_hot_dev = nullptr;      // {1, 0, 0, 0}: pre_mix at the first layer
    NativeDensePrefill* active_dense_prefill = nullptr;
    NativeFp8Experts* active_fp8_experts = nullptr;
    int native_coordinator_cpu = -1;
    bool use_graph = true;             // DS41_GRAPH=0: eager launches (also with a dump or DS41_DEBUG)
    int parity = 0;                    // pos % 2 of the step being enqueued (the ratio-2 compressor's slot)
    int64_t tcap1 = 1, tcap2 = 1;      // indexer capacities of the graph being enqueued
    std::map<uint64_t, cudaGraphExec_t> graphs;
    int graph_captures = 0;

    struct VerifyWorkspace;
    std::shared_ptr<VerifyWorkspace> verify_ws;
    bool verify_pending = false, verify_failed = false;
    void enqueue_verify(int m);
    VerifyResult verify(const std::vector<int>& window, int pos, bool logits, Timing& tm);
    void commit_verify(int n_keep);

    // shared attention state for the current token (SharedAttentionRuntime)
    const bf16* cur_comp = nullptr;
    const bf16* cur_index_k = nullptr;
    std::vector<float> lg;             // logits of the last step

    Impl(const std::string& dir, const EngineOptions& o)
        : pack(dir), opt(o), max_seq(o.max_seq), cpu_threads(o.cpu_threads) {}
    Impl(std::shared_ptr<NativeCheckpoint>checkpoint,EngramHash hash,const EngineOptions&o,const NativeEngineOptions&native)
        :pack(std::move(checkpoint),std::move(hash)),opt(o),max_seq(o.max_seq),cpu_threads(o.cpu_threads),native_options(native){
        if(o.max_seq<1||o.max_seq>1048576||o.prefill_chunk!=0||!o.expert_profile.empty()||native.cpus.empty())
            throw std::invalid_argument("native bridge currently requires bounded context, token-wise prefill, explicit CPUs and no EXL3 profile");
        native_cpu=std::make_unique<NativeExperts>(pack.native()->weights(),native.task_library,native.projection_library,native.a8,native.cpus,native.cpus.front(),native.mixed_f32);
    }
    Impl(std::shared_ptr<NativeCheckpoint>checkpoint,EngramHash hash,const EngineOptions&o,const NativeEngineOptions&native,
         EngineRoleV320 role,const HeadlessOptionsV320&limits)
        :pack(std::move(checkpoint),std::move(hash)),opt(o),max_seq(o.max_seq),cpu_threads(o.cpu_threads),native_options(native){
        if(role==EngineRoleV320::Headful){
            if(o.max_seq<1||o.max_seq>1048576||o.prefill_chunk!=0||!o.expert_profile.empty()||native.cpus.empty())
                throw std::invalid_argument("native bridge currently requires bounded context, token-wise prefill, explicit CPUs and no EXL3 profile");
            native_cpu=std::make_unique<NativeExperts>(pack.native()->weights(),native.task_library,native.projection_library,native.a8,native.cpus,native.cpus.front(),native.mixed_f32);return;
        }
        if(mode4_534())throw std::invalid_argument("534 one-span strategy has no headless helper");
        const char*enabled=std::getenv("DS1017_HEADLESS_PREFILL_V320");
        if(role!=EngineRoleV320::HeadlessPrefillRank1||!enabled||std::strcmp(enabled,"1")||!pack.native()||
           max_seq!=524288||o.prefill_chunk||!o.expert_profile.empty()||!native.cpus.empty()||
           !native.task_library.empty()||!native.projection_library.empty()||native.graph||
           limits.max_device_bytes<(uint64_t(2)<<30)||limits.max_device_bytes>(uint64_t(20)<<30)||
           limits.reserve_bytes<(uint64_t(512)<<20)||limits.reserve_bytes>(uint64_t(2)<<30))
            throw std::invalid_argument("320 requires explicit512 PP-only role/no CPU libraries/no CPUs/no graphs/2..20GiB budget");
        for(const char*name:{"DS41_NATIVE_PREFILL_STREAM_DENSE","DS41_MAPPED_EMBEDDING_R1","DS41_NATIVE_FP8_EXPERTS","DS41_NATIVE_DENSE_PREFILL","DS41_NATIVE_PACKED_FP8_WEIGHTS","DS41_NATIVE_OUTPUT16_COMPACT_V252"}){
            const char*v=std::getenv(name);if(!v||std::strcmp(v,"1"))throw std::invalid_argument(std::string("320 requires ")+name+"=1");
        }
        const char*stage=std::getenv("DEEPSEEK41_GPU_STAGE");if(stage&&std::strcmp(stage,"0"))throw std::invalid_argument("320 forbids staged generation tiers");
        headless_guard_device320();
        const auto typed=pair145::info(*pack.native()->weights());
        const auto old=pack.native()->weights()->output16_info_v248();
        if(typed[2]!=46080||typed[3]!=305764761600ull||!typed[4]||!typed[5]||typed[6]!=1||old[1])
            throw std::invalid_argument("320 requires SAME-DSO full typed139 registry; no raw/EXL3/cross-DSO fallback");
        headless320=true;headless_device320=1;headless_issuer320=std::this_thread::get_id();headless_options320=limits;
        gpu_native_prefill=true;alias_native_hc=true;compact_native_prefill_inputs=true;decoded_native_expert_cache=true;
        opt.prefill_chunk=fixed335.cap;opt.prefill_batch=fixed335.B;native_host_expert_batch=fixed335.E;pass_seam317=true;use_graph=false;
    }
    ~Impl() {
        close_upload534();
        if(headless320){try{headless_close_impl320();}catch(...){std::terminate();}}

        if(active_pass317)std::terminate(); // owned pass must join/close before Engine
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
        if(native_cpu&&ds387::lookup(*native_cpu)){
            // Exceptional destruction: Task worker is joined first; no callback
            // can race this owned target-stream drain or hook detach below.
            if(st&&cudaStreamSynchronize(st)!=cudaSuccess)std::terminate();
            ds387::cleanup(native_cpu.get());
        }

        native_gpu_experts.reset();
        native_cpu.reset();
        // the lookahead calls into both tiers, and the VRAM tier's copy thread writes into RAM slots: stop them first
        lookahead.reset();
        vram.reset();
        host.reset();
        for (auto& [k, e] : graphs) cudaGraphExecDestroy(e);
        if (st) cudaStreamSynchronize(st);
        prefill_end_quiet();
        verify_ws.reset();
        for(auto& layer:L)if(layer.moe_handle>=0)exl3_moe_cpu_free_layer(layer.moe_handle);
        for (cudaEvent_t e : pfp.ev) cudaEventDestroy(e);
        if (eng_host) cudaFreeHost(eng_host);
        if (hp) cudaFreeHost(hp);
        if (hp_next) cudaFreeHost(hp_next);
        if (lg_pinned) cudaFreeHost(lg_pinned);
        if (routes_pinned) cudaFreeHost(routes_pinned);
        if (st) cudaStreamDestroy(st);
        if (pfh.base) cudaFreeHost(pfh.base);
        if(dbg)std::fclose(dbg);
    }

    // a failure after the state changed (history pushed, a step released to the CPU worker, a prefill pass started)
    // leaves the KV caches and the history out of step: every later call is refused
    bool broken = false;
    bool pf_started = false;           // the running prefill has begun changing the state
    std::exception_ptr worker_error;   // set by the CPU worker; rethrown by the step that ran into it
    void usable() const {
        if(adaptive_terminal387)throw std::runtime_error("387 terminal adaptive close; destroy this Engine");
        if (broken) throw std::runtime_error("ds41 engine: unusable after an earlier failure; create a new Engine");
    }
    static void check_token(int t) {
        if (t < 0 || t >= kVocab) throw std::invalid_argument("ds41 engine: token id " + std::to_string(t) + " is outside [0, " + std::to_string(kVocab) + ")");
    }
    /// DS41_TEST_FAULT=<name>: throw at that point once, then the variable is cleared (tests of the failure paths)
    static bool fault(const char* name) {
        const char* v = std::getenv("DS41_TEST_FAULT");
        if (!v || std::strcmp(v, name) != 0) return false;
        unsetenv("DS41_TEST_FAULT");
        return true;
    }

    Fp8 fp8(const std::string& name) {
        const auto& w = pack.dense(name + ".weight");
        const auto& s = pack.dense(name + ".scale");
        if (w.dtype != DType::F8E4M3 || s.dtype != DType::E8M0) throw std::runtime_error(name + " is not FP8 + E8M0");
        return {(const uint8_t*) w.device, (const uint8_t*) s.device, w.shape[0], w.shape[1]};
    }
    const bf16* bf(const std::string& name) {
        const auto& t = pack.dense(name);
        if (t.dtype != DType::BF16) throw std::runtime_error(name + " is not BF16");
        return (const bf16*) t.device;
    }
    const float* f32(const std::string& name) {
        const auto& t = pack.dense(name);
        if (t.dtype != DType::F32) throw std::runtime_error(name + " is not F32");
        return (const float*) t.device;
    }

    //191: only immutable dense bindings; KV/window/carry and CPU weights stay owned by init.
    // Called for the activated layer, and once for every layer after full residency returns.
    void bind_layer_weights_v191(int l) {
        auto& y = L.at(size_t(l));
        const std::string p = "layers." + std::to_string(l) + ".";
        y.wq_a = fp8(p + "attn.wq_a");
        y.wq_b = fp8(p + "attn.wq_b");
        y.wkv = fp8(p + "attn.wkv");
        y.wo_b = fp8(p + "attn.wo_b");
        y.q_norm = bf(p + "attn.q_norm.weight");
        y.kv_norm = bf(p + "attn.kv_norm.weight");
        if (pack.has_dense(p + "attn.wo_a.scale")) y.wo_a8 = fp8(p + "attn.wo_a");
        else y.wo_a = bf(p + "attn.wo_a.weight");
        y.sink = f32(p + "attn.attn_sink");
        y.attn_norm = bf(p + "attn_norm.weight");
        y.ffn_norm = bf(p + "ffn_norm.weight");
        y.hc_attn_fn = f32(p + "hc_attn_fn");
        y.hc_attn_base = f32(p + "hc_attn_base");
        y.hc_attn_scale = f32(p + "hc_attn_scale");
        y.hc_ffn_fn = f32(p + "hc_ffn_fn");
        y.hc_ffn_base = f32(p + "hc_ffn_base");
        y.hc_ffn_scale = f32(p + "hc_ffn_scale");
        y.gate_w = bf(p + "ffn.gate.weight");
        y.gate_bias = f32(p + "ffn.gate.bias");
        y.gate_bias_vl=pack.native()?f32(p+"ffn.gate.bias_vl"):y.gate_bias;
        y.sh_w1 = fp8(p + "ffn.shared_experts.w1");
        y.sh_w2 = fp8(p + "ffn.shared_experts.w2");
        y.sh_w3 = fp8(p + "ffn.shared_experts.w3");
        if (is_kv_source(l)) {
            y.c_wkv = bf(p + "attn.compressor.wkv.weight");
            y.c_norm = bf(p + "attn.compressor.norm.weight");
            if (y.ratio > 1) y.c_wgate = bf(p + "attn.compressor.wgate.weight");
            y.idx_wk = bf(p + "attn.indexer.wk.weight");
            y.idx_knorm = bf(p + "attn.indexer.k_norm.weight");
        }
        if (is_index_source(l)) {
            y.idx_wq_b = fp8(p + "attn.indexer.wq_b");
            y.idx_wp = bf(p + "attn.indexer.weights_proj.weight");
        }
        if (is_engram_layer(l)) {
            y.eng_wkv = fp8(p + "engram.wkv");
            y.eng_qw = bf(p + "engram.q_weight");
            y.eng_kw = bf(p + "engram.k_weight");
        }
    }
    void check_stable_dense_v191() {
        if (embed != bf("embed.weight") || (!headless320 && head != bf("head.weight")) || (headless320 && head) || final_norm != bf("norm.weight"))
            throw std::logic_error("191 streamed dense moved a stable embed/head/norm pointer");
        for (int l = 0; l < kLayers; ++l) if (is_engram_layer(l)) {
            const std::string p = "layers." + std::to_string(l) + ".engram.";
            const auto value = fp8(p + "wkv");
            const auto& y = L[l];
            if (y.eng_wkv.w != value.w || y.eng_wkv.s != value.s ||
                y.eng_qw != bf(p + "q_weight") || y.eng_kw != bf(p + "k_weight"))
                throw std::logic_error("191 streamed dense moved a stable Engram pointer");
        }
    }
    void begin_streamed_dense_v191(int pos) {
        // No graph may retain the old layer allocation. CPU shape preparation and
        // main-ring allocation are permitted: neither creates a target graph.
        if (!native_math320() || !gpu_native_prefill || host_native_prefill || pos != 0 ||
            !history.empty() || verify_pending || verify_ws || !graphs.empty() ||
            graph_captures || native_gpu_experts || native_gpu_pool ||
            active_dense_prefill || active_fp8_experts || pf.own_scratch || pf.own_ring ||
            pack.dense_stream_active_v190() || pack.dense_stream_info_v190()[3])
            throw std::logic_error("191 dense streaming requires a fresh graph-free GPU-native prefill");
        // Reserve both banks before native_prefill_begin samples freeVRAM and sizes scratch.
        pack.begin_dense_stream_v190();
        check_stable_dense_v191();
    }
    void restore_streamed_dense_v191() {
        // prefill_pass's local dense/FP8 objects are already destroyed, and the
        // caller has synchronized/freed pf scratch. Never restore underneath them.
        if (active_dense_prefill || active_fp8_experts || pf.own_scratch || pf.own_ring)
            throw std::logic_error("191 dense restore still has live prefill consumers");
        pack.end_dense_stream_v190();
        check_stable_dense_v191();
        for (int l = 0; l < kLayers; ++l) bind_layer_weights_v191(l);
        const auto info = pack.dense_stream_info_v190();
        if (info[2] || info[8] || info[9] || info[3] != 1 || info[4] != 1 || info[23])
            throw std::logic_error("191 dense restore did not close every bank and complete exactly once");
    }

    void init() {
        if(ds387::enabled()&&!headless320&&max_seq!=524288)throw std::invalid_argument("387 firstqualifiedexperiment is native512only");
        // Current typed profiles are powers of two: cap_of(pos+M,max_seq+1)
        // cannot exceed max_seq under the original committed-position guards.
        if(candidate_cutoff375&&!headless320&&
           (!native_cpu||(max_seq!=262144&&max_seq!=524288)||kCandidateBlock!=8||kCandidateBlocks!=2048))
            throw std::invalid_argument("375 requires native256/512, literal block8/top2048");

        attention_woa_1017::require_opt_in(); // PRIVATE: no1017 engine inference before Root proof
        if(native_math320())if(const char*mode=std::getenv("DS41_NATIVE_PREFILL_MARGIN_MIB")){
            if(!std::strcmp(mode,"512"))native_prefill_margin_mib=512;
            else if(!std::strcmp(mode,"768"))native_prefill_margin_mib=768;
            else if(!std::strcmp(mode,"1024"))native_prefill_margin_mib=1024;
            else if(!std::strcmp(mode,"1536"))native_prefill_margin_mib=1536;
            else throw std::invalid_argument("DS41_NATIVE_PREFILL_MARGIN_MIB must be 512, 768, 1024 or 1536");
        }
        if(native_math320())if(const char*mode=std::getenv("DS41_NATIVE_CACHE_TAIL_ROWS")){
            if(!std::strcmp(mode,"0"))native_cache_tail_rows=0;
            else if(!std::strcmp(mode,"512"))native_cache_tail_rows=512;
            else if(!std::strcmp(mode,"1024"))native_cache_tail_rows=1024;
            else if(!std::strcmp(mode,"2048"))native_cache_tail_rows=2048;
            else throw std::invalid_argument("DS41_NATIVE_CACHE_TAIL_ROWS must be0,512,1024 or2048");
        }
        bool dense_stream_v191 = false;
        if (const char* mode = std::getenv("DS41_NATIVE_PREFILL_STREAM_DENSE")) {
            if (std::strcmp(mode,"0") && std::strcmp(mode,"1"))
                throw std::invalid_argument("DS41_NATIVE_PREFILL_STREAM_DENSE must be explicit0 or1");
            dense_stream_v191 = !std::strcmp(mode,"1");
        }
        if (dense_stream_v191) {
            const char* stage = std::getenv("DEEPSEEK41_GPU_STAGE");
            const char* woa = std::getenv("DS41_NATIVE_WOA_MANY");
            if (!native_math320() || max_seq > 524288 || (stage && std::strcmp(stage,"0")) ||
                (woa && std::strcmp(woa,"0")))
                throw std::invalid_argument("191 dense streaming requires native GPU0/context<=524288/WOA_MANY0");
        }
        if(headless320){
            pack.configure_headless_dense_v320();
            const auto plan=pack.headless_dense_info_v320();
            uint64_t owned=uint64_t(2)*max_seq*kRopeDim*sizeof(float)+uint64_t(kLayers)*kWindow*kHeadDim*2;
            for(int l=0;l<kLayers;++l)if(is_kv_source(l)){
                const uint64_t ratio=kCompressRatio[l];owned+=(uint64_t(max_seq)/ratio+1)*(kHeadDim+kIndexDim)*2;
                if(ratio>1)owned+=uint64_t(2)*ratio*kHeadDim*sizeof(float);
            }
            // eng_dev is retained for matching native metadata; decode scratch,
            // CPU pool/tasks, doorbell, logits and graphs are NOT allocated.
            owned+=std::max<uint64_t>(uint64_t(pack.engram_tables().size())*kEngRows*engram_row_storage(),1);
            owned+=uint64_t(128)*3*kDim*sizeof(uint16_t); // original PP-only MTP main ring, no step ring
            headless_startup_plan320=plan[3]+plan[4]+owned;
            size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"320 startup free memory");
            if(headless_startup_plan320>headless_options320.max_device_bytes||
               free<headless_options320.reserve_bytes||headless_startup_plan320>free-headless_options320.reserve_bytes)
                throw std::runtime_error("320 startup dense+KV+rope payload exceeds explicit budget/free reserve BEFORE allocations");
        }else pack.configure_dense_stream_v190(dense_stream_v191);
        pack.upload_dense();
        pack.map_experts();
        embed = bf("embed.weight");
        head = headless320?nullptr:bf("head.weight");
        final_norm = bf("norm.weight");
        L.resize(kLayers);
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            y.ratio = kCompressRatio[l];
            bind_layer_weights_v191(l);
            if (is_kv_source(l)) {
                y.comp = own_alloc<bf16>((size_t) (max_seq / y.ratio + 1) * kHeadDim);
                y.idx_keys = own_alloc<bf16>((size_t) (max_seq / y.ratio + 1) * kIndexDim);
                if (y.ratio > 1) {
                    y.kv_state = own_alloc<float>((size_t) y.ratio * kHeadDim);
                    y.score_state = own_alloc<float>((size_t) y.ratio * kHeadDim);
                }
            }
            y.window = own_alloc<bf16>((size_t) kWindow * kHeadDim);
            if(!headless320)register_cpu_experts(l);
        }
        if(const char*mode=std::getenv("DS41_NATIVE_PACKED_FP8_WEIGHTS")){
            if(std::strcmp(mode,"0")&&std::strcmp(mode,"1"))throw std::invalid_argument("Explicit packed FP8 weights0/1 required");
            direct_fp8_expert_weights=!std::strcmp(mode,"1");
            const char*enabled=std::getenv("DS41_NATIVE_FP8_EXPERTS");
            if(direct_fp8_expert_weights&&(!native_math320()||!enabled||std::strcmp(enabled,"1")))throw std::invalid_argument("Direct packed weights need native FP8 experts");
        }
        if(const char*mode=std::getenv("DS41_NATIVE_OUTPUT16_COMPACT_V252")){
            if(std::strcmp(mode,"0")&&std::strcmp(mode,"1"))throw std::invalid_argument("Explicit output16 compact mode0/1 required");
            output16_compact_v252=!std::strcmp(mode,"1");
        }
        if(pack.native()){
            const auto compact=pack.native()->weights()->output16_info_v248();
            const auto paired=pair145::info(*pack.native()->weights());
            paired_layout145=paired[2]!=0;
            if(paired_layout145&&(paired[2]!=46080||paired[3]!=305764761600ull||!paired[4]||!paired[5]||paired[6]!=1||compact[1]||!native_math320()||!direct_fp8_expert_weights||!output16_compact_v252))throw std::invalid_argument("typed139 requires complete unmixed registry and explicit original252 FP8 path");
            if(compact[1]&&!output16_compact_v252)throw std::invalid_argument("Compact owner requires explicit output16 GPU feeding252");
            if(output16_compact_v252&&(!native_math320()||!direct_fp8_expert_weights))throw std::invalid_argument("Compact GPU feeding requires native packed FP8 experts");
        }
        if(const char*mode=std::getenv("DS41_NATIVE_DECODER_SUFFIX")){
            if(std::strcmp(mode,"0")&&std::strcmp(mode,"1"))throw std::invalid_argument("Explicit decoder suffix0/1 required");
            decoder_suffix_prefill=!std::strcmp(mode,"1");
            if(decoder_suffix_prefill&&!native_math320())throw std::invalid_argument("Decoder suffix requires native CPU/GPU path");
        }
        if(const char*mode=std::getenv("DS41_NATIVE_COMPACT_CANDIDATES")){
            if(std::strcmp(mode,"0")&&std::strcmp(mode,"1"))throw std::invalid_argument("Explicit compact candidates0/1 required");
            compact_candidate_prefill=!std::strcmp(mode,"1");
            if(compact_candidate_prefill&&(!native_math320()||max_seq>524288||kCandidateBlock!=8||kCandidateBlocks!=2048))throw std::invalid_argument("Compact candidates require native context<=524288/block8/2048IDs");
        }
        // rope tables
        auto plain = rope_table(max_seq, false), yarn = rope_table(max_seq, true);
        rope_plain = own_alloc<float>(plain.size());
        rope_yarn = own_alloc<float>(yarn.size());
        ck(cudaMemcpy(rope_plain, plain.data(), plain.size() * 4, cudaMemcpyHostToDevice), "rope");
        ck(cudaMemcpy(rope_yarn, yarn.data(), yarn.size() * 4, cudaMemcpyHostToDevice), "rope");
        // engram tables
        {
            std::vector<EngramRows::Table> tabs;
            for (const auto& t : pack.engram_tables()) tabs.push_back({t.path, t.weight_offset, t.scale_offset});
            n_eng = (int) tabs.size();
            if (n_eng&&!pack.native()) eng_rows = std::make_unique<EngramRows>(tabs, kEngRows);
            eng_ids.assign(n_eng, std::vector<int64_t>(kEngRows, 0));
        }
        const size_t eng_bytes = (size_t) n_eng * kEngRows * engram_row_storage();
        ck(cudaHostAlloc((void**) &eng_host, std::max<size_t>(eng_bytes, 1), cudaHostAllocDefault), "engram pinned");
        eng_dev = own_alloc<uint8_t>(std::max<size_t>(eng_bytes, 1));
        if(headless320){
            history.reserve(max_seq);
            native_main_ring=own_alloc<uint16_t>(128*3*kDim);native_capture_main=true;
            size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"320 constructor final free reserve");
            if(free<headless_options320.reserve_bytes)throw std::runtime_error("320 constructor actual free reserve violated");
            return; // PP has private K15/router workspaces; no decoder kernel globals
        }
        // scratch
        h = own_alloc<bf16>(kHc * kDim);
        h2 = own_alloc<bf16>(kHc * kDim);
        xa = own_alloc<bf16>(kDim);
        xf = own_alloc<bf16>(kDim);
        qr = own_alloc<bf16>(kQLora);
        q = own_alloc<bf16>(kHeads * kHeadDim);
        kvv = own_alloc<bf16>(kHeadDim);
        o = own_alloc<bf16>(kHeads * kHeadDim);
        oa = own_alloc<bf16>(kOGroups * kOLora);
        attn_out = own_alloc<bf16>(kDim);
        latent = own_alloc<bf16>(kHeadDim);
        ik = own_alloc<bf16>(kIndexDim);
        iq = own_alloc<bf16>(kIndexHeads * kIndexDim);
        iw_raw = own_alloc<bf16>(kIndexHeads);
        iw = own_alloc<bf16>(kIndexHeads);
        g = own_alloc<bf16>(kMoeInter);
        u = own_alloc<bf16>(kMoeInter);
        sh_h = own_alloc<bf16>(kMoeInter);
        sh_out = own_alloc<bf16>(kDim);
        ffn_out = own_alloc<bf16>(kDim);
        eng_vals = own_alloc<bf16>(24 * 256);
        eng_kv = own_alloc<bf16>((kHc + 1) * kDim);
        final_x = own_alloc<bf16>(kDim);
        act = own_alloc<float>(8192);
        pre_mix = own_alloc<float>(kHc);
        pre = own_alloc<float>(kHc);
        post = own_alloc<float>(kHc);
        comb = own_alloc<float>(kHc * kHc);
        attn_pre = own_alloc<float>(kHc);
        attn_post = own_alloc<float>(kHc);
        attn_comb = own_alloc<float>(kHc * kHc);
        ffn_pre = own_alloc<float>(kHc);
        ffn_post = own_alloc<float>(kHc);
        ffn_comb = own_alloc<float>(kHc * kHc);
        ckv = own_alloc<float>(kHeadDim);
        cscore = own_alloc<float>(kHeadDim);
        scores = own_alloc<float>(max_seq + 1);
        routed = own_alloc<float>(kDim);
        logits = own_alloc<float>(kVocab);
        x_half_dev = own_alloc<uint16_t>(kDim);
        idx_dev = own_alloc<int32_t>(kWindow + kIndexTopK);
        routes_dev = own_alloc<int32_t>(kLayers * kTopK);
        weights_dev = own_alloc<float>(kLayers * kTopK);
        cand_dev = own_alloc<uint8_t>(max_seq + 1);
        if(candidate_cutoff375){
            candidate_workspace_bytes375=ds369_candidate_workspace_bytes(max_seq,kCandidateBlock);
            if(!candidate_workspace_bytes375)throw std::logic_error("375 invalid369 scratch geometry");
            candidate_workspace375=own_alloc<uint8_t>(candidate_workspace_bytes375);
            // Ordinary construction only, before any graph capture/target work.
            // own_alloc joins the existing post-stream-drain device ownership.
            ck(cudaMemset(candidate_workspace375,0,candidate_workspace_bytes375),"375 clear owned candidate workspace");
        }

        history.reserve(max_seq);
        db = std::make_unique<ExpertDoorbell>(1, kTopK, kDim);
        gpu_sel = own_alloc<int32_t>(kTopK);
        // decode stream, graph staging, outputs
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "decode stream");
        ck(cudaHostAlloc((void**) &hp, sizeof(StepParams), cudaHostAllocDefault), "step params");
        ck(cudaHostAlloc((void**) &hp_next, sizeof(int), cudaHostAllocDefault), "next token");
        ck(cudaHostAlloc((void**) &lg_pinned, (size_t) kVocab * 4, cudaHostAllocDefault), "logits");
        ck(cudaHostAlloc((void**) &routes_pinned, kLayers * kTopK * 4, cudaHostAllocDefault), "routes");
        dp = own_alloc<int>(4);
        d_next = own_alloc<int>(1);
        one_hot_dev = own_alloc<float>(kHc);
        {
            const float oh[kHc] = {1, 0, 0, 0};
            ck(cudaMemcpy(one_hot_dev, oh, sizeof oh, cudaMemcpyHostToDevice), "one hot");
        }
        kernels::hc_init();       // K7 / K8 scratch, allocated outside any capture
        kernels::router_init();
        if (const char* v = std::getenv("DS41_GRAPH")) use_graph = v[0] != '0';
        if(pack.native())use_graph=native_options.graph;
        // Reserve staging before the automatic VRAM cache consumes free memory. Keep this reservation at q=0
        // too, so quota sweeps within staged mode use the same initial residency.
        const char* stage_env = std::getenv("DS41_ZC_STAGE");
        if (stage_env && std::strcmp(stage_env, "0") != 0 && std::strcmp(stage_env, "1") != 0)
            throw std::invalid_argument("DS41_ZC_STAGE must be 0 or 1");
        if ((!stage_env || stage_env[0] != '0') && !opt.expert_profile.empty() && opt.ram_budget_gib != 0) {
            std::vector<ExpertBlob> blobs;
            size_t largest = 0;
            for (int l = 0; l < kLayers; ++l) {
                for (int e = 0; e < kExperts; ++e) {
                    const auto& slot = pack.expert(l, e);
                    blobs.push_back({size_t(slot.bytes), size_t(slot.comp_off[0])});
                    largest = std::max(largest, size_t(slot.bytes));
                }
            }
            zc_stage = std::make_unique<ExpertStaging>(kTopK, largest);
            zc_blobs = std::shared_ptr<ExpertBlob>(dalloc<ExpertBlob>(blobs.size()),
                                                  [](ExpertBlob* p) { cudaFree(p); });
            ck(cudaMemcpy(zc_blobs.get(), blobs.data(), blobs.size() * sizeof(ExpertBlob), cudaMemcpyHostToDevice),
               "staging blob metadata");
            ck(cudaStreamSynchronize(nullptr), "staging initialization");
            std::fprintf(stderr, "ds41: zero-copy staging %d slots x %zu bytes (pack max %zu)\n",
                         kTopK, zc_stage->stride(), largest);
        }
        // the VRAM expert tier last: an automatic slot count takes what the rest left free
        if (!opt.expert_profile.empty() && opt.vram_expert_slots != 0) {
            VramExperts::Adapt ad;
            ad.every = opt.adapt_every;
            ad.decay = opt.adapt_decay;
            ad.max_swaps = opt.adapt_swaps;
            vram = std::make_unique<VramExperts>(pack, opt.expert_profile, opt.vram_expert_slots,
                                                 opt.vram_reserve_bytes, ad);
            std::fprintf(stderr, "ds41: %d VRAM expert slots (%.2f GiB)\n", vram->slots(),
                         vram->arena_bytes() / (double) (1ull << 30));
        }
        // the RAM tier after it: the hottest experts the VRAM tier does not hold (upstream's resident budget)
        if (!opt.expert_profile.empty() && opt.ram_budget_gib != 0) {
            const size_t budget = opt.ram_budget_gib < 0 ? auto_ram_budget(24ull << 30)
                                                         : (size_t) (opt.ram_budget_gib * (double) (1ull << 30));
            std::vector<int64_t> handles;
            for (const auto& y : L) handles.push_back(y.moe_handle);
            const double t0 = now_ms();
            host = std::make_unique<HostExperts>(
                pack, read_expert_profile(opt.expert_profile, kLayers, kExperts),
                vram ? vram->res_host() : std::vector<int32_t>((size_t) kLayers * kExperts, -1), budget, handles, 8);
            if (vram) vram->set_host(host.get());
            std::fprintf(stderr, "ds41: RAM tier %d experts (%.1f GiB, %s), filled in %.1f s\n", host->slots(),
                         host->arena_bytes() / (double) (1ull << 30),
                         host->locked() ? "locked" : "not locked", (now_ms() - t0) / 1000.0);
        }
        if (host && host->experts_dev()) {
            // Four PCIe reads cost about 1.05 ms; two CPU experts cost about 1.0-1.2 ms.
            // This is a starting quota for six misses, not a measured optimum.
            int quota = 4;
            if (const char* value = std::getenv("DS41_ZC_QUOTA")) {
                char* end = nullptr;
                const long parsed = std::strtol(value, &end, 10);
                if (end == value || *end || parsed < 0 || parsed > kTopK)
                    throw std::invalid_argument("DS41_ZC_QUOTA must be an integer in [0, 6]");
                quota = (int) parsed;
            }
            zc_quota = std::shared_ptr<int>(dalloc<int>(kLayers), [](int* p) { cudaFree(p); });
            const std::vector<int> quotas(kLayers, quota);
            ck(cudaMemcpy(zc_quota.get(), quotas.data(), quotas.size() * sizeof(int), cudaMemcpyHostToDevice),
               "zero-copy quotas");
            if (!vram)
                zc_workspace = std::shared_ptr<void>(dalloc<uint8_t>(VramExperts::kWorkspaceBytes),
                                                     [](void* p) { cudaFree(p); });
            ck(cudaStreamSynchronize(nullptr), "zero-copy initialization");
            std::fprintf(stderr, "ds41: mapped RAM experts enabled, GPU quota %d per token per layer, %s\n",
                         quota, zc_stage ? "staged" : "direct");
        }
        // the router lookahead: every expert outside the VRAM and RAM tiers is read from the file (upstream turns it
        // on with a RAM budget; here the file tier exists whenever the experts do not all fit in RAM)
        if (const char* f = std::getenv("DS41_FETCH_NOW")) fetch_now = f[0] != '0';
        const char* la_env = std::getenv("DS41_LOOKAHEAD");
        if (!pack.native()&&!(la_env && la_env[0] == '0')) {
            std::vector<std::vector<uint16_t>> rw(kLayers, std::vector<uint16_t>((size_t) kExperts * kDim));
            std::vector<std::vector<float>> rb(kLayers, std::vector<float>(kExperts));
            for (int l = 0; l < kLayers; ++l) {
                ck(cudaMemcpy(rw[l].data(), L[l].gate_w, rw[l].size() * 2, cudaMemcpyDeviceToHost), "router weights");
                ck(cudaMemcpy(rb[l].data(), L[l].gate_bias, rb[l].size() * 4, cudaMemcpyDeviceToHost), "router bias");
            }
            // The VRAM and RAM tables change only between steps; a prediction racing that change warms one expert
            // more or less, never changes what is computed.
            lookahead = std::make_unique<RouterLookahead>(std::move(rw), std::move(rb), kExperts, kDim, kTopK,
                                                          [this](int l, int e) { return warm_file_expert(l, e); });
        }
        worker = std::thread([this] { worker_loop(); });
        if(native_cpu){
            if(const char* requested=std::getenv("DS41_NATIVE_COORDINATOR")){
                char*end=nullptr;long cpu=std::strtol(requested,&end,10);cpu_set_t allowed,mask;
                if(!*requested||*end||cpu<0||cpu>=CPU_SETSIZE||sched_getaffinity(0,sizeof(allowed),&allowed)||!CPU_ISSET(cpu,&allowed))throw std::invalid_argument("invalid native coordinator affinity");
                CPU_ZERO(&mask);CPU_SET(cpu,&mask);native_coordinator_cpu=int(cpu);
                if(pthread_setaffinity_np(worker.native_handle(),sizeof(mask),&mask))throw std::runtime_error("cannot pin native expert coordinator");
            }
        }
        if (const char* p = std::getenv("DS41_DEBUG")) {
            native_block_debug=pack.native() && std::getenv("DS41_DEBUG_NATIVE_BLOCKS");
            if(native_block_debug && std::strcmp(std::getenv("DS41_DEBUG_NATIVE_BLOCKS"),"1")!=0)
                throw std::invalid_argument("native block debug flag must equal1");
            dbg=std::fopen(p,"wb");
            if(!dbg)throw std::runtime_error("cannot open debug probe file");
        }
    }

    void register_cpu_experts(int l) {
        if(native_cpu){
            native_cpu->add_layer(l,"layers."+std::to_string(l)+".ffn.experts",kExperts,kDim,kMoeInter,kTopK,kSwigluLimit);
            return;
        }
        std::vector<MoeCpuMatrixDesc> gate(kExperts), up(kExperts), down(kExperts);
        const uint8_t* base = pack.expert_base();
        for (int e = 0; e < kExperts; ++e) {
            const auto& s = pack.expert(l, e);
            auto desc = [&](int c0, int k_tiles, int n_tiles) {
                MoeCpuMatrixDesc d;
                d.trellis = (const uint16_t*) (base + s.offset + s.comp_off[c0]);
                d.suh = (const at::Half*) (base + s.offset + s.comp_off[c0 + 1]);
                d.svh = (const at::Half*) (base + s.offset + s.comp_off[c0 + 2]);
                d.k_tiles = k_tiles;
                d.n_tiles = n_tiles;
                d.tile_w = (int) (s.comp_bytes[c0] / ((uint64_t) k_tiles * n_tiles * 2));
                return d;
            };
            gate[e] = desc(0, kDim / 16, kMoeInter / 16);   // w1: 5120 -> 2304
            up[e] = desc(4, kDim / 16, kMoeInter / 16);     // w3
            down[e] = desc(8, kMoeInter / 16, kDim / 16);   // w2: 2304 -> 5120
        }
        L[l].moe_handle = exl3_moe_cpu_make_layer_raw(gate.data(), up.data(), down.data(), kExperts, 0, kSwigluLimit, 0);
    }

    const float* rope_at(bool yarn, int pos) const {
        return (yarn ? rope_yarn : rope_plain) + (size_t) pos * kRopeDim;
    }

    /// model.py linear() for one token: K1's activation quantizer, then K1's GEMV (act holds up to 8192 floats)
    void fp8_linear(const bf16* x, const Fp8& w, bf16* y) {
        fp8_quantize_activation_f32((const uint16_t*) x, 1, w.k, act, st);
        fp8_block_gemv_q(act, 1, w.k, w.w, w.s, w.n, (uint16_t*) y, st);
    }

    // ------------------------------------------------------------------------------------- cpu experts
    /// The CPU expert thread: per released step, layers 0..39 in order, each when the GPU publishes it.
    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop.load() || go != seen; });
                if (stop) return;
                seen = go;
            }
            for (int l = 0; l < kLayers; ++l) {
                if (!db->wait_published(l + 1, stop)) return;
                if(native_cpu){
                    const double t0=now_ms();
                    try{
                        if(native_failed.load())throw std::runtime_error("earlier native CPU expert failure");
                        std::vector<uint16_t>x(db->x(),db->x()+kDim);
                        std::vector<int64_t>ids(db->ids(),db->ids()+kTopK);
                        std::vector<float>alpha(db->w(),db->w()+kTopK);
                        auto before=native_gpu_experts?native_gpu_experts->statistics():std::array<uint64_t,5>{};
                        auto output=native_gpu_experts?native_gpu_experts->run(*native_cpu,l,x,ids,alpha):native_cpu->run(l,1,x,ids,alpha);
                        auto after=native_gpu_experts?native_gpu_experts->statistics():std::array<uint64_t,5>{};
                        const int misses=native_gpu_experts?int(after[1]-before[1]):kTopK;worker_misses+=misses;worker_ram+=misses;
                        std::fill_n(db->y(),kDim,0.f);
                        // Adapter returns experts in ascending ID and preserves
                        // source BF16 rounding before this FP32 accumulation.
                        for(int e=0;e<kTopK;e++)for(int c=0;c<kDim;c++){
                            uint32_t bits=uint32_t(output[e*kDim+c])<<16;float value;std::memcpy(&value,&bits,4);db->y()[c]+=value;
                        }
                    }catch(const std::exception&error){
                        if(!native_failed.exchange(true))std::fprintf(stderr,"native CPU expert failure: %s\n",error.what());
                        std::fill_n(db->y(),kDim,NAN);
                    }
                    worker_us+=int64_t((now_ms()-t0)*1000.);
                    db->mark_done(l+1);continue;
                }
                if (lookahead) lookahead->post(l, db->x());   // predict layer l+1 while this layer computes
                c10::Half wh[kTopK];
                for (int i = 0; i < kTopK; ++i) {
                    const __half hv = __float2half_rn(db->w()[i]);
                    wh[i] = c10::Half(__half_as_ushort(hv), c10::Half::from_bits());
                }
                int misses = 0, n_file = 0;
                int32_t file_ids[kTopK];
                for (int i = 0; i < kTopK; ++i) {
                    const int32_t e = db->ids()[i];
                    if (e < 0) continue;
                    ++misses;
                    if (host && host->slot_of(l, e) >= 0) { ++worker_ram; continue; }
                    ++worker_file;
                    file_ids[n_file++] = e;
                    if (file_pages_missing(l, e)) {
                        ++worker_ssd;
                        // upstream fetches a layer's missing experts in one batch before computing: ask for the whole
                        // range now, so the reads run in parallel instead of page fault by page fault
                        if (fetch_now) warm_file_expert(l, e);
                    }
                }
                if (lookahead) lookahead->observe(l, file_ids, n_file);
                // Keep expert_hits as VRAM hits. Timing derives CPU and zero-copy counts from the partition.
                worker_misses += misses + db->counts().zero_copy;
                const double t0 = now_ms();
                try {
                    if (fault("worker")) throw std::runtime_error("ds41 test fault: CPU expert worker");
                    if (misses)
                        exl3_moe_cpu_forward_raw(L[l].moe_handle, (const at::Half*) db->x(), db->ids(), wh, db->y(), 1,
                                                 kTopK, cpu_threads);
                    else
                        std::fill_n(db->y(), kDim, 0.0f);
                } catch (...) {
                    // keep the protocol (the GPU waits for this layer); the step rethrows after its sync
                    std::fill_n(db->y(), kDim, 0.0f);
                    std::lock_guard<std::mutex> lk(mu);
                    if (!worker_error) worker_error = std::current_exception();
                }
                worker_us += (int64_t) ((now_ms() - t0) * 1000.0);
                db->mark_done(l + 1);
            }
        }
    }

    /// Lookahead callback: ask the OS for the pages of (layer, expert) when it is in neither the VRAM nor the RAM tier.
    bool warm_file_expert(int l, int e) {
        if (vram && vram->res_host()[(size_t) l * kExperts + e] >= 0) return false;
        if (host && host->slot_of(l, e) >= 0) return false;
        const ExpertSlot& x = pack.expert(l, e);
        const uintptr_t a = (uintptr_t) (pack.expert_base() + x.offset) & ~(uintptr_t) 4095;
        madvise((void*) a, (uintptr_t) (pack.expert_base() + x.offset + x.bytes) - a, MADV_WILLNEED);
        return true;
    }

    /// True when some page of (layer, expert) in the mapped file is not in RAM: computing it reads the SSD.
    bool file_pages_missing(int l, int e) {
        const ExpertSlot& x = pack.expert(l, e);
        const uintptr_t a = (uintptr_t) (pack.expert_base() + x.offset) & ~(uintptr_t) 4095;
        const size_t len = (uintptr_t) (pack.expert_base() + x.offset + x.bytes) - a;
        mincore_buf.resize((len + 4095) / 4096);
        if (mincore((void*) a, len, mincore_buf.data()) != 0) return false;
        for (unsigned char c : mincore_buf)
            if (!(c & 1)) return true;
        return false;
    }

    // ------------------------------------------------------------------------------------- engram
    /// Hash the n-grams ending at pos for engram layer li: its table rows go to eng_ids[li] (host only).
    void engram_ids(int l, int li, int pos) {
        const auto& hs = pack.engram_hash();
        const int n = hs.max_ngram, nh = hs.n_heads, cols = (n - 1) * nh;
        auto toks=native_ngram_lookback(history,pos,n,hs.pad);
        if (cols > kEngRows) throw std::runtime_error("engram: more n-gram heads than the row buffer holds");
        int64_t* ids = eng_ids[li].data();
        const auto& m = hs.multipliers[li];
        int64_t rolling = toks[0] * m[0];
        for (int i = 1; i < n; ++i) {
            rolling ^= toks[i] * m[i];
            for (int hh = 0; hh < nh; ++hh) {
                const int c = (i - 1) * nh + hh;
                ids[c] = rolling % hs.primes[li][c] + hs.offsets[li][c];
            }
        }
        if (pack.engram_tables()[li].layer != l)
            throw std::runtime_error("engram table order does not match engram_hash.txt");
    }

    /// The rows of every engram table for this step, into the pinned buffer, all reads in flight together.
    void engram_read_all() {
        if (!n_eng) return;
        const auto& hs = pack.engram_hash();
        const int cols = (hs.max_ngram - 1) * hs.n_heads;
        if(pack.native()){
            for(int li=0;li<n_eng;li++){
                std::vector<int64_t>ids(eng_ids[li].begin(),eng_ids[li].begin()+cols);
                auto rows=pack.native()->rows()->gather_bf16("layers."+std::to_string(hs.layers[li])+".engram.embed.weight",ids);
                std::memcpy(eng_host+size_t(li)*kEngRows*engram_row_storage(),rows.data(),rows.size()*2);
            }
            return;
        }
        std::vector<const int64_t*> ids;
        std::vector<uint8_t*> w, s;
        for (int li = 0; li < n_eng; ++li) {
            uint8_t* base = eng_host + (size_t) li * kEngRows * (256 + 8);
            ids.push_back(eng_ids[li].data());
            w.push_back(base);
            s.push_back(base + kEngRows * 256);
        }
        eng_rows->read(ids, cols, w, s);
    }

    /// Engram.forward for layer l (engram layer li) from the rows engram_read put on the device.
    void engram(int l, int li) {
        const auto& hs = pack.engram_hash();
        const int cols = (hs.max_ngram - 1) * hs.n_heads;
        const uint8_t* w = eng_dev + (size_t) li * kEngRows * engram_row_storage();
        if(pack.native())ck(cudaMemcpyAsync(eng_vals,w,size_t(cols)*256*2,cudaMemcpyDeviceToDevice,st),"native Engram rows");
        else ops::engram_dequant(w, w + kEngRows * 256, cols, eng_vals, st);
        fp8_linear(eng_vals, L[l].eng_wkv, eng_kv);
        ops::engram_apply(h, eng_kv, L[l].eng_qw, L[l].eng_kw, kNormEps, 1, st);
    }

    // ------------------------------------------------------------------------------------- indexer
    /// Top-k compressed positions for this layer (Indexer.forward, decode with one query), offset by kWindow,
    /// written after the window part of idx_dev.
    void select_candidate375(const float*input,const int32_t*position,int ratio,int64_t capacity,
                             uint8_t*output,cudaStream_t stream){
        if(!candidate_cutoff375){
            kernels::candidate_blocks_device(input,position,ratio,capacity,kCandidateBlocks,kCandidateBlock,output,stream);
            return;
        }
        // A single ordered target stream owns the workspace, including all
        // sequential verifier rows. No allocation or CUDA host queries in capture.
        if(ds369_candidate_dynamic(input,position,ratio,capacity,kCandidateBlocks,kCandidateBlock,
              output,candidate_workspace375,candidate_workspace_bytes375,stream,1))
            throw std::runtime_error("375 candidate369 submission rejected");
    }

    void indexer(int l, bool have_latent) {
        auto& y = L[l];
        const int ratio = y.ratio;
        const int64_t tcap = ratio == 1 ? tcap1 : tcap2;
        if (is_kv_source(l) && have_latent) {
            ops::bf16_linear(latent, nullptr, y.idx_wk, kHeadDim, kIndexDim, ik, nullptr, st);
            ops::rmsnorm(ik, y.idx_knorm, ik, kIndexDim, kNormEps, 1, st);
            ops::rope_device(ik, 1, kIndexDim, rope_yarn, dp + 1, 1 - ratio, false, st);
            ops::fp4_quant_inplace(ik, kIndexDim, 32, false, st);
            ops::row_copy_device(y.idx_keys, ik, kIndexDim * 2, dp + 1, ratio, max_seq + 1, st);   // row pos / ratio
        }
        if (is_kv_source(l)) cur_index_k = y.idx_keys;
        fp8_linear(qr, y.idx_wq_b, iq);
        ops::rope_device(iq, kIndexHeads, kIndexDim, rope_yarn, dp + 1, 0, false, st);
        ops::fp4_quant_inplace(iq, kIndexHeads * kIndexDim, 32, false, st);
        ops::bf16_linear(xa, nullptr, y.idx_wp, kDim, kIndexHeads, iw_raw, nullptr, st);
        ops::scale_bf16(iw_raw, (float) (std::pow(kIndexDim, -0.5) * std::pow(kIndexHeads, -0.5)), iw, kIndexHeads, st);
        // the candidate layer selects blocks from its own unmasked scores; the layers after it mask with them.
        // t = (pos + 1) / ratio on the device: no work while it is 0 (the first incomplete group)
        const uint8_t* cand = l > kCandidateLayer ? cand_dev : nullptr;
        kernels::indexer_topk_device(iq, cur_index_k, dp + 1, ratio, tcap, iw, cand, kIndexTopK, kWindow, scores,
                                     idx_dev + kWindow, st);
        if (l == kCandidateLayer)
            select_candidate375(scores,dp+1,ratio,tcap,cand_dev,st);
    }

    // ------------------------------------------------------------------------------------- attention
    void attention(int l) {
        auto& y = L[l];
        const bool yarn = y.ratio > 0;
        const float* rope = yarn ? rope_yarn : rope_plain;
        fp8_linear(xa, y.wq_a, qr);
        ops::rmsnorm(qr, y.q_norm, qr, kQLora, kNormEps, 1, st);
        fp8_linear(qr, y.wq_b, q);
        ops::rope_device(q, kHeads, kHeadDim, rope, dp + 1, 0, false, st);
        // sliding window: row pos % 128
        fp8_linear(xa, y.wkv, kvv);
        ops::rmsnorm(kvv, y.kv_norm, kvv, kHeadDim, kNormEps, 1, st);
        ops::rope_device(kvv, 1, kHeadDim, rope, dp + 1, 0, false, st);
        ops::act_quant_inplace(kvv, kHeadDim, st);
        ops::row_copy_device(y.window, kvv, kHeadDim * 2, dp + 1, 1, kWindow, st);
        // idx_dev holds the window part for the whole step (window_index_device at step start)
        if (y.ratio > 0) {
            const int ratio = y.ratio;
            bool have_latent = false;
            if (is_kv_source(l)) {
                if (ratio == 1) {
                    ops::bf16_linear(xa, nullptr, y.c_wkv, kDim, kHeadDim, latent, nullptr, st);
                    ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                    have_latent = true;
                } else {   // ratio 2: slot pos % 2; a group completes at odd positions (one graph per parity)
                    ops::bf16_linear(xa, nullptr, y.c_wkv, kDim, kHeadDim, nullptr, y.kv_state + parity * kHeadDim, st);
                    ops::bf16_linear(xa, nullptr, y.c_wgate, kDim, kHeadDim, nullptr, y.score_state + parity * kHeadDim,
                                     st);
                    if (parity == ratio - 1) {
                        ops::compress_pool(y.kv_state, y.score_state, ratio, latent, 1, st);
                        ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                        have_latent = true;
                    }
                }
                cur_comp = y.comp;
            }
            if (is_index_source(l)) indexer(l, have_latent);
            if (have_latent) {
                ops::rope_device(latent, 1, kHeadDim, rope_yarn, dp + 1, 1 - ratio, false, st);
                ops::fp4_quant_inplace(latent, kHeadDim, 16, true, st);
                ops::row_copy_device(y.comp, latent, kHeadDim * 2, dp + 1, ratio, max_seq + 1, st);   // row pos / ratio
            }
            // n_idx = 128 + min(512, (pos + 1) / ratio): this group's index source wrote them (none yet: 0)
            kernels::sparse_attn_decode_device(q, y.window, cur_comp, idx_dev, ratio == 1 ? dp + 2 : dp + 3, y.sink,
                                               (float) std::pow(kHeadDim, -0.5), o, st);
        } else {
            kernels::sparse_attn_decode(q, y.window, nullptr, idx_dev, 1, kWindow, y.sink,
                                        (float) std::pow(kHeadDim, -0.5), o, st);
        }
        ops::rope_device(o, kHeads, kHeadDim, rope, dp + 1, 0, true, st);
        attention_woa_1017::quantize(o, 1, st); //1017 boundary once, before either wo_a encoding
        if (y.wo_a8.w) wo_a_grouped_fp8(o, y.wo_a8.w, y.wo_a8.s, oa, st);
        else ops::wo_a_grouped(o, y.wo_a, oa, st);
        fp8_linear(oa, y.wo_b, attn_out);
    }

    // ------------------------------------------------------------------------------------- moe
    void moe(int l) {
        auto& y = L[l];
        int32_t* ids = routes_dev + l * kTopK;
        float* w = weights_dev + l * kTopK;
        kernels::router_topk(xf, 1, y.gate_w, y.gate_bias, ids, w, st);
        // K10 computes VRAM hits and a quota of RAM misses (staged or direct) while the CPU computes the rest.
        if(!native_cpu)ops::to_half_fp8q(xf, x_half_dev, kDim, st);
        const bool tier = vram && vram->slots() > 0;
        const auto* ram = host && host->experts_dev() ? host->experts_dev() + (size_t) l * kExperts : nullptr;
        db->publish(native_cpu?reinterpret_cast<const uint16_t*>(xf):x_half_dev, ids, w, 1, tier ? vram->res_dev() + (size_t) l * kExperts : nullptr, gpu_sel,
                    (uint32_t) (l + 1), st, tier ? vram->experts_dev() : nullptr, ram,
                    zc_quota ? zc_quota.get() + l : nullptr,
                    ram ? zc_stage.get() : nullptr, ram && zc_blobs ? zc_blobs.get() + (size_t) l * kExperts : nullptr);
        const bool staged = ram && zc_stage;
        if (staged) zc_stage->fork_copy(st);
        ck(cudaMemsetAsync(routed, 0, kDim * sizeof(float), st), "routed");
        if (!staged && (tier || ram))
            kernels::exl3_moe_decode((const __half*) x_half_dev, 1, gpu_sel, w, kTopK, db->gpu_experts(), routed,
                                     vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        // Shared expert overlaps the staging copy and CPU work. One K10 call preserves route reduction order.
        fp8_linear(xf, y.sh_w1, g);
        fp8_linear(xf, y.sh_w3, u);
        ops::swiglu(g, u, kSwigluLimit, sh_h, kMoeInter, st);
        fp8_linear(sh_h, y.sh_w2, sh_out);
        if (staged) {
            zc_stage->join(st);
            kernels::exl3_moe_decode((const __half*) x_half_dev, 1, gpu_sel, w, kTopK, db->gpu_experts(), routed,
                                     vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        }
        db->wait_add(routed, 1, (uint32_t) (l + 1), st);
        ops::add_f32_bf16(routed, sh_out, ffn_out, kDim, st);
    }

    // ------------------------------------------------------------------------------------- prefill (M3)
    // Layer-major prefill: a pass of S tokens (the whole prompt when it fits; positions p0 .. p0 + S - 1) runs through
    // the layers one at a time. Inside a layer the tokens go in sub-batches of B, in order, through the attention
    // part; then the routed experts run once for all S tokens (each expert copied to the GPU once per pass, not once
    // per sub-batch); then each sub-batch finishes the layer. Every block below is the batched form of the decode
    // code above: the same ops per token, with GEMMs (K2, cuBLAS) instead of GEMVs and the prefill task kernels (K12
    // experts, K13 attention, K14 indexer). It is defined to equal decode token by token (ds41/docs/m3-plan.html,
    // section 0): the window, the compressor state and the indexer's causal limit carry over between sub-batches and
    // passes exactly as between decode steps.
    pair_indexer_v1::Owner* active_pair_indexer_v1 = nullptr;

    static constexpr int kStreamAll = 1024;   ///< from this many tokens, every expert of every layer is streamed
    static constexpr int kNllRows = 64;       ///< logits rows per head GEMM when measuring nll
    static constexpr int kMinRing = 16;       ///< the ring shrinks to this many slots before the pass shrinks
    static constexpr int kEngBatch = 256;     ///< tokens per engram read (its O_DIRECT buffers: 16 KiB per row)
    /// engram reads in flight per table: the rows are small random reads, an NVMe needs a deep queue. Measured on
    /// the 7950X + 4090 box, 32K prompt: 16 (the decode default) 14.9 s of GPU wait, 64 2.6 s (1,148 tok/s).
    static constexpr int kEngIoThreads = 64;

    /// DS41_PF_PROFILE=1: the GPU time of prefill by phase, printed at its end. Events in stream order; the time
    /// between two marks goes to the phase of the first one, so host waits (the routes' sort, expert copies not yet
    /// issued) count in the phase during which the GPU stood idle.
    struct PfProfile {
        enum { kEngram, kHc, kProj, kComp, kIndexer, kAttn, kOut, kRouter, kExperts, kShared, kHead, kUpload, kScatter, kN };
        bool on = false;
        std::array<double,kN> accumulated{};
        std::vector<cudaEvent_t> ev;
        std::vector<int> cat;
        size_t n = 0;
        void mark(int c) {
            if (!on) return;
            if (n == ev.size()) {
                cudaEvent_t e;
                ck(cudaEventCreate(&e), "profile event");
                ev.push_back(e);
                cat.push_back(0);
            }
            ck(cudaEventRecord(ev[n], 0), "profile mark");
            cat[n++] = c;
        }
        void report() {
            if (!on || n < 2) return;
            static const char* names[kN] = {"engram", "hc mixes + norms", "q/kv projections", "compressor",
                                             "indexer", "attention", "o projection", "routing + sort",
                                             "routed experts", "shared expert + hc post", "head", "native packed upload", "native output scatter"};
            ck(cudaEventSynchronize(ev[n - 1]), "profile end");
            double t[kN] = {}, total = 0;
            for (size_t i = 0; i + 1 < n; ++i) {
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, ev[i], ev[i + 1]), "profile time");
                t[cat[i]] += ms;
                total += ms;
            }
            for(int c=0;c<kN;c++)accumulated[c]+=t[c];
            std::fprintf(stderr, "ds41 prefill profile (GPU timeline, %.0f ms):", total);
            for (int c = 0; c < kN; ++c) std::fprintf(stderr, " %s %.0f ms (%.0f%%);", names[c], t[c], 100 * t[c] / total);
            std::fprintf(stderr, "\n");
            n = 0;
        }
    } pfp;

    /// Device scratch of one prefill call: per-pass arrays ([cap] tokens) and per-sub-batch arrays ([sub] tokens),
    /// carved from lent VRAM tier slots or cudaMalloc; the ring separately
    struct Prefill {
        int cap = 0, sub = 0;
        uint8_t* lent = nullptr;                 // lent VRAM tier slots (scratch, ring or both)
        uint8_t* own_scratch = nullptr;          // or cudaMalloc
        uint8_t* own_ring = nullptr;
        size_t slot_bytes = 0;
        uint8_t* ring = nullptr;
        int ring_slots = 0;
        // per pass: the residual streams, the expert input and output, routing, indexer results
        bf16 *h, *h2, *xf;
        float *pre_mix, *ffn_pre, *ffn_post, *ffn_comb, *routed, *wts, *rows_w, *nll_out;
        float *recompute_pre_r1=nullptr,*recompute_inverse_r1=nullptr;
        uint16_t* x_half;
        int32_t *tok, *ids, *topk, *rows_tok, *targets;
        uint8_t* cand;
        // per sub-batch
        bf16 *xa, *qr, *q, *o, *oa, *attn_out, *kvv, *iq, *iw_raw, *iw, *g, *u, *sh_h, *sh_out, *ffn_out, *eng_vals,
            *eng_kv, *final_x, *latent, *ik, *attn_kv, *wo_a_deq;
        float *attn_pre, *attn_post, *attn_comb, *ftmp, *router_logits, *ckv, *csc, *nll_logits;
        int32_t* idx;
        uint8_t* eng_dev;
        kernels::Exl3Expert* desc;
        void *k2_ws, *k12_ws, *k14_ws, *k15_ws;
        size_t k12_bytes = 0, k14_bytes = 0, k15_bytes = 0;
        uint16_t *native_x=nullptr,*native_out=nullptr,*native_decoded=nullptr;
        uint8_t *native_input_codes=nullptr,*native_input_scales=nullptr;
        int32_t *native_sequential_ids=nullptr,*native_candidate_blocks=nullptr;
        uint16_t*native_candidate_u16=nullptr;uint32_t*native_candidate_valid=nullptr;
        int native_expert_batch=0;
        uint8_t *native_packed=nullptr,*native_workspace=nullptr;
        uint8_t *pair_inverse145=nullptr; // three old-layout views in same owned prefill scratch
        size_t native_packed_bytes=0,native_workspace_bytes=0;
    } pf;
    struct NativeHostBuffers {
        std::vector<uint16_t> hc,expert_output;
        std::vector<float> routed;
        std::vector<int32_t> topk,candidates;
        std::unique_ptr<NativeHostScatter> scatter;
    };
    std::unique_ptr<NativeHostBuffers> native_host;
    size_t prefill_h_offset(int row) const {return size_t(host_native_prefill?0:row)*kHc*kDim;}
    size_t prefill_routed_offset(int row)const{return size_t(host_native_prefill?0:row)*kDim;}
    bool host_topk_shadow_r6=false;
    std::vector<int32_t> topk_shadow_r6;
    size_t prefill_topk_offset(int row)const{return size_t((host_native_prefill||host_topk_shadow_r6)?0:row)*kIndexTopK;}
    static size_t available_host_bytes(){
        std::ifstream input("/proc/meminfo");std::string key,line;size_t kb=0;
        while(input>>key){if(key=="MemAvailable:"){input>>kb;return kb*1024;}std::getline(input,line);}
        throw std::runtime_error("cannot read current available host RAM");
    }
    /// Pinned host buffers of prefill, kept between calls (sized for the largest chunk so far)
    struct PrefillHost {
        uint8_t* base = nullptr;
        int cap = 0;
        int32_t *tok, *ids, *rows_tok, *targets;
        float *wts, *rows_w, *pre, *nll;
        kernels::Exl3Expert* desc;
        uint8_t* eng;   // per engram table: cap * kEngRows * 256 weight bytes, then cap * kEngRows * 8 scale bytes
    } pfh;
    std::vector<std::unique_ptr<EngramRows>> eng_rows_pf;   // one reader per table: read one after the other
    int eng_rows_pf_cap = 0;
    std::vector<std::vector<int64_t>> eng_ids_pf;
    std::unique_ptr<ExpertStream> estream;
    std::vector<int64_t> layer_first_job;   ///< stream-all mode: first job of each layer
    PrefillTiming* ptm = nullptr;

    template <typename T>
    static T* carve(uint8_t* base, size_t& used, size_t n) {
        used = (used + 255) & ~(size_t) 255;
        T* p = base ? (T*) (base + used) : nullptr;
        used += n * sizeof(T);
        return p;
    }


    /// Lays the scratch out from `base` (null: only counts) for passes of `cap` tokens and sub-batches of `sub`.
    /// Returns the bytes. The ring is separate.
    size_t layout(Prefill& p, uint8_t* base, int cap, int sub) {
        size_t u = 0;
        const size_t c = (size_t) cap, b = (size_t) sub, h = host_native_prefill?b:c;
        int64_t max_k = 0;
        for (const auto& y : L)
            for (const Fp8* f : {&y.wq_a, &y.wq_b, &y.wkv, &y.wo_b, &y.sh_w1, &y.sh_w2, &y.sh_w3, &y.idx_wq_b, &y.eng_wkv})
                if (f->w) max_k = std::max(max_k, f->k);
        // per pass
        p.h = carve<bf16>(base, u, h * kHc * kDim);
        p.h2 = alias_native_hc ? p.h : carve<bf16>(base, u, h * kHc * kDim);
        p.xf = carve<bf16>(base, u, (native_math320()&&compact_native_prefill_inputs?b:c) * kDim);
        p.x_half = carve<uint16_t>(base, u, native_math320()?1:c*kDim);
        p.routed = carve<float>(base, u, h * kDim);
        p.pre_mix = carve<float>(base, u, c * kHc);
        p.ffn_pre = carve<float>(base, u, c * kHc);
        p.ffn_post = carve<float>(base, u, c * kHc);
        p.ffn_comb = carve<float>(base, u, c * kHc * kHc);
        p.wts = carve<float>(base, u, c * kTopK);
        p.rows_w = carve<float>(base, u, c * kTopK);
        p.nll_out = carve<float>(base, u, c);
        p.tok = carve<int32_t>(base, u, c);
        p.ids = carve<int32_t>(base, u, c * kTopK);
        p.topk = carve<int32_t>(base, u, (host_topk_shadow_r6?b:h) * kIndexTopK);
        p.rows_tok = carve<int32_t>(base, u, c * kTopK);
        p.targets = carve<int32_t>(base, u, c);
        p.cand = carve<uint8_t>(base, u, optimized_native_prefill()?1:c*size_t(max_seq));
        // per sub-batch
        p.xa = carve<bf16>(base, u, b * kDim);
        p.qr = carve<bf16>(base, u, b * kQLora);
        p.q = carve<bf16>(base, u, b * kHeads * kHeadDim);
        p.o = carve<bf16>(base, u, b * kHeads * kHeadDim);
        p.oa = carve<bf16>(base, u, b * kOGroups * kOLora);
        p.attn_out = carve<bf16>(base, u, b * kDim);
        p.kvv = carve<bf16>(base, u, b * kHeadDim);
        p.iq = carve<bf16>(base, u, b * kIndexHeads * kIndexDim);
        p.iw_raw = carve<bf16>(base, u, b * kIndexHeads);
        p.iw = carve<bf16>(base, u, b * kIndexHeads);
        p.g = carve<bf16>(base, u, b * kMoeInter);
        p.u = carve<bf16>(base, u, b * kMoeInter);
        p.sh_h = carve<bf16>(base, u, b * kMoeInter);
        p.sh_out = carve<bf16>(base, u, b * kDim);
        p.ffn_out = carve<bf16>(base, u, b * kDim);
        p.eng_vals = carve<bf16>(base, u, b * kEngRows * 256);
        p.eng_kv = carve<bf16>(base, u, b * (kHc + 1) * kDim);
        p.final_x = carve<bf16>(base, u, b * kDim);
        p.latent = carve<bf16>(base, u, (b + 1) * kHeadDim);
        p.ik = carve<bf16>(base, u, (b + 1) * kIndexDim);
        p.attn_kv = carve<bf16>(base, u, ((optimized_native_prefill()?0:size_t(max_seq)) + kWindow + b) * kHeadDim);
        p.attn_pre = carve<float>(base, u, b * kHc);
        p.attn_post = carve<float>(base, u, b * kHc);
        p.attn_comb = carve<float>(base, u, b * kHc * kHc);
        p.ftmp = carve<float>(base, u, b * kOGroups * kOLora);
        // one layer's wo_a dequantized from FP8 (64 MiB), when the pack keeps it FP8
        p.wo_a_deq = carve<bf16>(base, u, L[0].wo_a8.w ? (size_t) kOGroups * kOLora * (kHeads * kHeadDim / kOGroups) : 1);
        p.router_logits = carve<float>(base, u, b * kExperts);
        p.ckv = carve<float>(base, u, (b + 2) * kHeadDim);
        p.csc = carve<float>(base, u, (b + 2) * kHeadDim);
        p.nll_logits = carve<float>(base, u, (size_t) kNllRows * kVocab);
        p.idx = carve<int32_t>(base, u, b * (kWindow + kIndexTopK));
        p.eng_dev = carve<uint8_t>(base, u, std::max<size_t>((size_t) n_eng * b * kEngRows * (256 + 8), 1));
        p.desc = carve<kernels::Exl3Expert>(base, u, 2 * kExperts);
        p.k2_ws = carve<uint8_t>(base, u, b * (size_t) max_k * 4);
        p.k12_bytes = native_math320() ? 1 : kernels::exl3_moe_prefill_workspace_bytes(cap * kTopK, kExperts);
        p.k12_ws = carve<uint8_t>(base, u, p.k12_bytes);
        p.k14_bytes = kernels::indexer_topk_prefill_workspace_bytes(sub, optimized_native_prefill()?std::max(max_seq,16384):max_seq);
        p.k14_ws = carve<uint8_t>(base, u, p.k14_bytes);
        p.k15_bytes = kernels::hc_mixes_pre_rows_workspace_bytes(sub);
        p.k15_ws = carve<uint8_t>(base, u, std::max<size_t>(p.k15_bytes, 1));
        if(native_math320()){
            if(compact_native_prefill_inputs){
                const auto inputs=native_act_storage_layout(cap,kDim);
                if(recompute_inputs_active_r1){
                    p.recompute_pre_r1=carve<float>(base,u,c*kHc);
                    p.recompute_inverse_r1=carve<float>(base,u,c);
                }else{
                    p.native_input_codes=carve<uint8_t>(base,u,inputs.codes_bytes);
                    p.native_input_scales=carve<uint8_t>(base,u,inputs.scales_bytes);
                }
                p.native_sequential_ids=carve<int32_t>(base,u,b);
            }
            p.native_expert_batch=optimized_native_prefill()?native_host_expert_batch:sub;
            p.native_x=carve<uint16_t>(base,u,size_t(p.native_expert_batch)*kDim);p.native_out=carve<uint16_t>(base,u,size_t(p.native_expert_batch)*kDim);
            if(optimized_native_prefill()){
                if(compact_candidate_prefill&&!host_native_prefill){
                    if(max_k<kCandidateBlocks)throw std::logic_error("Existing K2 scratch cannot hold candidate block batch");
                    // K2 GEMM scratch is dead once the query/weights have been computed.
                    // Pack/unpack and indexer use it in order on the same default stream.
                    p.native_candidate_blocks=reinterpret_cast<int32_t*>(p.k2_ws);
                    const size_t candidate_rows=suffix_candidate_active_r1?std::min<size_t>(c,8192):c;
                    p.native_candidate_u16=carve<uint16_t>(base,u,candidate_rows*kCandidateBlocks);
                    p.native_candidate_valid=carve<uint32_t>(base,u,candidate_rows*(kCandidateBlocks/32));
                }else p.native_candidate_blocks=carve<int32_t>(base,u,(host_native_prefill?b:c)*kCandidateBlocks);
            }
            p.native_packed_bytes=3*(size_t(kDim)*kMoeInter+size_t(kDim)*kMoeInter/32*2)+768;
            p.native_packed=carve<uint8_t>(base,u,p.native_packed_bytes);
            if(paired_layout145)p.pair_inverse145=carve<uint8_t>(base,u,pair145::TripleBytes);
            if(!direct_fp8_expert_weights){
                p.native_workspace_bytes=native_prefill_workspace_bytes(p.native_expert_batch,kDim,kMoeInter);
                p.native_workspace=carve<uint8_t>(base,u,p.native_workspace_bytes);
                if(decoded_native_expert_cache)p.native_decoded=carve<uint16_t>(base,u,3*size_t(kDim)*kMoeInter);
            }
        }
        return (u + 255) & ~(size_t) 255;
    }

    // Initial bounded native batching. Large contexts require the separate
    // host-layerwise scheduler; this path rejects them instead of overallocating.
    void native_prefill_begin(int n){
        if((!optimized_native_prefill()&&max_seq>8192)||opt.prefill_chunk<1||opt.prefill_chunk>(optimized_native_prefill()?1048576:8192)||opt.prefill_batch<1||opt.prefill_batch>(optimized_native_prefill()?4096:4096))
            throw std::invalid_argument("native batched prefill currently requires context<=8192 and batch<=4096");
        if(compact_candidate_prefill)prefill::initialize_blas();
        const char*suffix_mode=std::getenv("DS41_SUFFIX_CANDIDATE_STORAGE_R1");
        suffix_candidate_storage_r1=suffix_mode&&std::strcmp(suffix_mode,"1")==0;
        if(suffix_candidate_storage_r1&&(!decoder_suffix_prefill||!compact_candidate_prefill||host_native_prefill||!gpu_native_prefill))
            throw std::invalid_argument("Suffix candidate storage requires GPU-native compact decoder suffix");
        if(recompute_inputs_active_r1&&(!compact_native_prefill_inputs||!alias_native_hc||host_native_prefill||!gpu_native_prefill))
            throw std::invalid_argument("Recomputed inputs require compact GPU-native aliased HC");
        const char* topk_mode_r6=std::getenv("DS41_NATIVE_HOST_TOPK_R6");
        host_topk_shadow_r6=topk_mode_r6&&std::strcmp(topk_mode_r6,"1")==0;
        if(host_topk_shadow_r6&&(!optimized_native_prefill()||host_native_prefill||!gpu_native_prefill))throw std::invalid_argument("Host topk shadow requires GPU-native prefill only");
        pf=Prefill{};pf.cap=std::min(n,opt.prefill_chunk);pf.sub=std::min(pf.cap,opt.prefill_batch);
        size_t bytes=layout(pf,nullptr,pf.cap,pf.sub),free=0,total=0;
        ck(cudaMemGetInfo(&free,&total),"native prefill memory");
        const auto enabled=[](const char*name){const char*v=std::getenv(name);return v&&std::strcmp(v,"1")==0;};
        size_t reserve=size_t(native_prefill_margin_mib)*1024*1024; // Experimental generic margin; feature buffers below are separate.
        const bool fp8=enabled("DS41_NATIVE_FP8_EXPERTS");
        if(enabled("DS41_NATIVE_DENSE_PREFILL"))reserve+=size_t(fp8?384:512)*1024*1024+size_t((pf.sub+15)&~15)*8192*2+size_t((pf.sub+15)&~15)*32768*4;
        if(enabled("DS41_NATIVE_DENSE_PREFILL")&&enabled("DS41_NATIVE_DENSE_FP8"))reserve+=size_t(32)*1024*1024+size_t((pf.sub+15)&~15)*(8192+4)+4096;
        if(enabled("DS41_NATIVE_PREFILL_PIPELINE"))reserve+=2*pf.native_packed_bytes;
        if(fp8){
            const size_t rows=(pf.native_expert_batch+15)&~size_t(15);
            reserve+=size_t(32)*1024*1024+3*size_t(kDim)*kMoeInter+size_t(2*kMoeInter+kDim)*4;
            reserve+=rows*(7*size_t(kDim)+6*size_t(kMoeInter)+4);
        }
        if(gpu_native_prefill)pf.cap=native_chunk_capacity(pf.cap,pf.sub,free,reserve,[&](int cap){return layout(pf,nullptr,cap,pf.sub);});
        bytes=layout(pf,nullptr,pf.cap,pf.sub);
        if(suffix_candidate_active_r1){
            const size_t rows=std::min(pf.cap,8192);
            std::fprintf(stderr,"suffix candidate R1: cap%d allocated%zu rows, saved%zu bytes; exact block storage\n",pf.cap,rows,size_t(pf.cap-rows)*(2*kCandidateBlocks+4*(kCandidateBlocks/32)));
        }
        if(const char*expected=std::getenv("DS41_NATIVE_PREFILL_EXPECT_CAP_V252")){
            char*end=nullptr;long cap=std::strtol(expected,&end,10);
            if(!*expected||*end||cap<1||cap>524288||(mode4_534()?(cap!=max_seq/2||pf.cap<1||pf.cap>plan528.g.cap):(plan528.armed&&(plan528.mode==2||plan528.mode==3)?(cap!=max_seq/2||pf.cap!=plan528.g.cap):pf.cap!=std::min<long>(n,cap))))throw std::runtime_error("252 prefill actual cap differs from explicit completed-reference geometry");
        }
        if(recompute_inputs_active_r1)std::fprintf(stderr,"recomputed inputs R1: cap%d saved%zu bytes, unchanged FP8 quantization\n",pf.cap,size_t(pf.cap)*(kDim+kDim/32-(kHc+1)*sizeof(float)));
        output16_capacity_v252=pf.native_packed_bytes;output16_last_cap_v252=pf.cap;output16_last_sub_v252=pf.sub;
        if(free<bytes+reserve)throw std::runtime_error("native prefill scratch plus feature buffers exceeds free VRAM");
        if(host_topk_shadow_r6){
            topk_shadow_r6.resize(size_t(pf.cap)*kIndexTopK);
            std::fprintf(stderr,"host topk R6: cap%d batch%d GPUbytes%zu hostbytes%zu unchanged INT32 indices\n",pf.cap,pf.sub,size_t(pf.sub)*kIndexTopK*4,size_t(pf.cap)*kIndexTopK*4);
        }
        std::fprintf(stderr,"native prefill planned reserve %.3fGiB including optional feature buffers\n",double(reserve)/1073741824.0);
        if(headless320){
            headless_current320();if(pf.cap!=fixed335.cap||pf.sub!=fixed335.B||pf.native_expert_batch!=fixed335.E)throw std::runtime_error("320 refuses changed cap/B/E BEFORE PP allocation");const auto dense=pack.headless_dense_info_v320();
            const uint64_t persistent=headless_own_bytes320+dense[7];
            if(bytes>headless_options320.max_device_bytes||reserve>headless_options320.max_device_bytes-bytes||
               persistent>headless_options320.max_device_bytes-bytes-reserve||
               free<headless_options320.reserve_bytes||bytes+reserve>free-headless_options320.reserve_bytes)
                throw std::runtime_error("320 PP scratch/features exceed explicit total budget/free reserve BEFORE allocations");
        }
        ck(cudaMalloc(reinterpret_cast<void**>(&pf.own_scratch),bytes),"native prefill scratch");
        layout(pf,pf.own_scratch,pf.cap,pf.sub);
        if(paired_layout145)pair145::record_prefill(*pack.native()->weights(),pf.cap,pf.sub);
        if(compact_candidate_prefill&&!host_native_prefill)compact_candidate_saved_peak=std::max(compact_candidate_saved_peak,uint64_t(pf.cap)*3840);
        if(compact_native_prefill_inputs){
            std::vector<int32_t>ids(pf.sub);for(int i=0;i<pf.sub;i++)ids[i]=i;
            ck(cudaMemcpy(pf.native_sequential_ids,ids.data(),ids.size()*4,cudaMemcpyHostToDevice),"native compact input row IDs");
        }
        if(host_native_prefill){
            native_host_prefill_layout(pf.cap,native_host_budget,available_host_bytes(),size_t(80)<<30);
            native_host=std::make_unique<NativeHostBuffers>();
            native_host->hc.resize(size_t(pf.cap)*kHc*kDim);
            native_host->routed.resize(size_t(pf.cap)*kDim);
            native_host->topk.resize(size_t(pf.cap)*kIndexTopK);
            native_host->candidates.resize(size_t(pf.cap)*kCandidateBlocks);
            native_host->expert_output.resize(size_t(pf.native_expert_batch)*kDim);
            native_host->scatter=std::make_unique<NativeHostScatter>(native_host->routed.data(),native_host->routed.size(),pf.cap,kDim,native_options.cpus);
        }
        if(pfh.cap<pf.cap){
            if(pfh.base)cudaFreeHost(pfh.base);pfh.base=nullptr;pfh.cap=0;
            size_t used=0;auto host_layout=[&](uint8_t*base){used=0;const size_t c=pf.cap;
                pfh.tok=carve<int32_t>(base,used,c);pfh.ids=carve<int32_t>(base,used,c*kTopK);
                pfh.targets=carve<int32_t>(base,used,c);pfh.wts=carve<float>(base,used,c*kTopK);
                pfh.pre=carve<float>(base,used,c*kHc);pfh.nll=carve<float>(base,used,c);
                pfh.eng=carve<uint8_t>(base,used,host_native_prefill?1:std::max<size_t>(n_eng*c*kEngRows*512,1));};
            host_layout(nullptr);ck(cudaHostAlloc(reinterpret_cast<void**>(&pfh.base),used,cudaHostAllocDefault),"native prefill host");
            host_layout(pfh.base);pfh.cap=pf.cap;
            for(int i=0;i<pf.cap;i++)for(int j=0;j<kHc;j++)pfh.pre[i*kHc+j]=j==0?1.f:0.f;
        }
        if(!host_native_prefill)eng_ids_pf.assign(n_eng,std::vector<int64_t>(size_t(pf.cap)*kEngRows));
        std::fprintf(stderr,"native prefill: pass%d batch%d expert%d decoded_once%d scratch%.3fGiB; no EXL3 reinterpretation\n",pf.cap,pf.sub,pf.native_expert_batch,int(decoded_native_expert_cache),bytes/1073741824.0);
    }

    /// Chooses the pass and sub-batch sizes and gets the scratch and the ring: lent VRAM tier slots (up to 90% of the
    /// tier, as upstream) and free VRAM, in that order of preference. Larger passes first (each pass copies every
    /// expert to the GPU once, so the pass size decides the prefill speed of long prompts); for a pass, the largest
    /// sub-batch (up to opt.prefill_batch) and ring (opt.prefill_ring, at least 16 slots) that fit.
    void prefill_begin(int n) {
        if (const char* v = std::getenv("DS41_PF_PROFILE")) pfp.on = v[0] && v[0] != '0';
        else pfp.on=false;
        pfp.accumulated.fill(0.);
        if(native_math320()){native_prefill_begin(n);return;}
        pf = Prefill{};
        if (const char* v = std::getenv("DS41_PF_PROFILE")) pfp.on = v[0] && v[0] != '0';
        // a ring slot holds any expert of the pack (upstream: MAXBLOB); the tier's slots may be smaller
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e) pf.slot_bytes = std::max<size_t>(pf.slot_bytes, pack.expert(l, e).bytes);
        pf.slot_bytes = (pf.slot_bytes + 255) & ~(size_t) 255;
        // the bytes of the last 90% of the tier's slots (upstream lends up to 90% of the tier)
        const size_t lendable = vram ? vram->tail_bytes(vram->slots() * 9 / 10) : 0;
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t spare = free_b > (512ull << 20) ? free_b - (512ull << 20) : 0;   // cuBLAS and K13 scratch
        size_t scratch = 0, ring = 0;
        int plan = -1;   // 0: both lent; 1: scratch lent, ring cudaMalloc; 2: ring lent, scratch cudaMalloc; 3: both cudaMalloc
        int cap = std::max(1, std::min(n, opt.prefill_chunk)), sub = 0;
        for (; plan < 0; cap /= 2) {
            // the smallest pass is 16 tokens, or the whole prompt when it is shorter
            if (cap < std::min(n, 16)) throw std::runtime_error("ds41 prefill: not enough VRAM for a 16-token pass");
            for (sub = std::max(1, std::min(cap, opt.prefill_batch)); plan < 0 && sub >= std::min(cap, 512); sub /= 2) {
                scratch = layout(pf, nullptr, cap, sub);
                for (int r = std::max(opt.prefill_ring, kMinRing); plan < 0 && r >= kMinRing; r /= 2) {
                    pf.ring_slots = r;
                    ring = (size_t) r * pf.slot_bytes;
                    if (scratch + ring <= lendable) plan = 0;
                    else if (scratch <= lendable && ring <= spare) plan = 1;
                    else if (ring <= lendable && scratch <= spare) plan = 2;
                    else if (scratch + ring <= spare) plan = 3;
                }
                if (plan >= 0) break;
            }
            if (plan >= 0) break;
        }
        pf.cap = cap;
        pf.sub = sub;
        const size_t lend_bytes = plan == 0 ? scratch + ring : plan == 1 ? scratch : plan == 2 ? ring : 0;
        if (lend_bytes) pf.lent = vram->lend_bytes(lend_bytes);
        uint8_t* scratch_base = plan <= 1 ? pf.lent : nullptr;
        if (plan >= 2) ck(cudaMalloc((void**) &pf.own_scratch, scratch), "prefill scratch");
        if (plan == 1 || plan == 3) ck(cudaMalloc((void**) &pf.own_ring, ring), "prefill ring");
        if (!scratch_base) scratch_base = pf.own_scratch;
        layout(pf, scratch_base, cap, sub);
        pf.ring = plan == 0 ? pf.lent + scratch : plan == 2 ? pf.lent : pf.own_ring;
        std::fprintf(stderr, "ds41 prefill: pass %d tokens, sub-batch %d; scratch %.2f GiB %s; ring %d slots %s; %d tier "
                     "slots lent\n", cap, sub, scratch / 1073741824.0, plan <= 1 ? "in lent slots" : "cudaMalloc",
                     pf.ring_slots, plan == 0 || plan == 2 ? "in lent slots" : "cudaMalloc",
                     lend_bytes ? vram->lent() : 0);
        // pinned host buffers
        if (pfh.cap < cap) {
            if (pfh.base) cudaFreeHost(pfh.base);
            pfh.base = nullptr;   // a failed allocation below must not leave the freed block behind
            pfh.cap = 0;
            size_t u = 0;
            auto lay = [&](uint8_t* b) {
                u = 0;
                const size_t c = (size_t) cap;
                pfh.tok = carve<int32_t>(b, u, c);
                pfh.ids = carve<int32_t>(b, u, c * kTopK);
                pfh.rows_tok = carve<int32_t>(b, u, c * kTopK);
                pfh.targets = carve<int32_t>(b, u, c);
                pfh.wts = carve<float>(b, u, c * kTopK);
                pfh.rows_w = carve<float>(b, u, c * kTopK);
                pfh.pre = carve<float>(b, u, c * kHc);
                pfh.nll = carve<float>(b, u, c);
                pfh.desc = carve<kernels::Exl3Expert>(b, u, 2 * kExperts);
                pfh.eng = carve<uint8_t>(b, u, std::max<size_t>((size_t) n_eng * c * kEngRows * (256 + 8), 1));
            };
            lay(nullptr);
            ck(cudaHostAlloc((void**) &pfh.base, u, cudaHostAllocDefault), "prefill pinned buffers");
            lay(pfh.base);
            pfh.cap = cap;
            for (int i = 0; i < cap; ++i)
                for (int j = 0; j < kHc; ++j) pfh.pre[i * kHc + j] = j == 0 ? 1.0f : 0.0f;
        }
        if (n_eng && eng_rows_pf.empty())
            for (const auto& t : pack.engram_tables())
                eng_rows_pf.push_back(std::make_unique<EngramRows>(
                    std::vector<EngramRows::Table>{{t.path, t.weight_offset, t.scale_offset}}, kEngBatch * kEngRows,
                    256, 8, kEngIoThreads));
        if (n_eng && eng_rows_pf_cap < cap) {
            eng_rows_pf_cap = cap;
            eng_ids_pf.assign(n_eng, std::vector<int64_t>((size_t) cap * kEngRows, 0));
        }
        estream = std::make_unique<ExpertStream>(pack, host.get(), pf.ring, pf.ring_slots, pf.slot_bytes,
                                                 std::max(1, opt.prefill_threads), std::max(1, opt.prefill_host_buffers),
                                                 prefill_unbuffered());
    }


    /// Upstream's file-tier rule (v0.1.40 file_tier_unbuffered / file_cache_keeps, #577): the experts outside the
    /// RAM tier are read with O_DIRECT when the file cache could not keep them anyway (available RAM, after the RAM
    /// tier, less 4 GiB, below their bytes); otherwise through the file cache. The VRAM tier's experts count too:
    /// prefill lends their slots and refills them. DS41_UNBUFFERED=0 / 1 forces it.
    bool prefill_unbuffered() {
        if (const char* v = std::getenv("DS41_UNBUFFERED"); v && v[0]) return v[0] != '0';
        uint64_t read = 0;
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e)
                if (!(host && host->slot_of(l, e) >= 0)) read += pack.expert(l, e).bytes;
        const uint64_t avail = auto_ram_budget(0), margin = 4ull << 30;
        const bool keeps = avail > margin && avail - margin >= read;
        std::fprintf(stderr, "ds41 prefill: %.1f GiB available, %.1f GiB of experts read from the pack: %s\n",
                     avail / 1073741824.0, read / 1073741824.0,
                     keeps ? "through the file cache" : "unbuffered (O_DIRECT): the file cache could not keep them");
        return !keeps;
    }

    /// Returns the lent slots (or frees the scratch). After an error the stream is stopped without draining (its
    /// unreleased jobs would never complete).
    /// The destructor's form: the VRAM tier (owner of lent slots) is gone already
    void prefill_end_quiet() {
        estream.reset();
        if (pf.own_scratch) cudaFree(pf.own_scratch);
        if (pf.own_ring) cudaFree(pf.own_ring);
        pf.lent = pf.own_scratch = pf.own_ring = nullptr;
        native_host.reset();
    }

    void prefill_end() {
        estream.reset();
        ck(cudaDeviceSynchronize(), "prefill end");
        if (pf.own_scratch) cudaFree(pf.own_scratch);
        if (pf.own_ring) cudaFree(pf.own_ring);
        if (pf.lent && vram) vram->restore();
        pf.lent = pf.own_scratch = pf.own_ring = nullptr;
        native_host.reset();
    }

    bool resident(int l, int e) const { return vram && vram->res_host()[(size_t) l * kExperts + e] >= 0; }

    /// model.py linear() for T rows (K2)
    void fp8_rows(const bf16* x, int T, const Fp8& w, bf16* y) {
        if(active_dense_prefill&&active_dense_prefill->run(reinterpret_cast<const uint16_t*>(x),T,w.k,w.w,w.s,w.n,reinterpret_cast<uint16_t*>(y)))return;
        kernels::fp8_block_gemm(x, T, w.k, w.w, w.s, w.n, y, pf.k2_ws, 0);
    }

    /// K7's hyper-connection mixes for T rows (task K15)
    void hc_rows(const bf16* x, int T, const float* fn, const float* scale, const float* base, const float* pre_in,
                 bf16* y, float* pre, float* post, float* comb) {
        kernels::hc_mixes_pre_rows(x, T, fn, scale, base, pre_in, y, pre, post, comb, pf.k15_ws, pf.k15_bytes, 0);
    }

    /// The compressed entries of kv source l completed in this chunk (decode: attention(), compressor part), their
    /// indexer keys, and the compressor state left for decode.
    void compress_rows(int l, int T, int p0) {
        auto& y = L[l];
        const int r = y.ratio;
        int G, g0;
        if (r == 1) {
            prefill::bf16_gemm(pf.xa, y.c_wkv, T, kDim, kHeadDim, pf.latent, nullptr, pf.ftmp);
            ops::rmsnorm(pf.latent, y.c_norm, pf.latent, kHeadDim, kNormEps, T);
            G = T;
            g0 = p0;
        } else {
            // the rows of the group in progress (decode's state slots 0 .. carry-1), then this chunk's rows
            const int carry = p0 % r;
            if (carry) {
                ck(cudaMemcpyAsync(pf.ckv, y.kv_state, (size_t) carry * kHeadDim * 4, cudaMemcpyDeviceToDevice, 0), "carry");
                ck(cudaMemcpyAsync(pf.csc, y.score_state, (size_t) carry * kHeadDim * 4, cudaMemcpyDeviceToDevice, 0),
                   "carry");
            }
            prefill::bf16_gemm(pf.xa, y.c_wkv, T, kDim, kHeadDim, nullptr, pf.ckv + (size_t) carry * kHeadDim);
            prefill::bf16_gemm(pf.xa, y.c_wgate, T, kDim, kHeadDim, nullptr, pf.csc + (size_t) carry * kHeadDim);
            G = (carry + T) / r;
            g0 = (p0 - carry) / r;
            ops::compress_pool(pf.ckv, pf.csc, r, pf.latent, G);
            ops::rmsnorm(pf.latent, y.c_norm, pf.latent, kHeadDim, kNormEps, G);
            const int rem = (carry + T) % r;   // the next group's first rows: decode's state slots 0 .. rem-1
            if (rem) {
                ck(cudaMemcpyAsync(y.kv_state, pf.ckv + (size_t) G * r * kHeadDim, (size_t) rem * kHeadDim * 4,
                                   cudaMemcpyDeviceToDevice, 0), "state");
                ck(cudaMemcpyAsync(y.score_state, pf.csc + (size_t) G * r * kHeadDim, (size_t) rem * kHeadDim * 4,
                                   cudaMemcpyDeviceToDevice, 0), "state");
            }
        }
        if (G > 0) {
            // indexer keys (decode: indexer(), have_latent), from the latent before its RoPE
            prefill::bf16_gemm(pf.latent, y.idx_wk, G, kHeadDim, kIndexDim, pf.ik, nullptr, pf.ftmp);
            ops::rmsnorm(pf.ik, y.idx_knorm, pf.ik, kIndexDim, kNormEps, G);
            prefill::rope_rows(pf.ik, G, 1, kIndexDim, rope_yarn, g0 * r, r, false);
            ops::fp4_quant_inplace(pf.ik, G * kIndexDim, 32, false);
            ck(cudaMemcpyAsync(y.idx_keys + (size_t) g0 * kIndexDim, pf.ik, (size_t) G * kIndexDim * 2,
                               cudaMemcpyDeviceToDevice, 0), "index keys");
            if (active_pair_indexer_v1) active_pair_indexer_v1->publish(l, y.idx_keys, g0, G);
            prefill::rope_rows(pf.latent, G, 1, kHeadDim, rope_yarn, g0 * r, r, false);
            ops::fp4_quant_inplace(pf.latent, G * kHeadDim, 16, true);
            ck(cudaMemcpyAsync(y.comp + (size_t) g0 * kHeadDim, pf.latent, (size_t) G * kHeadDim * 2,
                               cudaMemcpyDeviceToDevice, 0), "compressed kv");
        }
        cur_comp = y.comp;
        cur_index_k = y.idx_keys;
    }

    /// Indexer of index source l for the T queries of the sub-batch at pass row b0 (decode: indexer()): their top-512
    /// compressed rows into pf.topk; the candidate layer writes each query's candidate mask, the later layers mask
    /// with it (both per pass, at row b0).
    void indexer_rows(int l, int T, int p0, int b0) {
        auto& y = L[l];
        fp8_rows(pf.qr, T, y.idx_wq_b, pf.iq);
        prefill::rope_rows(pf.iq, T, kIndexHeads, kIndexDim, rope_yarn, p0, 1, false);
        ops::fp4_quant_inplace(pf.iq, T * kIndexHeads * kIndexDim, 32, false);
        prefill::bf16_gemm(pf.xa, y.idx_wp, T, kDim, kIndexHeads, pf.iw_raw, nullptr, pf.ftmp);
        ops::scale_bf16(pf.iw_raw, (float) (std::pow(kIndexDim, -0.5) * std::pow(kIndexHeads, -0.5)), pf.iw,
                        T * kIndexHeads);
        if (active_pair_indexer_v1 && active_pair_indexer_v1->run(l, y.ratio, pf.iq, cur_index_k, pf.iw,
                T, p0, pf.topk + prefill_topk_offset(b0), pf.k14_ws, pf.k14_bytes)) {
            int32_t* top=pf.topk+prefill_topk_offset(b0);
            if (host_native_prefill) ck(cudaMemcpy(native_host->topk.data() + size_t(b0) * kIndexTopK,
                pf.topk + prefill_topk_offset(b0), size_t(T) * kIndexTopK * 4, cudaMemcpyDeviceToHost), "host pair topk down");
            if(host_topk_shadow_r6)ck(cudaMemcpy(topk_shadow_r6.data()+size_t(b0)*kIndexTopK,top,size_t(T)*kIndexTopK*4,cudaMemcpyDeviceToHost),"host topk R6 save unchanged indices");
            return;
        }
        if(optimized_native_prefill()){
            int32_t* top=pf.topk+prefill_topk_offset(b0);
            const bool packed=compact_candidate_prefill&&!host_native_prefill;
            int32_t* blocks=pf.native_candidate_blocks+size_t((host_native_prefill||packed)?0:b0)*kCandidateBlocks;
            if(l==kCandidateLayer){
                kernels::indexer_topk_prefill_blocks(pf.iq,cur_index_k,pf.iw,T,p0,y.ratio,kIndexTopK,kCandidateBlocks,kCandidateBlock,
                    top,blocks,pf.k14_ws,pf.k14_bytes,0);
                if(packed){
                    const int begin=suffix_candidate_active_r1?std::max(b0,suffix_candidate_first_r1):b0;
                    const int rows=b0+T-begin;
                    if(rows>0){
                        const int offset=begin-(suffix_candidate_active_r1?suffix_candidate_first_r1:0);
                        const int capacity=suffix_candidate_active_r1?std::min(pf.cap,8192):pf.cap;
                        if(offset<0||offset+rows>capacity)throw std::logic_error("Suffix candidate pack bounds");
                        native_candidate_pack(blocks+size_t(begin-b0)*kCandidateBlocks,pf.native_candidate_u16+size_t(offset)*kCandidateBlocks,pf.native_candidate_valid+size_t(offset)*(kCandidateBlocks/32),rows,0);
                        compact_candidate_packed_rows+=rows;
                    }
                }
                if(host_native_prefill)ck(cudaMemcpy(native_host->candidates.data()+size_t(b0)*kCandidateBlocks,blocks,size_t(T)*kCandidateBlocks*4,cudaMemcpyDeviceToHost),"host candidate blocks down");
            }else if(l>kCandidateLayer){
                if(packed){
                    const int offset=b0-(suffix_candidate_active_r1?suffix_candidate_first_r1:0);
                    const int capacity=suffix_candidate_active_r1?std::min(pf.cap,8192):pf.cap;
                    if(offset<0||offset+T>capacity)throw std::logic_error("Suffix candidate unpack bounds");
                    native_candidate_unpack(pf.native_candidate_u16+size_t(offset)*kCandidateBlocks,pf.native_candidate_valid+size_t(offset)*(kCandidateBlocks/32),blocks,T,0);compact_candidate_unpacked_rows+=T;
                }
                if(host_native_prefill)ck(cudaMemcpy(blocks,native_host->candidates.data()+size_t(b0)*kCandidateBlocks,size_t(T)*kCandidateBlocks*4,cudaMemcpyHostToDevice),"host candidate blocks up");
                kernels::indexer_topk_prefill_from_blocks(pf.iq,cur_index_k,pf.iw,blocks,T,p0,y.ratio,top,pf.k14_ws,pf.k14_bytes,0);
            }else kernels::indexer_topk_prefill(pf.iq,cur_index_k,pf.iw,T,p0,y.ratio,nullptr,nullptr,0,kIndexTopK,0,kCandidateBlocks,kCandidateBlock,top,pf.k14_ws,pf.k14_bytes,0);
            if(host_native_prefill)ck(cudaMemcpy(native_host->topk.data()+size_t(b0)*kIndexTopK,top,size_t(T)*kIndexTopK*4,cudaMemcpyDeviceToHost),"host topk down");
            if(host_topk_shadow_r6)ck(cudaMemcpy(topk_shadow_r6.data()+size_t(b0)*kIndexTopK,top,size_t(T)*kIndexTopK*4,cudaMemcpyDeviceToHost),"host topk R6 save unchanged indices");
            return;
        }
        uint8_t* cand = pf.cand + (size_t) b0 * max_seq;
        kernels::indexer_topk_prefill(pf.iq, cur_index_k, pf.iw, T, p0, y.ratio, l > kCandidateLayer ? cand : nullptr,
                                      l == kCandidateLayer ? cand : nullptr, max_seq, kIndexTopK, 0, kCandidateBlocks,
                                      kCandidateBlock, pf.topk + (size_t) b0 * kIndexTopK, pf.k14_ws, pf.k14_bytes, 0);
    }

    /// decode: attention(), for the T rows of the sub-batch at pass row b0 (positions p0 ..). One KV buffer per layer
    /// and sub-batch: [compressed rows][window rows for positions p0 - 127 .. p0 + T - 1] (K13's layout).
    void attention_rows(int l, int T, int p0, int b0) {
        auto& y = L[l];
        const bool yarn = y.ratio > 0;
        const float* table = yarn ? rope_yarn : rope_plain;
        pfp.mark(PfProfile::kProj);
        fp8_rows(pf.xa, T, y.wq_a, pf.qr);
        ops::rmsnorm(pf.qr, y.q_norm, pf.qr, kQLora, kNormEps, T);
        fp8_rows(pf.qr, T, y.wq_b, pf.q);
        prefill::rope_rows(pf.q, T, kHeads, kHeadDim, table, p0, 1, false);
        fp8_rows(pf.xa, T, y.wkv, pf.kvv);
        ops::rmsnorm(pf.kvv, y.kv_norm, pf.kvv, kHeadDim, kNormEps, T);
        prefill::rope_rows(pf.kvv, T, 1, kHeadDim, table, p0, 1, false);
        ops::act_quant_inplace(pf.kvv, T * kHeadDim);
        int win_base = 0, n_idx = kWindow;
        const int32_t* topk = nullptr;
        if (y.ratio > 0) {
            pfp.mark(PfProfile::kComp);
            if (is_kv_source(l)) compress_rows(l, T, p0);
            pfp.mark(PfProfile::kIndexer);
            if (is_index_source(l)) indexer_rows(l, T, p0, b0);
            else if(host_native_prefill)ck(cudaMemcpy(pf.topk,native_host->topk.data()+size_t(b0)*kIndexTopK,size_t(T)*kIndexTopK*4,cudaMemcpyHostToDevice),"host reused topk up");
            else if(host_topk_shadow_r6)ck(cudaMemcpy(pf.topk,topk_shadow_r6.data()+size_t(b0)*kIndexTopK,size_t(T)*kIndexTopK*4,cudaMemcpyHostToDevice),"host topk R6 restore unchanged indices");
            pfp.mark(PfProfile::kAttn);
            const int c_end = (p0 + T) / y.ratio;   // compressed rows any query of the chunk may see
            if (c_end && !optimized_native_prefill())
                ck(cudaMemcpyAsync(pf.attn_kv, cur_comp, (size_t) c_end * kHeadDim * 2, cudaMemcpyDeviceToDevice, 0),
                   "attention kv: compressed rows");
            win_base = c_end;
            n_idx = kWindow + kIndexTopK;
            topk = pf.topk + prefill_topk_offset(b0);
        }
        // window rows: the ring's positions before the chunk, then the chunk's own; then the ring for decode
        pfp.mark(PfProfile::kAttn);
        const int stored_win_base=optimized_native_prefill()?0:win_base;
        const int prev = std::min(p0, kWindow - 1);
        prefill::window_gather(y.window, p0, prev, pf.attn_kv + (size_t) (stored_win_base + kWindow - 1 - prev) * kHeadDim);
        ck(cudaMemcpyAsync(pf.attn_kv + (size_t) (stored_win_base + kWindow - 1) * kHeadDim, pf.kvv, (size_t) T * kHeadDim * 2,
                           cudaMemcpyDeviceToDevice, 0), "attention kv: window rows");
        prefill::window_scatter(y.window, pf.kvv, p0, T);
        prefill::attn_index_rows(T, p0, win_base, topk, kIndexTopK, pf.idx, n_idx);
        if(optimized_native_prefill())kernels::sparse_attn_prefill_split(pf.q,y.ratio?cur_comp:nullptr,pf.attn_kv,win_base,pf.idx,T,n_idx,y.sink,(float)std::pow(kHeadDim,-0.5),pf.o,0);
        else kernels::sparse_attn_prefill(pf.q,pf.attn_kv,pf.idx,T,n_idx,y.sink,(float)std::pow(kHeadDim,-0.5),pf.o,0);
        pfp.mark(PfProfile::kOut);
        prefill::rope_rows(pf.o, T, kHeads, kHeadDim, table, p0, 1, true);
        attention_woa_1017::quantize(pf.o, T, 0); //1017 boundary over T*8 contiguous4096 groups
        const bf16* wo_a = y.wo_a;
        if (y.wo_a8.w) {   // FP8 pack: the BF16 weight convert.py would store, then the same GEMM
            // Dedicated scratch persists across all sub-batches of this layer.
            // The layer loop starts at b0=0; source weights do not change.
            if(b0==native_first_prefill_row)dequant_wo_a(y.wo_a8.w, y.wo_a8.s, pf.wo_a_deq);
            wo_a = pf.wo_a_deq;
        }
        prefill::wo_a_grouped_rows(pf.o, wo_a, T, pf.oa, pf.ftmp);
        fp8_rows(pf.oa, T, y.wo_b, pf.attn_out);
    }


    /// decode: moe(), routing part, for the sub-batch at pass row b0: router (GPU logits), the routes into the pass
    /// arrays, and the experts' input (FP8-quantized, fp16).
    bf16* prefill_ffn_input(int b0){return pf.xf+size_t(native_math320()&&compact_native_prefill_inputs?0:b0)*kDim;}
    void moe_route(int l, int T, int b0) {
        auto& y = L[l];
        const bf16* xf = prefill_ffn_input(b0);
        pfp.mark(PfProfile::kRouter);
        prefill::bf16_gemm(xf, y.gate_w, T, kDim, kExperts, nullptr, pf.router_logits);
        if(native_images){
            for(auto run:native_images->runs(native_prefill_position+b0,T))
                prefill::route_rows(pf.router_logits+size_t(run.first)*kExperts,run.image?y.gate_bias_vl:y.gate_bias,run.count,
                    pf.ids+size_t(b0+run.first)*kTopK,pf.wts+size_t(b0+run.first)*kTopK);
        }else prefill::route_rows(pf.router_logits,y.gate_bias,T,pf.ids+size_t(b0)*kTopK,pf.wts+size_t(b0)*kTopK);
        if(!native_math320())ops::to_half_fp8q(xf, pf.x_half + (size_t) b0 * kDim, T * kDim);
        else if(compact_native_prefill_inputs&&!recompute_inputs_active_r1)native_act_storage_pack(reinterpret_cast<const uint16_t*>(xf),T,kDim,
            pf.native_input_codes+size_t(b0)*kDim,pf.native_input_scales+size_t(b0)*(kDim/32));
    }

    void native_moe_experts(int l,int S){
        auto& y=L[l];
        pfp.mark(PfProfile::kRouter);
        const int first=native_first_prefill_row;
        if(first<0||first>=S)throw std::logic_error("invalid suffix route range");
        ck(cudaMemcpyAsync(pfh.ids+size_t(first)*kTopK,pf.ids+size_t(first)*kTopK,size_t(S-first)*kTopK*4,cudaMemcpyDeviceToHost,0),"native routes down");
        ck(cudaMemcpyAsync(pfh.wts+size_t(first)*kTopK,pf.wts+size_t(first)*kTopK,size_t(S-first)*kTopK*4,cudaMemcpyDeviceToHost,0),"native routing weights down");
        if(host_native_prefill)native_host->scatter->begin_layer();
        else ck(cudaMemsetAsync(pf.routed+size_t(first)*kDim,0,size_t(S-first)*kDim*4,0),"native routed zero");
        ck(cudaStreamSynchronize(0),"native routes ready");
        auto plan=native_route_plan(pfh.ids+size_t(first)*kTopK,pfh.wts+size_t(first)*kTopK,S-first,kTopK,kExperts);
        for(auto&token:plan.tokens)token+=first;
        ck(cudaMemcpyAsync(pf.rows_tok,plan.tokens.data(),plan.tokens.size()*4,cudaMemcpyHostToDevice,0),"native row IDs");
        ck(cudaMemcpyAsync(pf.rows_w,plan.weights.data(),plan.weights.size()*4,cudaMemcpyHostToDevice,0),"native row weights");
        ck(cudaStreamSynchronize(0),"native row metadata ready");
        auto weights=pack.native()->weights();
        const bool direct531=pair531::enabled();
        uint64_t generation531=0,epoch531=0;
        if(direct531){
            if(!paired_layout145)throw std::logic_error("531 requires actual typed139 registry");
            const auto paired=pair145::info(*weights);
            if(paired[0]!=145||paired[1]!=139||paired[2]!=46080||!paired[4]||!paired[5])throw std::logic_error("531 actual139 registry generation/epoch required");
            generation531=paired[4];epoch531=paired[5];
        }
        const char*pipeline_mode534=std::getenv("DS41_NATIVE_PREFILL_PIPELINE");
        const bool pipeline_on534=pipeline_mode534&&!std::strcmp(pipeline_mode534,"1");
        if(mode4_534()){
            if(!direct531||!upload_lease534||!active_fp8_experts||host_native_prefill||headless320||!pipeline_on534)throw std::logic_error("534 mandatory exact FP8/531 lease missing");
            upload_lease534->validate(pf.native_packed_bytes,0,plan528.epoch,generation531,epoch531);
        }
        pass558::PipelineView pipeline(pass_upload558(l),mode4_534()?upload_lease534.get():nullptr,pipeline_on534,pf.native_packed_bytes,l);
        const char*stage_copy_mode=std::getenv("DS41_NATIVE_PREFILL_PARALLEL_STAGE_COPY");
        const bool parallel_stage_copy=pipeline&&stage_copy_mode&&std::strcmp(stage_copy_mode,"1")==0&&(!headless320||parallel_headless_copy331); //331 ON uses thread-owned325 pool; OFF preserves original320 headless exclusion
        int ordinal=0;
        for(int e=0;e<kExperts;e++){
            const int begin=plan.offsets[e],end=plan.offsets[e+1];if(begin==end)continue;
            if(native_gpu_budget||native_gu_budget_v19||headless320){
                if(native_cache_tail_active){
                    uint64_t uses=0;
                    for(int i=begin;i<end;i++){
                        const int position=native_prefill_position+plan.tokens[i];
                        if(position>=native_cache_tail_begin&&position<native_cache_tail_end)++uses;
                    }
                    native_expert_frequency[l][e]+=uses;
                    native_cache_tail_counted_uses+=uses;
                }else native_expert_frequency[l][e]+=end-begin;
            }
            pfp.mark(PfProfile::kUpload);
            NativeDeviceProjection device[3];NativeDeviceCompactV252 compact_device[3]{};size_t used=0;int which=0;
            bool any_compact=false,any_raw=false,any_pair145=false,any_old145=false;
            const int slot=ordinal++%2;uint8_t*host=pipeline?pipeline->begin(slot):nullptr;uint8_t*packed=pipeline?pipeline->device(slot):pf.native_packed;
            for(const char*name:{"w1","w3","w2"}){
                const bool down=std::strcmp(name,"w2")==0;
                const std::string prefix="layers."+std::to_string(l)+".ffn.experts."+std::to_string(e)+"."+name;
                if(pair145::contains(*weights,prefix)){
                    if(!paired_layout145||!pf.pair_inverse145||!output16_compact_v252||!direct_fp8_expert_weights||!active_fp8_experts)throw std::logic_error("Paired139 requires explicit145 GPU0 inverse staging");
                    const auto c=pair145::projection(*weights,prefix,down?kDim:kMoeInter,down?kMoeInter:kDim);
                    used=(used+255)&~size_t(255);
                    if(c.bytes>pf.native_packed_bytes-used)throw std::runtime_error("paired139 projection exceeds unchanged H2D staging allocation");
                    auto*destination=packed+used;
                    if(pipeline){if(parallel_headless_copy331)native_stage_copy_v331(host+used,static_cast<const uint8_t*>(c.data),c.bytes,parallel_stage_copy);else native_stage_copy_v1(host+used,static_cast<const uint8_t*>(c.data),c.bytes,parallel_stage_copy);}
                    else ck(cudaMemcpyAsync(destination,c.data,c.bytes,cudaMemcpyHostToDevice,0),"paired139 payload upload");
                    used+=c.bytes;compact_device[which++]={destination,size_t(c.bytes),0};any_compact=any_pair145=true;
                    output16_uploads_v252++;output16_bytes_v252+=c.bytes;
                }else if(weights->is_output16_compact(prefix)){
                    any_old145=true;
                    if(!output16_compact_v252||!direct_fp8_expert_weights||!active_fp8_experts)throw std::logic_error("Compact projection requires explicit252 FP8 feeding");
                    const auto c=weights->compact_projection(prefix,down?kDim:kMoeInter,down?kMoeInter:kDim);
                    used=(used+255)&~size_t(255);
                    if(c.bytes>pf.native_packed_bytes-used)throw std::runtime_error("compact output16 projection exceeds unchanged staging allocation");
                    auto*destination=packed+used;
                    if(pipeline){if(parallel_headless_copy331)native_stage_copy_v331(host+used,c.data,c.bytes,parallel_stage_copy);else native_stage_copy_v1(host+used,c.data,c.bytes,parallel_stage_copy);}
                    else ck(cudaMemcpyAsync(destination,c.data,c.bytes,cudaMemcpyHostToDevice,0),"compact expert payload upload");
                    used+=c.bytes;compact_device[which++]={destination,c.bytes,c.bits};any_compact=true;
                    output16_uploads_v252++;output16_bytes_v252+=c.bytes;
                }else{
                    // Existing raw representation, stage ownership, copy order,
                    // FP8 encoder and FFN operations are textually unchanged.
                    auto retained=weights->stage_projection(prefix,down?kDim:kMoeInter,down?kMoeInter:kDim);
                    const auto&v=retained.projection;
                    auto copy=[&](const uint8_t*data,size_t bytes){used=(used+255)&~size_t(255);
                        if(bytes>pf.native_packed_bytes-used)throw std::runtime_error("native packed projection exceeds staging allocation");
                        uint8_t*dest=packed+used;
                        if(pipeline){if(parallel_headless_copy331)native_stage_copy_v331(host+used,data,bytes,parallel_stage_copy);else native_stage_copy_v1(host+used,data,bytes,parallel_stage_copy);}else ck(cudaMemcpyAsync(dest,data,bytes,cudaMemcpyHostToDevice,0),"native expert packed upload");
                        used+=bytes;return dest;};
                    device[which++]={copy(v.weights,size_t(v.rows)*v.weight_row_bytes),v.scales?copy(v.scales,size_t(v.rows)*v.scale_row_bytes):nullptr,v.bits};
                    if(!pipeline&&retained.owned_bytes())ck(cudaStreamSynchronize(0),"bounded cross-shard upload lifetime");
                    any_raw=true;output16_raw_uploads_v252++;
                }
            }
            if((any_compact&&any_raw)||(any_pair145&&any_old145))throw std::invalid_argument("252/145 requires one complete expert layout triple, never partial aliases");
            if(mode4_534()&&(!any_pair145||!any_compact||any_raw||any_old145))throw std::logic_error("534 complete paired139 triple required");
            if(pipeline)pipeline->upload(slot,used);
            if(any_pair145&&!direct531){
                // Same default stream: original pipeline upload wait precedes
                // byte inverse, which precedes the LITERAL original252 encoder.
                // All three destinations are disjoint; no hot alloc/host sync.
                for(int i=0;i<3;i++){auto*old=pf.pair_inverse145+size_t(i)*pair145::MatrixBytes;
                    const int status=ds141_pair32_to_output16(compact_device[i].data,compact_device[i].bytes,i==2?kDim:kMoeInter,i==2?kMoeInter:kDim,139,old,pair145::MatrixBytes,headless320?1:0,0);
                    if(status)throw std::runtime_error("paired139 GPU0 inverse141 refused: "+std::to_string(status));
                    compact_device[i]={old,pair145::MatrixBytes,0};
                }
                pair145::record_inverse(*weights,3,pair145::TripleBytes);
            }
            if(any_compact){
                output16_max_triple_v252=std::max(output16_max_triple_v252,uint64_t(used));
                pfp.mark(PfProfile::kExperts);
                if(any_pair145&&direct531){
                    pair531::Inputs typed;typed.owner_generation=generation531;typed.bank_format_epoch=epoch531;
                    typed.expected_device=headless320?1:0;typed.stream=0;
                    for(int i=0;i<3;i++)typed.source[i]={compact_device[i].data,compact_device[i].bytes,compact_device[i].bits};
                    pair531::Scope scope(active_fp8_experts,typed);
                    active_fp8_experts->output16_weights_v252(compact_device[0],compact_device[1],compact_device[2]);
                    scope.require_consumed();
                    if(mode4_534()){fast534::add(14);fast534::add(15,3);}
                }else active_fp8_experts->output16_weights_v252(compact_device[0],compact_device[1],compact_device[2]);
                output16_encodes_v252+=3;direct_fp8_projection_encodes+=3;
            }else if(direct_fp8_expert_weights){
                if(!active_fp8_experts)throw std::logic_error("Direct packed FP8 experts missing");
                pfp.mark(PfProfile::kExperts);active_fp8_experts->packed_weights(device[0],device[1],device[2]);direct_fp8_projection_encodes+=3;
            }else if(decoded_native_expert_cache){
                pfp.mark(PfProfile::kExperts);
                native_expert_decode_weights(kDim,kMoeInter,device[0],device[1],device[2],pf.native_decoded,3*size_t(kDim)*kMoeInter);
                if(active_fp8_experts)active_fp8_experts->weights(pf.native_decoded);
            }
            for(int b=begin;b<end;b+=pf.native_expert_batch){
                pfp.mark(PfProfile::kExperts);
                int rows=std::min(pf.native_expert_batch,end-b);const int32_t*ids=pf.rows_tok+b;
                if(recompute_inputs_active_r1)ds41_recompute_gather_r1(reinterpret_cast<const uint16_t*>(pf.h2),pf.recompute_pre_r1,pf.recompute_inverse_r1,reinterpret_cast<const uint16_t*>(y.ffn_norm),ids,rows,S,pf.native_x);
                else if(compact_native_prefill_inputs)native_act_storage_gather(pf.native_input_codes,pf.native_input_scales,ids,rows,S,kDim,pf.native_x);
                else native_prefill_gather(reinterpret_cast<const uint16_t*>(pf.xf),ids,rows,S,kDim,pf.native_x);
                if(active_fp8_experts)active_fp8_experts->run(pf.native_x,pf.rows_w+b,rows,pf.native_out);
                else if(decoded_native_expert_cache)native_expert_prefill_decoded(pf.native_x,pf.rows_w+b,rows,kDim,kMoeInter,kSwigluLimit,pf.native_decoded,3*size_t(kDim)*kMoeInter,pf.native_out,pf.native_workspace,pf.native_workspace_bytes);
                else native_expert_prefill(pf.native_x,pf.rows_w+b,rows,kDim,kMoeInter,kSwigluLimit,device[0],device[1],device[2],pf.native_out,pf.native_workspace,pf.native_workspace_bytes);
                pfp.mark(PfProfile::kScatter);
                if(host_native_prefill){
                    ck(cudaMemcpy(native_host->expert_output.data(),pf.native_out,size_t(rows)*kDim*2,cudaMemcpyDeviceToHost),"host expert output down");
                    native_host->scatter->add_view(e,plan.tokens.data()+b,rows,native_host->expert_output.data(),size_t(rows)*kDim);
                }else native_prefill_scatter(pf.native_out,ids,rows,S,kDim,pf.routed);
            }
            if(pipeline)pipeline->used(slot);
            ++ptm->streamed;
        }
    }

    /// decode: moe(), routed experts, for all S tokens of the pass at once: the host sorts the (token, expert) rows
    /// by expert, VRAM tier experts first (one K12 call from their slots), then the streamed experts in job order
    /// (one K12 call per group of ring slots, each released when its call has run). Each expert is copied once.
    void moe_experts(int l, int S) {
        if(native_math320()){native_moe_experts(l,S);return;}
        pfp.mark(PfProfile::kRouter);
        ck(cudaMemcpyAsync(pfh.ids, pf.ids, (size_t) S * kTopK * 4, cudaMemcpyDeviceToHost, 0), "routes down");
        ck(cudaMemcpyAsync(pfh.wts, pf.wts, (size_t) S * kTopK * 4, cudaMemcpyDeviceToHost, 0), "weights down");
        ck(cudaMemsetAsync(pf.routed, 0, (size_t) S * kDim * 4, 0), "routed");
        ck(cudaStreamSynchronize(0), "routes");
        std::vector<int> count(kExperts, 0);
        for (int i = 0; i < S * kTopK; ++i) count[pfh.ids[i]]++;
        std::vector<int> order;
        for (int e = 0; e < kExperts; ++e)
            if (count[e] && resident(l, e)) order.push_back(e);
        const int n_res = (int) order.size();
        int64_t first_job;
        int n_jobs = 0;
        if (!native_math320() && S >= kStreamAll) {
            first_job = layer_first_job[l];
            n_jobs = (int) (layer_first_job[l + 1] - first_job);
            for (int e = 0; e < kExperts; ++e)
                if (!resident(l, e)) order.push_back(e);
        } else {
            std::vector<std::pair<int, int>> jobs;
            for (int e = 0; e < kExperts; ++e)
                if (count[e] && !resident(l, e)) {
                    jobs.push_back({l, e});
                    order.push_back(e);
                }
            n_jobs = (int) jobs.size();
            first_job = n_jobs ? estream->push(jobs) : 0;
        }
        std::vector<int> start(kExperts, 0), fill(kExperts, 0);
        std::vector<int32_t> off(order.size() + 1, 0);
        for (size_t i = 0; i < order.size(); ++i) {
            start[order[i]] = off[i];
            off[i + 1] = off[i] + count[order[i]];
        }
        for (int t = 0; t < S; ++t)
            for (int j = 0; j < kTopK; ++j) {
                const int e = pfh.ids[t * kTopK + j];
                const int r = start[e] + fill[e]++;
                pfh.rows_tok[r] = t;
                pfh.rows_w[r] = pfh.wts[t * kTopK + j];
            }
        ck(cudaMemcpyAsync(pf.rows_tok, pfh.rows_tok, (size_t) S * kTopK * 4, cudaMemcpyHostToDevice, 0), "rows up");
        ck(cudaMemcpyAsync(pf.rows_w, pfh.rows_w, (size_t) S * kTopK * 4, cudaMemcpyHostToDevice, 0), "rows up");
        const auto* xh = (const __half*) pf.x_half;
        pfp.mark(PfProfile::kExperts);
        if (n_res) {
            for (int i = 0; i < n_res; ++i) pfh.desc[i] = vram->desc(vram->res_host()[(size_t) l * kExperts + order[i]]);
            ck(cudaMemcpyAsync(pf.desc, pfh.desc, n_res * sizeof(kernels::Exl3Expert), cudaMemcpyHostToDevice, 0),
               "descriptors up");
            kernels::exl3_moe_prefill(xh, pf.rows_tok, pf.rows_w, off.data(), n_res, pf.desc, pf.routed, pf.k12_ws,
                                      pf.k12_bytes, 0);
            ptm->vram_experts += n_res;
        }
        const int group = std::max(1, pf.ring_slots / 4);
        for (int g0 = 0; g0 < n_jobs; g0 += group) {
            const int n = std::min(group, n_jobs - g0);
            for (int i = 0; i < n; ++i) {
                uint8_t* slot = estream->wait(first_job + g0 + i, 0);   // stream 0 waits for the copy on the GPU
                pfh.desc[kExperts + g0 + i] = VramExperts::describe_at(pack, l, order[n_res + g0 + i], slot);
            }
            ck(cudaMemcpyAsync(pf.desc + kExperts + g0, pfh.desc + kExperts + g0, n * sizeof(kernels::Exl3Expert),
                               cudaMemcpyHostToDevice, 0), "descriptors up");
            kernels::exl3_moe_prefill(xh, pf.rows_tok, pf.rows_w, off.data() + n_res + g0, n, pf.desc + kExperts + g0,
                                      pf.routed, pf.k12_ws, pf.k12_bytes, 0);
            for (int i = 0; i < n; ++i) estream->release(first_job + g0 + i, 0);
        }
        ptm->streamed += n_jobs;
    }

    /// decode: moe() end and the ffn sub-block's hc_post, for the sub-batch at pass row b0: the shared expert, plus
    /// the routed sum, into the residual stream h.
    void prefill_hc_post(const bf16*out,const bf16*res,const float*post,const float*comb,bf16*dest,int rows){
        if(alias_native_hc){
            if(res!=dest)throw std::logic_error("in-place HC buffers differ");
            ops::hc_post_inplace(out,dest,post,comb,rows);
        }else ops::hc_post(out,res,post,comb,dest,rows);
    }
    void moe_finish(int l, int T, int b0) {
        auto& y = L[l];
        if(recompute_inputs_active_r1)ds41_recompute_gather_r1(reinterpret_cast<const uint16_t*>(pf.h2+prefill_h_offset(b0)),pf.recompute_pre_r1+size_t(b0)*kHc,pf.recompute_inverse_r1+b0,reinterpret_cast<const uint16_t*>(y.ffn_norm),pf.native_sequential_ids,T,T,reinterpret_cast<uint16_t*>(pf.xf));
        else if(native_math320()&&compact_native_prefill_inputs)native_act_storage_gather(pf.native_input_codes+size_t(b0)*kDim,
            pf.native_input_scales+size_t(b0)*(kDim/32),pf.native_sequential_ids,T,T,kDim,reinterpret_cast<uint16_t*>(pf.xf));
        const bf16* xf = prefill_ffn_input(b0);
        pfp.mark(PfProfile::kShared);
        fp8_rows(xf, T, y.sh_w1, pf.g);
        fp8_rows(xf, T, y.sh_w3, pf.u);
        ops::swiglu(pf.g, pf.u, kSwigluLimit, pf.sh_h, T * kMoeInter);
        fp8_rows(pf.sh_h, T, y.sh_w2, pf.sh_out);
        ops::add_f32_bf16(pf.routed + prefill_routed_offset(b0), pf.sh_out, pf.ffn_out, T * kDim);
        prefill_hc_post(pf.ffn_out, pf.h2 + prefill_h_offset(b0), pf.ffn_post + b0 * kHc, pf.ffn_comb + b0 * kHc * kHc,
                     pf.h + prefill_h_offset(b0), T);
        ck(cudaMemcpyAsync(pf.pre_mix + b0 * kHc, pf.ffn_pre + b0 * kHc, (size_t) T * kHc * 4, cudaMemcpyDeviceToDevice,
                           0), "pre_mix");
    }

    /// The engram rows of a pass, table by table (engram layer order), into the pinned buffer in token order. A row
    /// that several positions share is read once. Runs on its own thread while the GPU works; done[t] is set when
    /// table t is in the buffer (or carries the read error).
    void engram_pass(int S, int p0, std::vector<std::promise<void>>& done, double& ms) {
        const double t0 = now_ms();
        size_t t = 0;
        try {
            if(native_coordinator_cpu>=0){cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(native_coordinator_cpu,&mask);if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask))throw std::runtime_error("cannot pin native Engram prefill worker");}
            const auto& hs = pack.engram_hash();
            const int cols = (hs.max_ngram - 1) * hs.n_heads;
            if (cols != kEngRows) throw std::runtime_error("engram: the prefill buffers assume 24 rows per token");
            const size_t eng_table = (size_t) pf.cap * kEngRows * engram_row_storage();
            for (int l = 0; l < kLayers; ++l) {
                if (!is_engram_layer(l)) continue;
                const int li = (int) t;
                std::vector<int64_t>& all = eng_ids_pf[li];
                for (int i = 0; i < S; ++i) {
                    engram_ids(l, li, p0 + i);
                    std::copy(eng_ids[li].begin(), eng_ids[li].begin() + cols, all.begin() + (size_t) i * cols);
                }
                const size_t n = (size_t) S * cols;
                const char*hash_mode=std::getenv("DS41_ENGRAM_HASH_DEDUP_R1");
                if(native_math320()&&hash_mode&&std::strcmp(hash_mode,"1")==0){
                    const double plan_started=now_ms();
                    const size_t limit=size_t(2)*1024*1024*1024/512;
                    std::unordered_map<int64_t,uint32_t>row_index;
                    row_index.max_load_factor(.8f);row_index.reserve(std::min(n,size_t(262144)));
                    std::vector<int64_t>unique_rows;unique_rows.reserve(std::min(n,size_t(262144)));
                    std::vector<uint32_t>indices(n);bool bounded=true;
                    for(size_t i=0;i<n;i++){
                        auto[it,inserted]=row_index.try_emplace(all[i],uint32_t(unique_rows.size()));
                        if(inserted){
                            if(unique_rows.size()==limit){bounded=false;break;}
                            unique_rows.push_back(all[i]);
                        }
                        indices[i]=it->second;
                    }
                    const double plan_ms=now_ms()-plan_started;
                    if(bounded&&unique_rows.size()*2<n){
                        const double gather_started=now_ms();
                        std::vector<uint16_t>values(unique_rows.size()*256);

                        const char*workers_env_r1=std::getenv("DS41_ENGRAM_GATHER_WORKERS_R1");
                        int gather_workers_r1=0;
                        if(workers_env_r1&&std::strcmp(workers_env_r1,"0")!=0){
                            if(std::strcmp(workers_env_r1,"8")!=0)throw std::invalid_argument("Engram parallel R1 requires explicit8 physical spread workers");
                            gather_workers_r1=8;
                        }
                        if(gather_workers_r1){
                            std::atomic<size_t>next_row_r1{0};std::vector<std::future<void>>jobs_r1;jobs_r1.reserve(8);
                            const std::string name_r1="layers."+std::to_string(l)+".engram.embed.weight";
                            for(int worker_r1=0;worker_r1<8;worker_r1++)jobs_r1.emplace_back(std::async(std::launch::async,[&,worker_r1]{
                                const int planned_cpu539=(worker_r1%2)*128+(worker_r1/2)*32;
                                const int cpu_r1=planned_cpu539==64?68:planned_cpu539; //64..67 belong exclusively to continuous HF uploads
                                cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu_r1,&mask);
                                if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)||sched_getcpu()!=cpu_r1)throw std::runtime_error("Engram worker physical affinity differs");
                                eng_parallel_cpus_r1[cpu_r1/64].fetch_or(uint64_t(1)<<(cpu_r1%64),std::memory_order_relaxed);
                                uint64_t local_rows_r1=0;
                                while(true){
                                    const size_t first=next_row_r1.fetch_add(4096,std::memory_order_relaxed);
                                    if(first>=unique_rows.size())break;
                                    const size_t last=std::min(unique_rows.size(),first+4096);
                                    std::vector<int64_t>rows(unique_rows.begin()+first,unique_rows.begin()+last);
                                    auto decoded=pack.native()->rows()->gather_bf16(name_r1,rows);
                                    if(decoded.size()!=(last-first)*256)throw std::logic_error("Parallel Engram row width differs");
                                    std::memcpy(values.data()+first*256,decoded.data(),decoded.size()*2);local_rows_r1+=last-first;
                                }
                                eng_parallel_nodes_r1[cpu_r1/128].fetch_add(local_rows_r1,std::memory_order_relaxed);
                            }));
                            // Each future owns its thread; unwinding also waits for all readers.
                            for(auto&job_r1:jobs_r1)job_r1.get();
                            eng_parallel_calls_r1.fetch_add(1,std::memory_order_relaxed);eng_parallel_rows_r1.fetch_add(unique_rows.size(),std::memory_order_relaxed);eng_parallel_workers_r1.store(8,std::memory_order_relaxed);
                            std::fprintf(stderr,"Engram parallel gather R1: workers8 physicalCPUs0,32,68,96,128,160,192,224 rows%zu; independent4096-row jobs, exact original decoder\n",unique_rows.size());
                        }else{
                        for(size_t first=0;first<unique_rows.size();first+=4096){
                            std::vector<int64_t>rows(unique_rows.begin()+first,unique_rows.begin()+std::min(unique_rows.size(),first+4096));
                            auto decoded=pack.native()->rows()->gather_bf16("layers."+std::to_string(l)+".engram.embed.weight",rows);
                            std::memcpy(values.data()+first*256,decoded.data(),decoded.size()*2);
                        }
                        }

                        const double gather_ms=now_ms()-gather_started,expand_started=now_ms();
                        auto*out=reinterpret_cast<uint16_t*>(pfh.eng+size_t(li)*eng_table);
                        for(size_t i=0;i<n;i++)std::memcpy(out+i*256,values.data()+size_t(indices[i])*256,512);
                        std::fprintf(stderr,"Engram hash dedup R2 layer%d rows%zu unique%zu plan_ms%.3f gather_ms%.3f expand_ms%.3f\n",l,n,unique_rows.size(),plan_ms,gather_ms,now_ms()-expand_started);
                        ptm->engram_rows+=n;ptm->engram_unique+=unique_rows.size();done[li].set_value();++t;continue;
                    }
                    std::fprintf(stderr,"Engram hash dedup R2 fallback layer%d rows%zu unique%zu bounded%d\n",l,n,unique_rows.size(),int(bounded));
                }
                std::vector<int64_t> uniq(all.begin(), all.begin() + n);
                std::sort(uniq.begin(), uniq.end());
                uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
                if(native_math320()&&hash_mode&&std::strcmp(hash_mode,"1")==0)std::fprintf(stderr,"Engram hash fallback sorted R2 layer%d rows%zu unique%zu\n",l,n,uniq.size());
                if(native_math320()){
                    // Bounded gathers also cover segmented own Engram tensors.
                    auto*out=reinterpret_cast<uint16_t*>(pfh.eng+size_t(li)*eng_table);
                    // Repeated n-grams share immutable rows. Cap this temporary
                    // decoded cache at256MiB; large unique sets use bounded tiles.
                    const bool dedup=uniq.size()<=size_t(256)*1024*1024/512&&uniq.size()*2<n;
                    if(dedup){
                        std::vector<uint16_t>values(uniq.size()*256);
                        for(size_t first=0;first<uniq.size();first+=4096){
                            std::vector<int64_t>rows(uniq.begin()+first,uniq.begin()+std::min(uniq.size(),first+4096));
                            auto decoded=pack.native()->rows()->gather_bf16("layers."+std::to_string(l)+".engram.embed.weight",rows);
                            std::memcpy(values.data()+first*256,decoded.data(),decoded.size()*2);
                        }
                        for(size_t i=0;i<n;i++){size_t index=std::lower_bound(uniq.begin(),uniq.end(),all[i])-uniq.begin();std::memcpy(out+i*256,values.data()+index*256,512);}
                    }else for(size_t first=0;first<n;first+=4096){
                        std::vector<int64_t>rows(all.begin()+first,all.begin()+std::min(n,first+4096));
                        auto decoded=pack.native()->rows()->gather_bf16("layers."+std::to_string(l)+".engram.embed.weight",rows);
                        std::memcpy(out+first*256,decoded.data(),decoded.size()*2);
                    }
                    ptm->engram_rows+=n;ptm->engram_unique+=uniq.size();done[li].set_value();++t;continue;
                }
                std::vector<uint8_t> uw(uniq.size() * 256), us(uniq.size() * 8);
                const size_t batch = (size_t) kEngBatch * kEngRows;
                for (size_t r = 0; r < uniq.size(); r += batch)
                    eng_rows_pf[li]->read({uniq.data() + r}, (int) std::min(batch, uniq.size() - r), {uw.data() + r * 256},
                                          {us.data() + r * 8});
                uint8_t* w = pfh.eng + (size_t) li * eng_table;
                uint8_t* sc = w + (size_t) pf.cap * kEngRows * 256;
                for (size_t i = 0; i < n; ++i) {
                    const size_t u = (size_t) (std::lower_bound(uniq.begin(), uniq.end(), all[i]) - uniq.begin());
                    std::memcpy(w + i * 256, uw.data() + u * 256, 256);
                    std::memcpy(sc + i * 8, us.data() + u * 8, 8);
                }
                ptm->engram_rows += (int64_t) n;
                ptm->engram_unique += (int64_t) uniq.size();
                done[li].set_value();
                ++t;
            }
        } catch (...) {
            for (; t < done.size(); ++t) done[t].set_exception(std::current_exception());
        }
        ms = now_ms() - t0;
    }

    /// One pass: tokens[c0 .. c0 + S) at positions p0 ..; with nll, the nll of every next token inside `tokens`.
    //317 mechanical pass/layer lifetime seam. ONE active state per Engine.
    // No rank1 Engine/CPU pool/headless mode or cross-device transport is created.
    struct PrefillBoundary317 {
        uint64_t epoch=0;int layer=-1,next_layer=0,c0=0,S=0,p0=0,suffix_first=0,absolute_end=0,ratio=0,carry=0,source_layer=-1;
        cudaEvent_t ready=nullptr; // borrowed, re-recorded by NEXT advance
        const bf16*window=nullptr,*comp=nullptr,*index_keys=nullptr;const float*kv_state=nullptr,*score_state=nullptr;
        int compressed_rows=0;size_t window_bytes=0,comp_bytes=0,index_bytes=0,carry_bytes=0;
    };
    // PRIVATE attention-state certificate. It deliberately has NO next_layer or
    // completed-layer field: FFN and original317layer_done are still pending.
    struct AttentionBoundary337 {
        uint32_t version=337,phase=337;uint64_t epoch=0;int layer=-1,absolute_end=0,ratio=0,carry=0,source_layer=-1,compressed_rows=0;
        cudaEvent_t ready=nullptr;const bf16*window=nullptr,*comp=nullptr,*index_keys=nullptr;const float*kv_state=nullptr,*score_state=nullptr;
        size_t window_bytes=0,comp_bytes=0,index_bytes=0,carry_bytes=0;
    };
    struct PrefillPassState317 {
        Impl&owner;const std::vector<int>&tokens;int c0,S,p0,B=0,suffix_first=0,eng_i=0,next_layer=0,device=-1,source_layer=-1;
        std::vector<float>*nll;uint64_t epoch=0;std::thread::id issuer;bool bound=false,finished=false,later_pushed=false,streamed_dense_v191=false;
        size_t eng_table=0;std::unique_ptr<NativeFp8Experts>fp8_scope;std::unique_ptr<NativeDensePrefill>dense_scope;
        std::unique_ptr<pass558::Lease>upload558;
        std::vector<std::promise<void>>eng_done;std::vector<std::future<void>>eng_ready;double eng_thread_ms=0;std::thread eng_thread;
        std::vector<std::pair<int,int>>later_jobs;cudaEvent_t attention_ready337=nullptr;ds337::Protocol early_phase337;std::function<void(const AttentionBoundary337&)>early_callback337;
        void early_abort_after_drain337()noexcept{if(early_phase337.borrowed()){early337::drop(14);early337::add(15);}early_phase337.abort_after_drain();early_callback337={};}
        cudaEvent_t layer_done=nullptr;const void*scratch_identity=nullptr,*hc_identity=nullptr,*routed_identity=nullptr;
        PrefillPassState317(Impl&e,const std::vector<int>&t,int c,int count,int pos,std::vector<float>*loss):owner(e),tokens(t),c0(c),S(count),p0(pos),nll(loss),issuer(std::this_thread::get_id()){}
        ~PrefillPassState317()noexcept{
            if(attention_ready337&&early_phase337.borrowed())std::terminate();
            if(bound){int current=-1;if(issuer!=std::this_thread::get_id()||cudaGetDevice(&current)!=cudaSuccess||current!=device)std::terminate();}
            // The worker captures THIS owned state, never begin() stackrefs.
            // Join before promises/history/row scratch can be destroyed.
            if(eng_thread.joinable())eng_thread.join();eng_ready.clear();eng_done.clear();
            if(bound){
                if(owner.active_pass317!=this)std::terminate();
                if(upload558){upload558->close();upload558.reset();}
                owner.active_dense_prefill=nullptr;dense_scope.reset();owner.active_fp8_experts=nullptr;fp8_scope.reset();
                owner.native_first_prefill_row=0;owner.active_pass317=nullptr;
                if(!finished)prefill317_detail::bump(4);prefill317_detail::drop(5);
            }
            if(attention_ready337){if(cudaEventDestroy(attention_ready337)!=cudaSuccess)std::terminate();early337::drop(16);}
            if(layer_done&&cudaEventDestroy(layer_done)!=cudaSuccess)std::terminate();
        }
    };
    pass558::Lease*pass_upload558(int layer){
        if(!pass_upload_enabled558)return nullptr;
        if(!plan528.armed)throw std::logic_error("558 mandatory flag requires prepared528 geometry");
        if(plan528.mode==4)return nullptr;
        if(plan528.mode!=2&&plan528.mode!=3)throw std::logic_error("558 no ordinary pipeline fallback");
        if(!active_pass317)throw std::logic_error("558 requires actual live317 pass");
        auto&q=*active_pass317;check_pass317(q);
        const char*pipeline=std::getenv("DS41_NATIVE_PREFILL_PIPELINE");
        if(!pipeline||std::strcmp(pipeline,"1")||layer!=q.next_layer||!paired_layout145||!output16_compact_v252||!direct_fp8_expert_weights||!active_fp8_experts)
            throw std::logic_error("558 original paired139 FP8 pipeline and ordered317 layer required");
        const auto typed=pair145::info(*pack.native()->weights());
        if(typed[0]!=145||typed[1]!=139||typed[2]!=46080||!typed[4]||!typed[5])throw std::logic_error("558 actual immutable139 generation/format required");
        const int device=headless320?1:0;
        // Lazy first-MoE allocation preserves the old layer's allocation phase.
        // The317 pass owns it until finish/abort; this Engine owns no global rank pointer.
        if(!q.upload558)q.upload558=std::make_unique<pass558::Lease>(pf.native_packed_bytes,device,plan528.epoch,q.epoch,typed[4],typed[5]);
        q.upload558->validate(pf.native_packed_bytes,device,plan528.epoch,q.epoch,typed[4],typed[5]);
        return q.upload558.get();
    }
    void check_pass317(const PrefillPassState317&q)const{
        int device=-1;ck(cudaGetDevice(&device),"317 current pass device");
        if(&q.owner!=this||active_pass317!=&q||!q.bound||q.finished||device!=q.device||device!=(headless320?1:0)||q.issuer!=std::this_thread::get_id())throw std::logic_error("317 live sameEngine/device/issuer pass required");
        if(active_fp8_experts!=q.fp8_scope.get()||active_dense_prefill!=q.dense_scope.get()||pf.cap!=fixed335.cap||pf.sub!=fixed335.B||pf.native_expert_batch!=fixed335.E||q.scratch_identity!=pf.own_scratch||q.hc_identity!=pf.h||q.routed_identity!=pf.routed)throw std::logic_error("317 pass feature/scratch owner changed");
    }
    std::unique_ptr<PrefillPassState317>prefill_pass_begin317(const std::vector<int>&input,int first,int count,int position,std::vector<float>*loss){
        // First admitted slice is ORIGINAL fresh512 two-pass geometry only.
        // No silent fallback/new chunking when a caller requests this seam.
        if(active_pass317||active_fp8_experts||active_dense_prefill||!native_math320()||host_native_prefill||!gpu_native_prefill||max_seq!=524288||pf.cap!=fixed335.cap||pf.sub!=fixed335.B||pf.native_expert_batch!=fixed335.E||loss||input.size()<=size_t(fixed335.split)||input.size()>size_t(fixed335.context)||(plan528.armed&&(plan528.mode==2||plan528.mode==3)?!fixed335.pass(first,count,position,int(input.size())):((first!=0&&first!=fixed335.split)||position!=first||count!=std::min(fixed335.cap,int(input.size())-first)))||count<=0||history.size()!=size_t(position))throw std::invalid_argument("317 requires one original512 pass withB1024/E2048/cap262144 andfresh ordered history");
        auto state=std::make_unique<PrefillPassState317>(*this,input,first,count,position,loss);
        ck(cudaGetDevice(&state->device),"317 begin device");if(state->device!=(headless320?1:0))throw std::invalid_argument("317/320 pass device differs from owning role");
        state->scratch_identity=pf.own_scratch;state->hc_identity=pf.h;state->routed_identity=pf.routed;state->epoch=++pass_epoch317;active_pass317=state.get();state->bound=true;prefill317_detail::bump(1);prefill317_detail::bump(5);
        prefill317_detail::set(6,uint64_t(first));prefill317_detail::set(7,uint64_t(count));prefill317_detail::set(8,uint64_t(position));
        ck(cudaEventCreateWithFlags(&state->layer_done,cudaEventDisableTiming),"317 pass layer event");
        if(early_attention337){ck(cudaEventCreateWithFlags(&state->attention_ready337,cudaEventDisableTiming),"337 separate attention-only event");early337::add(16);}
        auto&tokens=state->tokens;const int c0=state->c0,S=state->S,p0=state->p0;auto*nll=state->nll;
        auto&B=state->B;auto&suffix_first=state->suffix_first;auto&fp8_scope=state->fp8_scope;auto&dense_scope=state->dense_scope;
        auto&eng_table=state->eng_table;auto&eng_done=state->eng_done;auto&eng_ready=state->eng_ready;auto&eng_thread_ms=state->eng_thread_ms;auto&eng_thread=state->eng_thread;
        auto&later_jobs=state->later_jobs;auto&later_pushed=state->later_pushed;auto&eng_i=state->eng_i;auto&streamed_dense_v191=state->streamed_dense_v191;

        B = pf.sub;
        suffix_first=(decoder_suffix_prefill&&native_math320()&&!host_native_prefill&&!nll&&S>8192)?((S-4096)/B)*B:0;
        suffix_candidate_first_r1=suffix_first;
        if(suffix_candidate_active_r1&&nll)throw std::logic_error("NLL must retain full candidate rows");
        native_first_prefill_row=0;
        if(suffix_first)++decoder_suffix_passes;
        if(native_math320()){if(const char*mode=std::getenv("DS41_NATIVE_FP8_EXPERTS")){if(std::strcmp(mode,"1")==0){if(!decoded_native_expert_cache)throw std::invalid_argument("FP8 experts require decoded cache");fp8_scope=std::make_unique<NativeFp8Experts>(pf.native_expert_batch);}}}
        active_fp8_experts=fp8_scope.get();
        if(native_math320()){if(const char*mode=std::getenv("DS41_NATIVE_DENSE_PREFILL")){if(std::strcmp(mode,"1")==0)dense_scope=std::make_unique<NativeDensePrefill>(size_t(active_fp8_experts?384:512)*1024*1024,pf.sub);}}
        active_dense_prefill=dense_scope.get();
        native_prefill_position=p0;
        for(int i=0;i<S;++i)history.push_back(native_images&&native_images->contains(p0+i)?-1:pack.engram_hash().token_map[tokens[c0+i]]);
        eng_table = (size_t) pf.cap * kEngRows * engram_row_storage();   // pinned bytes per engram table
        // the engram rows are read on their own thread while the GPU works; a layer waits for its table only
        eng_done=std::vector<std::promise<void>>(n_eng);
        for (auto& d : eng_done) eng_ready.push_back(d.get_future());
        eng_thread_ms = 0;
        if(!host_native_prefill)eng_thread=std::thread([this,owned=state.get()] { engram_pass(owned->S,owned->p0,owned->eng_done,owned->eng_thread_ms); });
        for (int i = 0; i < S; ++i) pfh.tok[i] = tokens[c0 + i];
        ck(cudaMemcpyAsync(pf.tok, pfh.tok, (size_t) S * 4, cudaMemcpyHostToDevice, 0), "tokens up");
        ck(cudaMemcpyAsync(pf.pre_mix, pfh.pre, (size_t) S * kHc * 4, cudaMemcpyHostToDevice, 0), "pre_mix");
        if(host_native_prefill){
            for(int b0=0;b0<S;b0+=B){const int T=std::min(B,S-b0);
                prefill::embed_rows(embed,pf.tok+b0,T,pf.h);native_image_embed(pf.h,p0+b0,T);
                ck(cudaMemcpy(native_host->hc.data()+size_t(b0)*kHc*kDim,pf.h,size_t(T)*kHc*kDim*2,cudaMemcpyDeviceToHost),"host initialHC down");
            }
        }else {prefill::embed_rows(embed,pf.tok,S,pf.h);native_image_embed(pf.h,p0,S);}
        // stream mode: with this many tokens nearly every expert is routed in every layer, so the stream starts
        // with all non-resident experts of all layers before any routing is known (upstream's stream_all)
        // The first engram table is needed at layer 1; its rows are small random reads that lose most of their
        // throughput when the expert stream's large reads share the SSD (measured at 32K: 10.3 s alone, 14.9 s
        // shared). So the stream starts with layer 0's experts only and gets the rest once that table is read.
        if (!native_math320() && S >= kStreamAll) {
            layer_first_job.assign(kLayers + 1, 0);
            std::vector<std::pair<int, int>> jobs;
            for (int l = 0; l < kLayers; ++l) {
                for (int e = 0; e < kExperts; ++e)
                    if (!resident(l, e)) jobs.push_back({l, e});
                layer_first_job[l + 1] = (int64_t) jobs.size();
            }
            const size_t now = n_eng ? (size_t) layer_first_job[1] : jobs.size();
            later_jobs.assign(jobs.begin() + now, jobs.end());
            jobs.resize(now);
            const int64_t first = estream->push(jobs);   // later_jobs follow in the same numbering
            for (auto& j : layer_first_job) j += first;
        }
        later_pushed = later_jobs.empty();
        eng_i = 0;
        streamed_dense_v191 = pack.dense_stream_active_v190();
        if (streamed_dense_v191) pack.prefetch_dense_layer_v190(0);
        prefill317_detail::set(9,uint64_t(suffix_first));prefill317_detail::set(10,0);
        return state;
    }
    void prefill_pass_advance317(PrefillPassState317&owned,int l){
        check_pass317(owned);if(l!=owned.next_layer||l<0||l>=kLayers)throw std::logic_error("317 original nextlayer order required");
        auto*state=&owned;
        if(early_attention337){if(!state->attention_ready337||!state->early_callback337)throw std::logic_error("337 enabled path requires typed early callback");state->early_phase337.begin_attention(l);}
        auto&tokens=state->tokens;const int c0=state->c0,S=state->S,p0=state->p0;auto*nll=state->nll;
        auto&B=state->B;auto&suffix_first=state->suffix_first;auto&fp8_scope=state->fp8_scope;auto&dense_scope=state->dense_scope;
        auto&eng_table=state->eng_table;auto&eng_done=state->eng_done;auto&eng_ready=state->eng_ready;auto&eng_thread_ms=state->eng_thread_ms;auto&eng_thread=state->eng_thread;
        auto&later_jobs=state->later_jobs;auto&later_pushed=state->later_pushed;auto&eng_i=state->eng_i;auto&streamed_dense_v191=state->streamed_dense_v191;

            if (streamed_dense_v191) {
                pack.activate_dense_layer_v190(l, nullptr);
                bind_layer_weights_v191(l);
                if (l + 1 < kLayers) pack.prefetch_dense_layer_v190(l + 1);
            }
            native_first_prefill_row=l>kCandidateLayer?suffix_first:0;
            if(native_first_prefill_row){
                if(is_kv_source(l)||is_engram_layer(l))throw std::logic_error("Suffix cannot drop global state source");
                ++decoder_suffix_layers;decoder_suffix_rows_skipped+=native_first_prefill_row;
            }
            if(active_dense_prefill)active_dense_prefill->next_layer();
            auto& y = L[l];
            for (int b0 = native_first_prefill_row; b0 < S; b0 += B) {
                const int T = std::min(B, S - b0);
                bf16* h = pf.h + prefill_h_offset(b0);
                bf16* h2 = pf.h2 + prefill_h_offset(b0);
                if(host_native_prefill)ck(cudaMemcpy(h,native_host->hc.data()+size_t(b0)*kHc*kDim,size_t(T)*kHc*kDim*2,cudaMemcpyHostToDevice),"host HC up");
                if(host_native_prefill && is_engram_layer(l)){
                    std::vector<int64_t>rows;rows.reserve(size_t(T)*kEngRows);
                    for(int i=0;i<T;i++){engram_ids(l,eng_i,p0+b0+i);rows.insert(rows.end(),eng_ids[eng_i].begin(),eng_ids[eng_i].begin()+kEngRows);}
                    auto values=pack.native()->rows()->gather_bf16("layers."+std::to_string(l)+".engram.embed.weight",rows);
                    ck(cudaMemcpy(pf.eng_vals,values.data(),values.size()*2,cudaMemcpyHostToDevice),"host bounded Engram up");
                    fp8_rows(pf.eng_vals,T,y.eng_wkv,pf.eng_kv);ops::engram_apply(h,pf.eng_kv,y.eng_qw,y.eng_kw,kNormEps,T);
                    ptm->engram_rows+=rows.size();
                }
                else if (is_engram_layer(l)) {   // this sub-batch's rows of this table, pinned -> device
                    pfp.mark(PfProfile::kEngram);
                    if (b0 == 0) {
                        const double w0 = now_ms();
                        eng_ready[eng_i].get();   // rethrows a read error
                        ptm->engram_ms += now_ms() - w0;
                        if (!later_pushed) {      // the first table is in: the SSD goes to the expert stream
                            estream->push(later_jobs);
                            later_pushed = true;
                        }
                    }
                    const uint8_t* hw = pfh.eng + (size_t) eng_i * eng_table;
                    uint8_t* dw = pf.eng_dev;
                    const size_t rows = (size_t) T * kEngRows, r0 = (size_t) b0 * kEngRows;
                    if(native_math320()){ck(cudaMemcpyAsync(pf.eng_vals,hw+r0*512,rows*512,cudaMemcpyHostToDevice,0),"native prefill Engram rows");}
                    else {
                    ck(cudaMemcpyAsync(dw, hw + r0 * 256, rows * 256, cudaMemcpyHostToDevice, 0), "engram rows");
                    ck(cudaMemcpyAsync(dw + rows * 256, hw + (size_t) pf.cap * kEngRows * 256 + r0 * 8, rows * 8,
                                       cudaMemcpyHostToDevice, 0), "engram scales");
                    ops::engram_dequant(dw, dw + rows * 256, (int) rows, pf.eng_vals);
                    }
                    fp8_rows(pf.eng_vals, T, y.eng_wkv, pf.eng_kv);
                    if(native_images){
                        for(auto run:native_images->runs(p0+b0,T))if(!run.image)
                            ops::engram_apply(h+size_t(run.first)*kHc*kDim,pf.eng_kv+size_t(run.first)*(kHc+1)*kDim,y.eng_qw,y.eng_kw,kNormEps,run.count);
                    }else ops::engram_apply(h,pf.eng_kv,y.eng_qw,y.eng_kw,kNormEps,T);
                }
                pfp.mark(PfProfile::kHc);
                if(native_capture_main&&l>=37)native_main_capture(reinterpret_cast<const uint16_t*>(h),T,p0+b0,l-37,native_main_ring);
                hc_rows(h, T, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, pf.pre_mix + b0 * kHc, pf.xa, pf.attn_pre,
                        pf.attn_post, pf.attn_comb);
                ops::rmsnorm(pf.xa, y.attn_norm, pf.xa, kDim, kNormEps, T);
                attention_rows(l, T, p0 + b0, b0);
                pfp.mark(PfProfile::kHc);
                prefill_hc_post(pf.attn_out, h, pf.attn_post, pf.attn_comb, h2, T);
                bf16* xf = prefill_ffn_input(b0);
                hc_rows(h2, T, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, pf.attn_pre, xf, pf.ffn_pre + b0 * kHc,
                        pf.ffn_post + b0 * kHc, pf.ffn_comb + b0 * kHc * kHc);
                if(recompute_inputs_active_r1){
                    ck(cudaMemcpyAsync(pf.recompute_pre_r1+size_t(b0)*kHc,pf.attn_pre,size_t(T)*kHc*sizeof(float),cudaMemcpyDeviceToDevice,0),"recompute input collapse coefficients");
                    ds41_recompute_norm_r1(reinterpret_cast<const uint16_t*>(xf),reinterpret_cast<const uint16_t*>(y.ffn_norm),reinterpret_cast<uint16_t*>(xf),pf.recompute_inverse_r1+b0,T);
                }else ops::rmsnorm(xf, y.ffn_norm, xf, kDim, kNormEps, T);
                moe_route(l,T,b0);
                if(host_native_prefill)ck(cudaMemcpy(native_host->hc.data()+size_t(b0)*kHc*kDim,h2,size_t(T)*kHc*kDim*2,cudaMemcpyDeviceToHost),"host attentionHC down");
            }
            if(early_attention337){
                const int rank=headless320?1:0;
                ck(cudaEventRecord(state->attention_ready337,0),"337 attention/router ready BEFORE experts");state->early_phase337.attention_ready(l);early337::add(2+rank);early337::add(14);early337::set(20+rank,l+1);
                AttentionBoundary337 b;b.epoch=state->epoch;b.layer=l;b.absolute_end=p0+S;b.ratio=y.ratio;b.carry=y.ratio>1?b.absolute_end%y.ratio:0;
                b.source_layer=y.ratio?(is_kv_source(l)?l:state->source_layer):-1;b.ready=state->attention_ready337;b.window=y.window;b.window_bytes=size_t(kWindow)*kHeadDim*2;
                if(y.ratio){const auto&source=L[b.source_layer];b.kv_state=source.kv_state;b.score_state=source.score_state;b.comp=cur_comp;b.index_keys=cur_index_k;b.compressed_rows=b.absolute_end/y.ratio;b.comp_bytes=size_t(b.compressed_rows)*kHeadDim*2;b.index_bytes=size_t(b.compressed_rows)*kIndexDim*2;b.carry_bytes=size_t(b.carry)*kHeadDim*sizeof(float);}
                early337::add(13);
                try{state->early_callback337(b);early337::drop(13);}catch(...){early337::drop(13);throw;}
                state->early_phase337.acknowledge(l);early337::drop(14);early337::add(10+rank);
                state->early_phase337.enter_ffn(l);early337::add(8+rank);
            }
            if (is_engram_layer(l)) ++eng_i;
            moe_experts(l, S);
            for(int b0=native_first_prefill_row;b0<S;b0+=B){const int T=std::min(B,S-b0);
                if(host_native_prefill){
                    ck(cudaMemcpy(pf.h2,native_host->hc.data()+size_t(b0)*kHc*kDim,size_t(T)*kHc*kDim*2,cudaMemcpyHostToDevice),"host residualHC up");
                    ck(cudaMemcpy(pf.routed,native_host->routed.data()+size_t(b0)*kDim,size_t(T)*kDim*4,cudaMemcpyHostToDevice),"host routed up");
                }
                moe_finish(l,T,b0);
                if(host_native_prefill)ck(cudaMemcpy(native_host->hc.data()+size_t(b0)*kHc*kDim,pf.h,size_t(T)*kHc*kDim*2,cudaMemcpyDeviceToHost),"host nextHC down");
            }
            // Default-stream retire event follows all weight consumers, including
            // shared-expert moe_finish. The alternate bank may only reuse it afterwards.
            if (streamed_dense_v191) pack.retire_dense_layer_v190(l, nullptr);
        
        // Event publication is NOT a host readiness claim. A future coordinator
        // must await/copy/ACK this epoch BEFORE reusing/re-recording the event.
        ck(cudaEventRecord(state->layer_done,0),"317 original layer boundary");
        if(is_kv_source(l))state->source_layer=l;state->next_layer=l+1;
        prefill317_detail::bump(2);prefill317_detail::bump(11);prefill317_detail::set(10,uint64_t(state->next_layer));
        if(early_attention337)state->early_phase337.full_layer_enqueued(l);
    }
    PrefillBoundary317 prefill_pass_boundary317(const PrefillPassState317&q)const{
        check_pass317(q);if(q.next_layer<1)throw std::logic_error("317 no completed layer enqueue");
        const int l=q.next_layer-1;const auto&y=L[l];PrefillBoundary317 v;
        v.epoch=q.epoch;v.layer=l;v.next_layer=q.next_layer;v.c0=q.c0;v.S=q.S;v.p0=q.p0;v.suffix_first=q.suffix_first;v.absolute_end=q.p0+q.S;v.ratio=y.ratio;v.carry=y.ratio>1?v.absolute_end%y.ratio:0;v.source_layer=y.ratio?q.source_layer:-1;v.ready=q.layer_done;
        v.window=y.window;v.window_bytes=size_t(kWindow)*kHeadDim*2;
        if(y.ratio){const auto&source=L[v.source_layer];v.comp=cur_comp;v.index_keys=cur_index_k;v.kv_state=source.kv_state;v.score_state=source.score_state;v.compressed_rows=v.absolute_end/y.ratio;v.comp_bytes=size_t(v.compressed_rows)*kHeadDim*2;v.index_bytes=size_t(v.compressed_rows)*kIndexDim*2;v.carry_bytes=size_t(v.carry)*kHeadDim*sizeof(float);}
        return v; // borrowed pointers, no P2P/copy/visibility or global publication
    }
    void prefill_pass_finish317(PrefillPassState317&owned){
        check_pass317(owned);if(owned.next_layer!=kLayers)throw std::logic_error("317 all40 original layers required before head");
        auto*state=&owned;
        auto&tokens=state->tokens;const int c0=state->c0,S=state->S,p0=state->p0;auto*nll=state->nll;
        auto&B=state->B;auto&suffix_first=state->suffix_first;auto&fp8_scope=state->fp8_scope;auto&dense_scope=state->dense_scope;
        auto&eng_table=state->eng_table;auto&eng_done=state->eng_done;auto&eng_ready=state->eng_ready;auto&eng_thread_ms=state->eng_thread_ms;auto&eng_thread=state->eng_thread;
        auto&later_jobs=state->later_jobs;auto&later_pushed=state->later_pushed;auto&eng_i=state->eng_i;auto&streamed_dense_v191=state->streamed_dense_v191;

        const int n = (int) tokens.size();
        if (nll) {
            for (int i = 0; i < S; ++i) pfh.targets[i] = c0 + i + 1 < n ? tokens[c0 + i + 1] : -1;
            ck(cudaMemcpyAsync(pf.targets, pfh.targets, (size_t) S * 4, cudaMemcpyHostToDevice, 0), "targets up");
        }
        pfp.mark(PfProfile::kHead);
        for (int b0 = native_first_prefill_row; b0 < S; b0 += B) {
            const int T = std::min(B, S - b0);
            if(host_native_prefill)ck(cudaMemcpy(pf.h,native_host->hc.data()+size_t(b0)*kHc*kDim,size_t(T)*kHc*kDim*2,cudaMemcpyHostToDevice),"host finalHC up");
            ops::hc_pre(pf.h + prefill_h_offset(b0), pf.pre_mix + b0 * kHc, pf.final_x, T);
            ops::rmsnorm(pf.final_x, final_norm, pf.final_x, kDim, kNormEps, T);
            if (nll)
                for (int r0 = 0; r0 < T; r0 += kNllRows) {
                    const int nb = std::min(kNllRows, T - r0);
                    prefill::bf16_gemm(pf.final_x + (size_t) r0 * kDim, head, nb, kDim, kVocab, nullptr, pf.nll_logits);
                    prefill::nll_rows(pf.nll_logits, nb, kVocab, pf.targets + b0 + r0, pf.nll_out + b0 + r0);
                }
            if (c0 + b0 + T == n) {   // the last token's logits, as step() computes them
                if(headless320){
                    headless_final320.resize(kDim);
                    ck(cudaMemcpy(headless_final320.data(),pf.final_x+size_t(T-1)*kDim,kDim*2,cudaMemcpyDeviceToHost),"320 final normalized BF16 row, no head/logits");
                }else{
                ops::bf16_linear(pf.final_x + (size_t) (T - 1) * kDim, nullptr, head, kDim, kVocab, nullptr, logits);
                lg.resize(kVocab);
                ck(cudaMemcpy(lg.data(), logits, kVocab * 4, cudaMemcpyDeviceToHost), "logits");
                }
            }
        }
        if (nll) {
            ck(cudaMemcpy(pfh.nll, pf.nll_out, (size_t) S * 4, cudaMemcpyDeviceToHost), "nll down");
            for (int i = 0; i < S; ++i)
                if (c0 + i + 1 < n) (*nll)[c0 + i] = pfh.nll[i];
        }
        pfp.mark(PfProfile::kHead);
        pfp.report();
        if (estream && S >= kStreamAll) estream->drain();   // every job of the pass was consumed
        if(native_math320()){
            // Sample once per native pass, before the dense/FP8 scopes release
            // their feature buffers. This is an end-pass sample, not a peak proof.
            ck(cudaDeviceSynchronize(),"native prefill memory sample synchronization");
            size_t free=0,total=0;
            ck(cudaMemGetInfo(&free,&total),"native prefill memory sample");
            native_prefill_minimum_free_bytes=std::min(native_prefill_minimum_free_bytes,uint64_t(free));
            ++native_prefill_sampled_passes;
        }
    
        state->finished=true;prefill317_detail::bump(3);
    }
    void prefill_pass(const std::vector<int>&tokens,int c0,int S,int p0,std::vector<float>*nll){
        if(!pass_seam317){prefill_pass_literal317(tokens,c0,S,p0,nll);return;}
        auto state=prefill_pass_begin317(tokens,c0,S,p0,nll);
        for(int l=0;l<kLayers;l++){
            prefill_pass_advance317(*state,l);
            const auto boundary=prefill_pass_boundary317(*state);
            if(boundary.layer!=l||boundary.next_layer!=l+1||boundary.absolute_end!=p0+S||!boundary.ready||!boundary.window||(boundary.ratio&&(!boundary.comp||!boundary.index_keys||boundary.source_layer<0)))throw std::logic_error("317 literal pass boundary certificate differs");
        }
        prefill_pass_finish317(*state);
    }

    void prefill_pass_literal317(const std::vector<int>& tokens, int c0, int S, int p0, std::vector<float>* nll) {
        const int B = pf.sub;
        const int suffix_first=(decoder_suffix_prefill&&native_math320()&&!host_native_prefill&&!nll&&S>8192)?((S-4096)/B)*B:0;
        suffix_candidate_first_r1=suffix_first;
        if(suffix_candidate_active_r1&&nll)throw std::logic_error("NLL must retain full candidate rows");
        native_first_prefill_row=0;
        struct ResetSuffix {int&row;~ResetSuffix(){row=0;}}reset_suffix{native_first_prefill_row};
        if(suffix_first)++decoder_suffix_passes;
        std::unique_ptr<NativeFp8Experts>fp8_scope;
        if(native_math320()){if(const char*mode=std::getenv("DS41_NATIVE_FP8_EXPERTS")){if(std::strcmp(mode,"1")==0){if(!decoded_native_expert_cache)throw std::invalid_argument("FP8 experts require decoded cache");fp8_scope=std::make_unique<NativeFp8Experts>(pf.native_expert_batch);}}}
        active_fp8_experts=fp8_scope.get();
        struct ClearFp8 {NativeFp8Experts*&p;~ClearFp8(){p=nullptr;}}clear_fp8{active_fp8_experts};
        std::unique_ptr<NativeDensePrefill>dense_scope;
        if(native_math320()){if(const char*mode=std::getenv("DS41_NATIVE_DENSE_PREFILL")){if(std::strcmp(mode,"1")==0)dense_scope=std::make_unique<NativeDensePrefill>(size_t(active_fp8_experts?384:512)*1024*1024,pf.sub);}}
        active_dense_prefill=dense_scope.get();
        struct ClearDense {NativeDensePrefill*&p;~ClearDense(){p=nullptr;}}clear_dense{active_dense_prefill};
        native_prefill_position=p0;
        for(int i=0;i<S;++i)history.push_back(native_images&&native_images->contains(p0+i)?-1:pack.engram_hash().token_map[tokens[c0+i]]);
        const size_t eng_table = (size_t) pf.cap * kEngRows * engram_row_storage();   // pinned bytes per engram table
        // the engram rows are read on their own thread while the GPU works; a layer waits for its table only
        std::vector<std::promise<void>> eng_done(n_eng);
        std::vector<std::future<void>> eng_ready;
        for (auto& d : eng_done) eng_ready.push_back(d.get_future());
        double eng_thread_ms = 0;
        std::thread eng_thread;
        if(!host_native_prefill)eng_thread=std::thread([&] { engram_pass(S,p0,eng_done,eng_thread_ms); });
        struct Join {
            std::thread& t;
            ~Join() { if (t.joinable()) t.join(); }
        } eng_join{eng_thread};
        for (int i = 0; i < S; ++i) pfh.tok[i] = tokens[c0 + i];
        ck(cudaMemcpyAsync(pf.tok, pfh.tok, (size_t) S * 4, cudaMemcpyHostToDevice, 0), "tokens up");
        ck(cudaMemcpyAsync(pf.pre_mix, pfh.pre, (size_t) S * kHc * 4, cudaMemcpyHostToDevice, 0), "pre_mix");
        if(host_native_prefill){
            for(int b0=0;b0<S;b0+=B){const int T=std::min(B,S-b0);
                prefill::embed_rows(embed,pf.tok+b0,T,pf.h);native_image_embed(pf.h,p0+b0,T);
                ck(cudaMemcpy(native_host->hc.data()+size_t(b0)*kHc*kDim,pf.h,size_t(T)*kHc*kDim*2,cudaMemcpyDeviceToHost),"host initialHC down");
            }
        }else {prefill::embed_rows(embed,pf.tok,S,pf.h);native_image_embed(pf.h,p0,S);}
        // stream mode: with this many tokens nearly every expert is routed in every layer, so the stream starts
        // with all non-resident experts of all layers before any routing is known (upstream's stream_all)
        // The first engram table is needed at layer 1; its rows are small random reads that lose most of their
        // throughput when the expert stream's large reads share the SSD (measured at 32K: 10.3 s alone, 14.9 s
        // shared). So the stream starts with layer 0's experts only and gets the rest once that table is read.
        std::vector<std::pair<int, int>> later_jobs;
        if (!native_math320() && S >= kStreamAll) {
            layer_first_job.assign(kLayers + 1, 0);
            std::vector<std::pair<int, int>> jobs;
            for (int l = 0; l < kLayers; ++l) {
                for (int e = 0; e < kExperts; ++e)
                    if (!resident(l, e)) jobs.push_back({l, e});
                layer_first_job[l + 1] = (int64_t) jobs.size();
            }
            const size_t now = n_eng ? (size_t) layer_first_job[1] : jobs.size();
            later_jobs.assign(jobs.begin() + now, jobs.end());
            jobs.resize(now);
            const int64_t first = estream->push(jobs);   // later_jobs follow in the same numbering
            for (auto& j : layer_first_job) j += first;
        }
        bool later_pushed = later_jobs.empty();
        int eng_i = 0;
        const bool streamed_dense_v191 = pack.dense_stream_active_v190();
        if (streamed_dense_v191) pack.prefetch_dense_layer_v190(0);
        for (int l = 0; l < kLayers; ++l) {
            if (streamed_dense_v191) {
                pack.activate_dense_layer_v190(l, nullptr);
                bind_layer_weights_v191(l);
                if (l + 1 < kLayers) pack.prefetch_dense_layer_v190(l + 1);
            }
            native_first_prefill_row=l>kCandidateLayer?suffix_first:0;
            if(native_first_prefill_row){
                if(is_kv_source(l)||is_engram_layer(l))throw std::logic_error("Suffix cannot drop global state source");
                ++decoder_suffix_layers;decoder_suffix_rows_skipped+=native_first_prefill_row;
            }
            if(active_dense_prefill)active_dense_prefill->next_layer();
            auto& y = L[l];
            for (int b0 = native_first_prefill_row; b0 < S; b0 += B) {
                const int T = std::min(B, S - b0);
                bf16* h = pf.h + prefill_h_offset(b0);
                bf16* h2 = pf.h2 + prefill_h_offset(b0);
                if(host_native_prefill)ck(cudaMemcpy(h,native_host->hc.data()+size_t(b0)*kHc*kDim,size_t(T)*kHc*kDim*2,cudaMemcpyHostToDevice),"host HC up");
                if(host_native_prefill && is_engram_layer(l)){
                    std::vector<int64_t>rows;rows.reserve(size_t(T)*kEngRows);
                    for(int i=0;i<T;i++){engram_ids(l,eng_i,p0+b0+i);rows.insert(rows.end(),eng_ids[eng_i].begin(),eng_ids[eng_i].begin()+kEngRows);}
                    auto values=pack.native()->rows()->gather_bf16("layers."+std::to_string(l)+".engram.embed.weight",rows);
                    ck(cudaMemcpy(pf.eng_vals,values.data(),values.size()*2,cudaMemcpyHostToDevice),"host bounded Engram up");
                    fp8_rows(pf.eng_vals,T,y.eng_wkv,pf.eng_kv);ops::engram_apply(h,pf.eng_kv,y.eng_qw,y.eng_kw,kNormEps,T);
                    ptm->engram_rows+=rows.size();
                }
                else if (is_engram_layer(l)) {   // this sub-batch's rows of this table, pinned -> device
                    pfp.mark(PfProfile::kEngram);
                    if (b0 == 0) {
                        const double w0 = now_ms();
                        eng_ready[eng_i].get();   // rethrows a read error
                        ptm->engram_ms += now_ms() - w0;
                        if (!later_pushed) {      // the first table is in: the SSD goes to the expert stream
                            estream->push(later_jobs);
                            later_pushed = true;
                        }
                    }
                    const uint8_t* hw = pfh.eng + (size_t) eng_i * eng_table;
                    uint8_t* dw = pf.eng_dev;
                    const size_t rows = (size_t) T * kEngRows, r0 = (size_t) b0 * kEngRows;
                    if(native_math320()){ck(cudaMemcpyAsync(pf.eng_vals,hw+r0*512,rows*512,cudaMemcpyHostToDevice,0),"native prefill Engram rows");}
                    else {
                    ck(cudaMemcpyAsync(dw, hw + r0 * 256, rows * 256, cudaMemcpyHostToDevice, 0), "engram rows");
                    ck(cudaMemcpyAsync(dw + rows * 256, hw + (size_t) pf.cap * kEngRows * 256 + r0 * 8, rows * 8,
                                       cudaMemcpyHostToDevice, 0), "engram scales");
                    ops::engram_dequant(dw, dw + rows * 256, (int) rows, pf.eng_vals);
                    }
                    fp8_rows(pf.eng_vals, T, y.eng_wkv, pf.eng_kv);
                    if(native_images){
                        for(auto run:native_images->runs(p0+b0,T))if(!run.image)
                            ops::engram_apply(h+size_t(run.first)*kHc*kDim,pf.eng_kv+size_t(run.first)*(kHc+1)*kDim,y.eng_qw,y.eng_kw,kNormEps,run.count);
                    }else ops::engram_apply(h,pf.eng_kv,y.eng_qw,y.eng_kw,kNormEps,T);
                }
                pfp.mark(PfProfile::kHc);
                if(native_capture_main&&l>=37)native_main_capture(reinterpret_cast<const uint16_t*>(h),T,p0+b0,l-37,native_main_ring);
                hc_rows(h, T, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, pf.pre_mix + b0 * kHc, pf.xa, pf.attn_pre,
                        pf.attn_post, pf.attn_comb);
                ops::rmsnorm(pf.xa, y.attn_norm, pf.xa, kDim, kNormEps, T);
                attention_rows(l, T, p0 + b0, b0);
                pfp.mark(PfProfile::kHc);
                prefill_hc_post(pf.attn_out, h, pf.attn_post, pf.attn_comb, h2, T);
                bf16* xf = prefill_ffn_input(b0);
                hc_rows(h2, T, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, pf.attn_pre, xf, pf.ffn_pre + b0 * kHc,
                        pf.ffn_post + b0 * kHc, pf.ffn_comb + b0 * kHc * kHc);
                if(recompute_inputs_active_r1){
                    ck(cudaMemcpyAsync(pf.recompute_pre_r1+size_t(b0)*kHc,pf.attn_pre,size_t(T)*kHc*sizeof(float),cudaMemcpyDeviceToDevice,0),"recompute input collapse coefficients");
                    ds41_recompute_norm_r1(reinterpret_cast<const uint16_t*>(xf),reinterpret_cast<const uint16_t*>(y.ffn_norm),reinterpret_cast<uint16_t*>(xf),pf.recompute_inverse_r1+b0,T);
                }else ops::rmsnorm(xf, y.ffn_norm, xf, kDim, kNormEps, T);
                moe_route(l,T,b0);
                if(host_native_prefill)ck(cudaMemcpy(native_host->hc.data()+size_t(b0)*kHc*kDim,h2,size_t(T)*kHc*kDim*2,cudaMemcpyDeviceToHost),"host attentionHC down");
            }
            if (is_engram_layer(l)) ++eng_i;
            moe_experts(l, S);
            for(int b0=native_first_prefill_row;b0<S;b0+=B){const int T=std::min(B,S-b0);
                if(host_native_prefill){
                    ck(cudaMemcpy(pf.h2,native_host->hc.data()+size_t(b0)*kHc*kDim,size_t(T)*kHc*kDim*2,cudaMemcpyHostToDevice),"host residualHC up");
                    ck(cudaMemcpy(pf.routed,native_host->routed.data()+size_t(b0)*kDim,size_t(T)*kDim*4,cudaMemcpyHostToDevice),"host routed up");
                }
                moe_finish(l,T,b0);
                if(host_native_prefill)ck(cudaMemcpy(native_host->hc.data()+size_t(b0)*kHc*kDim,pf.h,size_t(T)*kHc*kDim*2,cudaMemcpyDeviceToHost),"host nextHC down");
            }
            // Default-stream retire event follows all weight consumers, including
            // shared-expert moe_finish. The alternate bank may only reuse it afterwards.
            if (streamed_dense_v191) pack.retire_dense_layer_v190(l, nullptr);
        }
        const int n = (int) tokens.size();
        if (nll) {
            for (int i = 0; i < S; ++i) pfh.targets[i] = c0 + i + 1 < n ? tokens[c0 + i + 1] : -1;
            ck(cudaMemcpyAsync(pf.targets, pfh.targets, (size_t) S * 4, cudaMemcpyHostToDevice, 0), "targets up");
        }
        pfp.mark(PfProfile::kHead);
        for (int b0 = native_first_prefill_row; b0 < S; b0 += B) {
            const int T = std::min(B, S - b0);
            if(host_native_prefill)ck(cudaMemcpy(pf.h,native_host->hc.data()+size_t(b0)*kHc*kDim,size_t(T)*kHc*kDim*2,cudaMemcpyHostToDevice),"host finalHC up");
            ops::hc_pre(pf.h + prefill_h_offset(b0), pf.pre_mix + b0 * kHc, pf.final_x, T);
            ops::rmsnorm(pf.final_x, final_norm, pf.final_x, kDim, kNormEps, T);
            if (nll)
                for (int r0 = 0; r0 < T; r0 += kNllRows) {
                    const int nb = std::min(kNllRows, T - r0);
                    prefill::bf16_gemm(pf.final_x + (size_t) r0 * kDim, head, nb, kDim, kVocab, nullptr, pf.nll_logits);
                    prefill::nll_rows(pf.nll_logits, nb, kVocab, pf.targets + b0 + r0, pf.nll_out + b0 + r0);
                }
            if (c0 + b0 + T == n) {   // the last token's logits, as step() computes them
                if(headless320){
                    headless_final320.resize(kDim);
                    ck(cudaMemcpy(headless_final320.data(),pf.final_x+size_t(T-1)*kDim,kDim*2,cudaMemcpyDeviceToHost),"320 final normalized BF16 row, no head/logits");
                }else{
                ops::bf16_linear(pf.final_x + (size_t) (T - 1) * kDim, nullptr, head, kDim, kVocab, nullptr, logits);
                lg.resize(kVocab);
                ck(cudaMemcpy(lg.data(), logits, kVocab * 4, cudaMemcpyDeviceToHost), "logits");
                }
            }
        }
        if (nll) {
            ck(cudaMemcpy(pfh.nll, pf.nll_out, (size_t) S * 4, cudaMemcpyDeviceToHost), "nll down");
            for (int i = 0; i < S; ++i)
                if (c0 + i + 1 < n) (*nll)[c0 + i] = pfh.nll[i];
        }
        pfp.mark(PfProfile::kHead);
        pfp.report();
        if (estream && S >= kStreamAll) estream->drain();   // every job of the pass was consumed
        if(native_math320()){
            // Sample once per native pass, before the dense/FP8 scopes release
            // their feature buffers. This is an end-pass sample, not a peak proof.
            ck(cudaDeviceSynchronize(),"native prefill memory sample synchronization");
            size_t free=0,total=0;
            ck(cudaMemGetInfo(&free,&total),"native prefill memory sample");
            native_prefill_minimum_free_bytes=std::min(native_prefill_minimum_free_bytes,uint64_t(free));
            ++native_prefill_sampled_passes;
        }
    }

    int prefill(const std::vector<int>& tokens, int pos, std::vector<float>* nll, PrefillTiming& pt) {
        usable();
        if (pos != (int) history.size()) throw std::runtime_error("prefill must continue at the tokens fed so far");
        const int n = (int) tokens.size();
        if (n == 0) throw std::runtime_error("prefill of no tokens");
        if (pos + n > max_seq) throw std::runtime_error("prefill past max_seq");
        for (int t : tokens) check_token(t);
        if(!serving516::enabled()||!plan528.armed||(plan528.mode!=2&&plan528.mode!=3&&plan528.mode!=4)||opt.prefill_chunk<=0)throw std::logic_error("534 mandatory prepared FP8/wavefront strategy required");
        append_plan534(n,pos,bool(nll));
        native_cache_tail_active=native_cpu&&!host_native_prefill&&native_gpu_budget&&native_cache_tail_rows&&opt.prefill_chunk>0;
        native_cache_tail_begin=native_cache_tail_end=0;
        native_cache_tail_counted_uses=0;
        if(native_cache_tail_active){
            // One global window for this whole call, including every scratch-sized pass.
            // All rows in this call are >=pos, so clamping a short prompt to pos
            // selects the same rows as the absolute cutoff pos+n-tail_rows.
            native_cache_tail_end=pos+n;
            native_cache_tail_begin=std::max(pos,native_cache_tail_end-native_cache_tail_rows);
            for(auto&counts:native_expert_frequency)counts.fill(0);
        }
        if (pack.dense_stream_enabled_v190() && opt.prefill_chunk <= 0)
            throw std::logic_error("191 streamed dense requires batched GPU-native prefill before any decode");
        if (opt.prefill_chunk <= 0) return prefill_steps(tokens, pos, nll, pt);
        const char*suffix_mode=std::getenv("DS41_SUFFIX_CANDIDATE_STORAGE_R1");
        suffix_candidate_active_r1=suffix_mode&&std::strcmp(suffix_mode,"1")==0&&!nll;
        const char*recompute_mode=std::getenv("DS41_RECOMPUTE_INPUTS_R1");
        recompute_inputs_active_r1=recompute_mode&&std::strcmp(recompute_mode,"1")==0&&!nll;
        pf_started = false;
        try {
            return prefill_batched(tokens, pos, nll, pt);
        } catch (...) {
            if (pf_started) broken = true;   // a pass may have written part of the layers' caches and the history
            throw;
        }
    }

    int prefill_steps(const std::vector<int>& tokens, int pos, std::vector<float>* nll, PrefillTiming& pt) {
        const int n = (int) tokens.size();
        pt = PrefillTiming{};
        ptm = &pt;
        const double t0 = now_ms();
        if (nll) nll->assign(n - 1, 0.0f);
        int next = -1;
        {   // token by token
            Timing tm;
            for (int i = 0; i < n; ++i) {
                next = step(tokens[i], pos + i, nullptr, tm);
                if (nll && i + 1 < n) {
                    double mx = -1e300, se = 0;
                    for (float v : lg) mx = std::max(mx, (double) v);
                    for (float v : lg) se += std::exp((double) v - mx);
                    (*nll)[i] = (float) (mx + std::log(se) - lg[tokens[i + 1]]);
                }
            }
            pt.total_ms = now_ms() - t0;
            return next;
        }
    }

    int prefill_batched(const std::vector<int>& tokens, int pos, std::vector<float>* nll, PrefillTiming& pt) {
        serving516::Scope serving_scope516(wavefront_latched325,pass_seam317,max_seq,
            int(tokens.size()),pos,history.empty(),bool(native_images),bool(nll),
            opt.prefill_chunk,opt.prefill_batch,native_host_expert_batch);

        std::unique_ptr<fast534::Call> fast_call534;
        if(mode4_534()){
            fast_call534=std::make_unique<fast534::Call>(plan528.epoch,max_seq,int(tokens.size()),pos,native_host_expert_batch);
            const char*v=std::getenv("DS41_NATIVE_PREFILL_PIPELINE");
            if(!pair531::enabled()||!v||std::strcmp(v,"1")||!native_math320()||!gpu_native_prefill||host_native_prefill||headless320||!paired_layout145||!direct_fp8_expert_weights||!output16_compact_v252||!decoded_native_expert_cache)
                throw std::logic_error("534 mandatory FP8 pipeline/531 typed139 configuration required");
        }

        serving516::Disable ordinary_copy516(parallel_headless_copy331,serving_scope516.fallback||(plan528.armed&&plan528.mode==3));
        serving516::Disable serial_early528(early_attention337,plan528.armed&&plan528.mode==3);
        if(wavefront_latched325&&!headless320)return prefill_wavefront_impl325(tokens,pos,nll,pt);
        const int n = (int) tokens.size();
        pt = PrefillTiming{};
        ptm = &pt;
        const double t0 = now_ms();
        if (nll) nll->assign(n - 1, 0.0f);
        bool dense_transition_v191 = false;
        std::unique_ptr<pair_indexer_v1::Owner> pair_indexer_scope_v1;
        try {
            if(mode4_534()&&pos>0&&pack.dense_stream_enabled_v190()){
                const auto dense534=pack.dense_stream_info_v190();
                if(pack.dense_stream_active_v190()||dense534[3]!=1||dense534[4]!=1||dense534[2]||dense534[8]||dense534[9]||dense534[23])throw std::logic_error("534 append requires fully restored stable resident dense weights");
                check_stable_dense_v191();
            }else if (pack.dense_stream_enabled_v190()) {
                // A failed begin may already have released the full layer arena.
                // Mark the Engine poisoned even if no KV pass has started yet.
                dense_transition_v191 = true;
                begin_streamed_dense_v191(pos);
            }
            prefill_begin(n);   // a partial failure returns what it took (lent slots, scratch, ring)

            if(mode4_534()){
                const auto typed534=pair145::info(*pack.native()->weights());
                if(typed534[0]!=145||typed534[1]!=139||typed534[2]!=46080||!typed534[4]||!typed534[5]||typed534[6]!=1||typed534[7])throw std::logic_error("534 requires actual complete139 bank");
                upload_lease534=std::make_unique<fast534::UploadLease>(pf.native_packed_bytes,0,plan528.epoch,typed534[4],typed534[5]);
                fast534::set(28,pf.cap);fast534::set(29,pf.sub);
            }

            if(!(plan528.armed&&plan528.mode==3))pair_indexer_scope_v1 = std::make_unique<pair_indexer_v1::Owner>(max_seq, pf.sub, pos, cudaStream_t{});
            active_pair_indexer_v1 = pair_indexer_scope_v1.get();
            pt.chunk_tokens = pf.cap;
            pt.sub_batch = pf.sub;
            if (fault("prefill")) throw std::runtime_error("ds41 test fault: before the first prefill pass");
            pf_started = true;
            if (fault("prefill_pass")) throw std::runtime_error("ds41 test fault: inside a prefill pass");
            for (int c0 = 0; c0 < n;) {
                const int count528=plan528.armed&&plan528.mode==3?(c0? n-fixed335.split:fixed335.split):std::min(pf.cap,n-c0);
                prefill_pass(tokens, c0, count528, pos + c0, nll);
                c0+=count528;
                ++pt.chunks;
            }
            if(estream){
            estream->drain();
            const auto st = estream->take_stats();
            pt.from_ram = st.from_ram;
            pt.from_cache = st.from_cache;
            pt.from_ssd = st.from_ssd;
            pt.stream_wait_ms = st.consumer_wait_ms;
            }
            active_pair_indexer_v1 = nullptr;
            if(pair_indexer_scope_v1)pair_indexer_scope_v1->finish();
            pair_indexer_scope_v1.reset();
        } catch (...) {
            active_pair_indexer_v1 = nullptr;
            pair_indexer_scope_v1.reset(); // drain before Engine prefill scratch is freed
            if (dense_transition_v191||mode4_534()) broken = true;
            close_upload534();
            prefill_end();
            throw;
        }
        close_upload534();
        prefill_end();
        if (dense_transition_v191) {
            try { restore_streamed_dense_v191(); }
            catch (...) { broken = true; throw; }
        }
        // Includes every streamed layer upload plus the full post-prefill restore.
        pt.total_ms = now_ms() - t0;
        if(fast_call534){fast534::set(26,pt.chunks);fast_call534->complete();}
        return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    }

#include "headless_session_v320.inc"
#include "wavefront_session_v325.inc"

    // ------------------------------------------------------------------------------------- step
    /// The GPU work of one decode step on stream st: every position dependent value comes from dp (staged from hp
    /// by the first node), so one capture serves every position with the same parity and indexer capacities.
    /// dump: eager only (it reads the device between layers).
    void enqueue_step(StepDump* dump) {
        ck(cudaMemcpyAsync(dp, hp, sizeof(StepParams), cudaMemcpyHostToDevice, st), "step params");
        if (n_eng)
            ck(cudaMemcpyAsync(eng_dev, eng_host, (size_t) n_eng * kEngRows * engram_row_storage(), cudaMemcpyHostToDevice, st),
               "engram rows");
        ops::window_index_device(dp + 1, idx_dev, st);
        ops::embed_device(embed, dp, h, st);
        ck(cudaMemcpyAsync(pre_mix, one_hot_dev, kHc * 4, cudaMemcpyDeviceToDevice, st), "pre_mix");
        int eng_i = 0;
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            if (is_engram_layer(l)) engram(l, eng_i++);
            if(native_capture_main&&l>=37)native_main_capture(reinterpret_cast<const uint16_t*>(h),1,0,l-37,native_main_step,st);
            const bool dbg_layer = dbg && (native_block_debug ? l < 4 : (l == 1 || l == 2));
            if (dbg_layer) dbg_write(h, kHc * kDim);                    // block input (after engram)
            // attention sub-block: h -> h2
            kernels::hc_mixes_pre(h, 1, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, pre_mix, xa, attn_pre, attn_post,
                                  attn_comb, st);
            ops::rmsnorm(xa, y.attn_norm, xa, kDim, kNormEps, 1, st);
            if (dbg_layer) dbg_write(xa, kDim);                         // attention input
            attention(l);
            if (dbg_layer) dbg_write(attn_out, kDim);                   // attention output
            ops::hc_post(attn_out, h, attn_post, attn_comb, h2, 1, st);
            // ffn sub-block: h2 -> h
            kernels::hc_mixes_pre(h2, 1, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, attn_pre, xf, ffn_pre, ffn_post,
                                  ffn_comb, st);
            ops::rmsnorm(xf, y.ffn_norm, xf, kDim, kNormEps, 1, st);
            if (dbg_layer) dbg_write(xf, kDim);                         // ffn input
            moe(l);
            if (dbg_layer) dbg_write(ffn_out, kDim);                    // ffn output
            ops::hc_post(ffn_out, h2, ffn_post, ffn_comb, h, 1, st);
            ck(cudaMemcpyAsync(pre_mix, ffn_pre, kHc * 4, cudaMemcpyDeviceToDevice, st), "pre_mix");
            if (dump) {
                ck(cudaStreamSynchronize(st), "dump hidden");
                ck(cudaMemcpy(dump->hidden.data() + (size_t) l * kHc * kDim, h, kHc * kDim * 2, cudaMemcpyDeviceToHost),
                   "dump hidden");
            }
        }
        ops::hc_pre(h, pre_mix, final_x, 1, st);
        ops::rmsnorm(final_x, final_norm, final_x, kDim, kNormEps, 1, st);
        ops::bf16_linear(final_x, nullptr, head, kDim, kVocab, nullptr, logits, st);
        ops::argmax_logits(logits, d_next, st);
        ck(cudaMemcpyAsync(hp_next, d_next, sizeof(int), cudaMemcpyDeviceToHost, st), "next token");
        ck(cudaMemcpyAsync(lg_pinned, logits, (size_t) kVocab * 4, cudaMemcpyDeviceToHost, st), "logits");
        ck(cudaMemcpyAsync(routes_pinned, routes_dev, kLayers * kTopK * 4, cudaMemcpyDeviceToHost, st), "routes");
    }

    /// Indexer capacity of a graph: the next power of two of t (at least 1), at most the allocated rows
    static int64_t cap_of(int64_t t, int64_t rows) {
        int64_t c = 1;
        while (c < t) c <<= 1;
        return std::min(c, rows);
    }

    int step(int token, int pos, StepDump* dump, Timing& tm) {
        usable();
        if(verify_pending)throw std::logic_error("commit the pending verify window before decode");
        check_token(token);
        if(pos<0||native_failed.load())throw std::runtime_error("invalid position or failed native engine");
        if (pos != (int) history.size()) throw std::runtime_error("tokens must be fed in order from position 0");
        if (pos >= max_seq) throw std::runtime_error("position past max_seq");
        bool released = false;
        try {
            return step_body(token, pos, dump, tm, released);
        } catch (...) {
            // nothing reached the device or the worker: undo the history, the step may be retried
            if (released) broken = true;
            else if ((int) history.size() == pos + 1) history.pop_back();
            throw;
        }
    }

    int step_body(int token, int pos, StepDump* dump, Timing& tm, bool& released) {
        tm = Timing{};
        const double t_start = now_ms();
        history.push_back(pack.engram_hash().token_map[token]);
        {
            const double t0 = now_ms();
            int li = 0;
            for (int l = 0; l < kLayers; ++l)
                if (is_engram_layer(l)) engram_ids(l, li++, pos);
            if (fault("engram")) throw std::runtime_error("ds41 test fault: engram read");
            engram_read_all();
            tm.engram_ms = now_ms() - t0;
        }
        if (vram) tm.vram_swaps = vram->between_steps();   // the device is idle: the last step ended in a sync
        db->reset();
        worker_us = 0;
        worker_misses = 0;
        worker_ram = worker_file = worker_ssd = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            ++go;
        }
        released = true;
        cv.notify_one();
        if (dump) {
            dump->hidden.assign((size_t) kLayers * kHc * kDim, 0);
            dump->routes.assign(kLayers, {});
            dump->weights.assign(kLayers, {});
        }
        *hp = StepParams{token, pos, pos + 1, (pos + 1) / 2};
        parity = pos & 1;
        tcap1 = cap_of(pos + 1, max_seq + 1);
        tcap2 = cap_of((pos + 1) / 2, max_seq / 2 + 1);
        if (use_graph && !dump && !dbg) {
            const uint64_t key = (uint64_t) parity | (uint64_t) tcap1 << 1 | (uint64_t) tcap2 << 33;
            auto it = graphs.find(key);
            if (it == graphs.end()) {
                // relaxed: other threads (the tier's copier) keep using CUDA during the capture
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "begin capture");
                enqueue_step(nullptr);
                cudaGraph_t gr;
                ck(cudaStreamEndCapture(st, &gr), "end capture");
                cudaGraphExec_t ex;
                ck(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
                cudaGraphDestroy(gr);
                it = graphs.emplace(key, ex).first;
                ++graph_captures;
            }
            ck(cudaGraphLaunch(it->second, st), "graph launch");
        } else {
            enqueue_step(dump);
        }
        ck(cudaStreamSynchronize(st), "step");
        if(native_failed.load())throw std::runtime_error("native CPU expert computation failed; inspect engine log");
        {
            std::lock_guard<std::mutex> lk(mu);
            if (worker_error) {
                std::exception_ptr e = worker_error;
                worker_error = nullptr;
                std::rethrow_exception(e);
            }
        }
        const int best = *hp_next;
        if(native_capture_main)ck(cudaMemcpy(native_main_ring+size_t(pos%128)*3*kDim,native_main_step,3*kDim*sizeof(uint16_t),cudaMemcpyDeviceToDevice),"native main commit step");
        lg.assign(lg_pinned, lg_pinned + kVocab);
        if (vram) vram->count(routes_pinned, kTopK);
        if (dump) {
            float wv[kLayers * kTopK];
            ck(cudaMemcpy(wv, weights_dev, sizeof wv, cudaMemcpyDeviceToHost), "dump weights");
            for (int l = 0; l < kLayers; ++l)
                for (int i = 0; i < kTopK; ++i) {
                    dump->routes[l][i] = routes_pinned[l * kTopK + i];
                    dump->weights[l][i] = wv[l * kTopK + i];
                }
            std::vector<int> order(kVocab);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            dump->top_logits.clear();
            for (int i = 0; i < 8; ++i) dump->top_logits.push_back({order[i], lg[order[i]]});
        }
        tm.total_ms = now_ms() - t_start;
        tm.cpu_experts_ms = worker_us.load() / 1000.0;
        tm.expert_total = kLayers * kTopK;
        tm.expert_hits = tm.expert_total - worker_misses.load();
        tm.ram_experts = worker_ram.load();
        tm.file_experts = worker_file.load();
        tm.ssd_experts = worker_ssd.load();
        if (lookahead) {
            const auto st_ = lookahead->take_stats();
            tm.warmed = (int) st_.warmed;
            tm.warmed_useful = (int) st_.useful;
        }
        tm.gpu_ms = tm.total_ms - tm.engram_ms;
        return best;
    }
};

#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/524288/verify_causal_batch_v310.cu"

Engine::Engine(const std::string& pack_dir, const EngineOptions& opt) : impl_(new Impl(pack_dir, opt)) {
    impl_->outer325=this;impl_->init();
}
Engine::Engine(std::shared_ptr<NativeCheckpoint>checkpoint,EngramHash hash,const EngineOptions&options,const NativeEngineOptions&native)
    :impl_(new Impl(std::move(checkpoint),std::move(hash),options,native)){impl_->outer325=this;impl_->init();}

Engine::Engine(std::shared_ptr<NativeCheckpoint>checkpoint,EngramHash hash,const EngineOptions&options,const NativeEngineOptions&native,EngineRoleV320 role,const HeadlessOptionsV320&limits)
    :impl_(new Impl(std::move(checkpoint),std::move(hash),options,native,role,limits)){impl_->outer325=this;impl_->init();}

Engine::Engine(const std::string& pack_dir, int max_seq, int cpu_threads)
    : Engine(pack_dir, [&] {
          EngineOptions o;
          o.max_seq = max_seq;
          o.cpu_threads = cpu_threads;
          return o;
      }()) {}

int Engine::vram_expert_slots() const { return impl_->vram ? impl_->vram->slots() : 0; }

Engine::~Engine() = default;
#include "../../../verifier-woa-quant-many-engine-v456/source/base524288/headless_public_v320.inc"

std::vector<double> Engine::native_prefill_profile()const{
    return {impl_->pfp.accumulated.begin(),impl_->pfp.accumulated.end()};
}
void Engine::native_gu_cache_v19(size_t bytes){
    auto&p=*impl_;if(!p.native_cpu||!p.history.empty()||p.verify_pending||p.native_gpu_budget||p.native_gu_budget_v19||bytes>(22ull<<30))throw std::invalid_argument("fresh independent GPU1 GU budget required");
    p.native_gu_budget_v19=bytes;
}
std::array<uint64_t,32> Engine::native_gu_statistics_v19()const{return impl_->native_cpu?impl_->native_cpu->gu_statistics_v19():std::array<uint64_t,32>{};}
void Engine::native_expert_cache(size_t bytes){
    auto&p=*impl_;p.usable();
    if(!p.native_cpu||!p.history.empty()||p.native_gpu_experts||p.verify_pending||bytes>(size_t(22)<<30)||(bytes&&bytes<(size_t(1)<<28)))throw std::invalid_argument("native expert cache needs a fresh engine and256MiB..22GiB budget");
    p.native_gpu_budget=bytes;
}
std::array<uint64_t,5>Engine::native_expert_cache_statistics()const{return impl_->native_gpu_experts?impl_->native_gpu_experts->statistics():std::array<uint64_t,5>{};}
std::vector<std::array<uint64_t,4>>Engine::native_expert_device_statistics()const{return impl_->native_gpu_experts?impl_->native_gpu_experts->device_statistics():std::vector<std::array<uint64_t,4>>{};}
void Engine::native_expert_cache_pool(std::shared_ptr<NativeExpertCachePool>pool){
    auto&p=*impl_;
    if(!pool||!p.native_cpu||!p.history.empty()||!p.native_gpu_budget||p.native_gpu_experts||p.verify_pending||p.native_gpu_pool)throw std::invalid_argument("persistent expert pool requires a fresh configured native engine");
    const char*multiple=std::getenv("DS41_NATIVE_EXPERT_DEVICES"),*single=std::getenv("DS41_NATIVE_EXPERT_DEVICE");
    std::string devices=multiple?multiple:single?single:"0";
    if(pool->requested){
        if(pool->weights.lock().get()!=p.pack.native()->weights().get()||pool->requested!=p.native_gpu_budget||pool->devices!=devices)throw std::invalid_argument("persistent expert pool checkpoint/budget/device identity differs");
    }else{pool->weights=p.pack.native()->weights();pool->requested=p.native_gpu_budget;pool->devices=devices;}
    p.native_gpu_pool=std::move(pool);
    if(p.native_gpu_pool->cache){p.native_gpu_experts=p.native_gpu_pool->cache;p.native_gpu_reused=true;}
}
bool Engine::native_expert_cache_reused()const{return impl_->native_gpu_reused;}

void Engine::native_image_inputs(std::vector<int>positions,const std::vector<uint16_t>&features){
    auto&p=*impl_;p.usable();
    if(!p.native_cpu||!p.history.empty()||p.pf.own_scratch||p.verify_pending||p.native_images||positions.empty())throw std::logic_error("image features require a fresh native engine");
    auto plan=std::make_unique<NativeImagePlan>(std::move(positions),p.max_seq);
    if(features.size()!=plan->positions.size()*kDim)throw std::invalid_argument("image feature dimensions differ");
    for(auto v:features)if((v&0x7f80)==0x7f80)throw std::invalid_argument("nonfinite image feature");
    auto*device=p.own_alloc<bf16>(features.size());
    ck(cudaMemcpy(device,features.data(),features.size()*2,cudaMemcpyHostToDevice),"native image features");
    p.native_image_features=device;p.native_images=std::move(plan);
}

void Engine::cache_native_decoded_expert(bool enabled){
    auto&p=*impl_;p.usable();
    if(!p.native_cpu||!p.history.empty()||p.pf.own_scratch||p.verify_pending)throw std::logic_error("decoded expert cache requires a fresh native engine");
    p.decoded_native_expert_cache=enabled;
}

void Engine::prepare_native_verify_cpu(int rows){
    auto&p=*impl_;p.usable();
    if(!p.native_cpu||p.verify_pending)throw std::logic_error("native CPU preparation requires an idle engine");
    p.native_cpu->prepare_batch(rows);
}
void Engine::capture_native_main(bool enabled){
    auto&p=*impl_;p.usable();
    if(!p.native_cpu||!p.history.empty()||p.verify_pending)throw std::logic_error("native main capture requires a fresh native engine");
    if(enabled&&!p.native_main_ring){
        p.native_main_ring=p.own_alloc<uint16_t>(128*3*kDim);
        p.native_main_step=p.own_alloc<uint16_t>(128*3*kDim);
    }
    p.native_capture_main=enabled;
}
std::vector<uint16_t> Engine::native_main_tail(int&first,int maximum_rows)const{
    auto&p=*impl_;p.usable();
    if(!p.native_capture_main||p.verify_pending||p.history.empty())throw std::logic_error("no committed native main hidden states");
    if(p.headless320){p.headless_current320();if(p.headless_state320||p.headless_ack320||int(p.history.size())-p.headless_observed_first320<std::min(maximum_rows,int(p.history.size())))throw std::logic_error("320 main tail requires completed captured suffix, never seeded-only prefix slots");}
    if(maximum_rows<1||maximum_rows>128)throw std::invalid_argument("bounded native main tail required");
    const int end=int(p.history.size()),rows=std::min(maximum_rows,end);first=end-rows;
    std::vector<uint16_t> result(size_t(rows)*3*kDim);
    const int initial=std::min(rows,128-first%128);
    ck(cudaMemcpy(result.data(),p.native_main_ring+size_t(first%128)*3*kDim,size_t(initial)*3*kDim*sizeof(uint16_t),cudaMemcpyDeviceToHost),"native main tail read");
    if(initial<rows)ck(cudaMemcpy(result.data()+size_t(initial)*3*kDim,p.native_main_ring,size_t(rows-initial)*3*kDim*sizeof(uint16_t),cudaMemcpyDeviceToHost),"native main wrapped tail read");
    return result;
}

int Engine::step(int token, int pos, StepDump* dump) {
    if(impl_->headless320)throw std::logic_error("320 headless role has no decoder");
    if(impl_->native_images&&pos<=impl_->native_images->positions.back())throw std::logic_error("image spans require batched prefill before decode");
    if (impl_->verify_pending) throw std::logic_error("commit the pending verify window first");
    return impl_->step(token, pos, dump, timing_);
}

int Engine::prefill(const std::vector<int>& tokens, int pos, std::vector<float>* nll) {
    if(impl_->headless320)throw std::logic_error("320 use typed headless pass API, no logits/normal prefill");
    if(impl_->native_images&&impl_->history.empty()&&(impl_->opt.prefill_chunk<=0||pos!=0||tokens.size()<=size_t(impl_->native_images->positions.back())))throw std::invalid_argument("first prefill must cover every image span");
    if (impl_->verify_pending) throw std::logic_error("commit the pending verify window first");
    int token=impl_->prefill(tokens,pos,nll,prefill_timing_);
    if(impl_->native_gu_budget_v19)impl_->native_cpu->build_gu_cache_v19(impl_->native_gu_budget_v19,impl_->native_expert_frequency);
    if(impl_->native_gpu_budget&&!impl_->native_gpu_experts){
        impl_->native_gpu_experts=std::make_shared<NativeGpuExperts>(impl_->pack.native()->weights(),impl_->native_gpu_budget,impl_->native_expert_frequency);
        if(impl_->native_gpu_pool)impl_->native_gpu_pool->cache=impl_->native_gpu_experts;
    }
    return token;
}

void Engine::compact_native_prefill_inputs(bool enabled){
    impl_->usable();if(!impl_->native_cpu||impl_->pf.own_scratch||impl_->verify_pending)
        throw std::logic_error("compact native prefill inputs require an idle native engine");
    if(impl_->optimized_native_prefill()&&!enabled)throw std::invalid_argument("host prefill requires compact expert inputs");
    impl_->compact_native_prefill_inputs=enabled;
}
void Engine::host_native_prefill(bool enabled,size_t budget,int expert_batch){
    impl_->usable();
    if(!impl_->native_cpu||impl_->pf.own_scratch||impl_->pfh.cap||impl_->verify_pending||!impl_->history.empty()||(enabled&&impl_->gpu_native_prefill)||budget<(size_t(2)<<30)||budget>(size_t(128)<<30)||expert_batch<1||expert_batch>4096)
        throw std::invalid_argument("host prefill needs a fresh idle native engine and2..128GiB explicit budget");
    impl_->host_native_prefill=enabled;impl_->native_host_budget=budget;impl_->native_host_expert_batch=expert_batch;
    if(enabled)impl_->compact_native_prefill_inputs=true;
}
void Engine::gpu_native_prefill(bool enabled,int expert_batch,bool alias_hc){
    impl_->usable();
    if(!impl_->native_cpu||impl_->pf.own_scratch||impl_->pfh.cap||impl_->verify_pending||!impl_->history.empty()||
       (enabled&&impl_->host_native_prefill)||expert_batch<1||expert_batch>4096)
        throw std::invalid_argument("GPU chunk prefill needs a fresh idle native engine and bounded expert batch");
    if(impl_->plan528.armed&&impl_->plan528.mode==4&&(!enabled||expert_batch!=impl_->plan528.g.E))throw std::invalid_argument("534 GPU prefill E differs from actual prepared plan");
    impl_->gpu_native_prefill=enabled;impl_->alias_native_hc=enabled&&alias_hc;impl_->native_host_expert_batch=expert_batch;
    if(enabled)impl_->compact_native_prefill_inputs=true;
}
void Engine::configure_native_prefill(int chunk,int batch){
    impl_->usable();
    if(!impl_->native_cpu||impl_->verify_pending||(!impl_->optimized_native_prefill()&&impl_->max_seq>8192)||chunk<1||chunk>(impl_->optimized_native_prefill()?1048576:8192)||batch<1||batch>(impl_->optimized_native_prefill()?4096:4096))
        throw std::invalid_argument("native batched prefill requires context<=8192, chunk<=8192 and batch<=4096");
    if(impl_->plan528.armed&&
       (chunk!=impl_->plan528.g.cap||batch!=impl_->plan528.g.B))throw std::invalid_argument("528 setter differs from idle geometry");
    impl_->opt.prefill_chunk=chunk;impl_->opt.prefill_batch=batch;
}
const std::vector<float>& Engine::last_logits() const { return impl_->lg; }
std::vector<uint64_t> Engine::native_worker_rows()const{
    if(!impl_->native_cpu)throw std::logic_error("not a native expert engine");
    return impl_->native_cpu->worker_rows();
}

const uint16_t* Engine::native_shared_dense(int which)const{
    impl_->usable();if(!impl_->native_cpu||(which!=0&&which!=1))throw std::invalid_argument("shared dense weights require native engine and embed/head ID");
    return reinterpret_cast<const uint16_t*>(which==0?impl_->embed:impl_->head);
}

std::array<uint64_t,4>Engine::native_compact_candidate_info()const{const auto&p=*impl_;return{p.compact_candidate_prefill?1ull:0ull,p.compact_candidate_packed_rows,p.compact_candidate_unpacked_rows,p.compact_candidate_saved_peak};}
std::array<uint64_t,5>Engine::native_expert_cache_tail_info()const{const auto&p=*impl_;return{uint64_t(p.native_cache_tail_rows),p.native_cache_tail_active?1ull:0ull,uint64_t(p.native_cache_tail_begin),uint64_t(p.native_cache_tail_end),p.native_cache_tail_counted_uses};}
std::array<uint64_t,3>Engine::native_prefill_memory_info()const{const auto&p=*impl_;return{uint64_t(p.native_prefill_margin_mib),p.native_prefill_sampled_passes?p.native_prefill_minimum_free_bytes:0ull,p.native_prefill_sampled_passes};}
std::array<uint64_t,4>Engine::native_decoder_suffix_info()const{const auto&p=*impl_;return{p.decoder_suffix_prefill?1ull:0ull,p.decoder_suffix_passes,p.decoder_suffix_layers,p.decoder_suffix_rows_skipped};}
std::array<uint64_t,3>Engine::native_packed_fp8_info()const{const auto&p=*impl_;return{p.direct_fp8_expert_weights?1ull:0ull,p.direct_fp8_projection_encodes,p.direct_fp8_expert_weights?3ull*kDim*kMoeInter*2+native_prefill_workspace_bytes(std::max(1,p.pf.native_expert_batch),kDim,kMoeInter):0ull};}
std::array<uint64_t,16> Engine::native_output16_info_v252()const{
    const auto&p=*impl_;const auto registered=p.pack.native()?p.pack.native()->weights()->output16_info_v248():std::vector<uint64_t>{1,0,0};
    const auto cache=p.native_gpu_experts?p.native_gpu_experts->output16_info_v252():std::array<uint64_t,4>{};
    return {252,registered[1],registered[2],p.output16_compact_v252?1ull:0ull,
        p.output16_uploads_v252,p.output16_bytes_v252,p.output16_encodes_v252,p.output16_capacity_v252,
        p.output16_max_triple_v252,p.output16_raw_uploads_v252,
        cache[0],cache[1],cache[2],cache[3],uint64_t(p.output16_last_cap_v252),uint64_t(p.output16_last_sub_v252)};
}
std::array<uint64_t,24> Engine::native_dense_stream_info_v191() const {
    return impl_->pack.dense_stream_info_v190();
}

std::array<uint64_t,7> Engine::native_k3_causal_batch_info_v310()const{
    if(!impl_->verify_ws)return {310,0,0,0,0,0,0};
    const auto&v=*impl_->verify_ws;
    return {310,v.k3_causal_batch_v310?1ull:0ull,1,
        v.k3_verify_calls_v310,v.k3_launches_v310,v.k3_rows_v310,v.k3_reused_graphs_v310};
}
// Called only through the real Handle mutex in the selected native Engine CABI.
// This is an ordinary-idle diagnostic, never an inference/capture-time API.
std::array<uint64_t,24> Engine::native_candidate_cutoff_info_v375()const{
    const auto&p=*impl_;
    std::array<uint64_t,24>out{};
    out[0]=375;out[1]=p.candidate_cutoff375?1:0;out[2]=p.candidate_workspace_bytes375;
    out[3]=uint64_t(p.max_seq);out[4]=kCandidateBlock;out[5]=kCandidateBlocks;
    out[19]=p.headless320?1:0;
    if(!p.candidate_workspace375)return out; // OFF/headless: no fabricated device state.
    int device=-1;ck(cudaGetDevice(&device),"375 getter current device");
    if(device!=0||p.headless320||p.active_pass317)
        throw std::logic_error("375 requires completed headful GPU0 target/prefill idle");
    cudaStreamCaptureStatus capture{};
    ck(cudaStreamIsCapturing(p.st,&capture),"375 getter stream capture");
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("375 getter cannot read during capture");
    const cudaError_t idle=cudaStreamQuery(p.st);
    if(idle==cudaErrorNotReady)throw std::logic_error("375 target stream still active");
    ck(idle,"375 idle stream query");
    Ds369State state{};
    ck(cudaMemcpyAsync(&state,p.candidate_workspace375,sizeof(state),cudaMemcpyDeviceToHost,p.st),"375 actual device state down");
    ck(cudaStreamSynchronize(p.st),"375 getter state completion");
    out[6]=state.positions;out[7]=state.blocks;out[8]=state.remaining;
    out[9]=state.prefix;out[10]=state.prefix_mask;out[11]=uint64_t(int64_t(state.k));
    out[12]=uint64_t(int64_t(state.mode));out[13]=uint64_t(int64_t(state.status));
    out[14]=uint64_t(int64_t(state.reserved0));out[15]=state.reserved[0];out[16]=state.reserved[1];
    out[17]=uint64_t(device);out[18]=1; // a real same-stream state read completed.
    // 20..23 reserved0. No host call counter is used as graph/replay evidence.
    return out;
}

std::shared_ptr<ds387::Lease>Engine::native_adaptive_lease_v387()const{
    const auto&p=*impl_;return p.native_cpu?ds387::lookup(*p.native_cpu):std::shared_ptr<ds387::Lease>{};
}
void Engine::native_adaptive_idle_v387(const std::shared_ptr<ds387::Lease>&lease)const{
    const auto&p=*impl_;
    if(p.headless320||!p.native_cpu||!lease||lease->closed()||p.active_pass317||
       p.adaptive_terminal387||ds387::lookup(*p.native_cpu)!=lease)
        throw std::logic_error("387 requires actual live headful NativeExperts lease");
    int device=-1;ck(cudaGetDevice(&device),"387 current target device");
    if(device!=0)throw std::logic_error("387 controls require currentGPU0, no implicit devicehop");
    cudaStreamCaptureStatus capture{};ck(cudaStreamIsCapturing(p.st,&capture),"387 target capture state");
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("387 controls forbidden during capture");
    const auto status=cudaStreamQuery(p.st);if(status==cudaErrorNotReady)throw std::logic_error("387 controls require completed target stream");
    ck(status,"387 target stream idle");
}
void Engine::native_adaptive_close_v387(const std::shared_ptr<ds387::Lease>&lease){
    auto&p=*impl_;if(!p.native_cpu||!lease)throw std::logic_error("387 missing actual native lease");
    if(!lease->closed())native_adaptive_idle_v387(lease);
    lease->close(*p.native_cpu);p.adaptive_terminal387=true;
}

}  // namespace strata::ds41
