#pragma once
#include "strata/ds41/native_adapter.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    const int32_t* token_map; size_t token_count;
    const int64_t* multipliers; size_t multiplier_count; // [2,4]
    const int64_t* primes; const int64_t* offsets; size_t modulus_count; // each [2,24]
    int64_t pad;
} ds41_native_hash;
typedef struct {
    int context;
    const int* cpus;size_t cpu_count;
    const char* task_library;const char* projection_library;
    int a8; // Stable ABI selector:0 nativeF32,1 A8,2 explicit mixed-format F32.
    int graph;
} ds41_native_engine_options;
const char* ds41_native_engine_error(void);
void* ds41_native_engine_create(void* checkpoint,const ds41_native_hash*,const ds41_native_engine_options*);
// Optional hidden/routes must either both be absent or have exact model sizes.
int ds41_native_engine_step(void*,int token,int position,int* next,float* logits,size_t logit_count,
                           uint16_t* hidden,size_t hidden_count,int32_t*routes,size_t route_count,double*times,size_t time_count);
int ds41_native_engine_compact_prefill_inputs(void*,int enabled);
int ds41_native_engine_host_prefill(void*,int enabled,size_t host_budget_bytes,int expert_batch);
int ds41_native_engine_gpu_prefill(void*,int enabled,int expert_batch,int alias_hc);
// Tentative1..4 rows; a successful call must be followed by commit(1..count).
int ds41_native_engine_verify(void*,const int32_t*tokens,size_t count,int position,
                              int32_t*next,float*logits,size_t logit_count,double*times,size_t time_count);
int ds41_native_engine_commit(void*,int keep);
int ds41_native_engine_capture_main(void*,int enabled);
int ds41_native_engine_main_tail(void*,uint16_t*out,size_t capacity,int*first,int*rows);
int ds41_native_engine_main_recent(void*,int maximum_rows,uint16_t*out,size_t capacity,int*first,int*rows);
int ds41_native_engine_prepare_verify_cpu(void*,int rows);
int ds41_native_engine_verify_graph_reused(void*,int*out);
//13 GPU-timeline buckets in milliseconds; includes host waits between events.
int ds41_native_engine_compact_candidate_info(void*handle,uint64_t*out,size_t count);
int ds41_native_engine_decoder_suffix_info(void*handle,uint64_t*out,size_t count);
int ds41_native_engine_prefill_profile(void*,double*out,size_t count);
int ds41_native_engine_decoded_expert_cache(void*,int enabled);
int ds41_native_engine_prefill_shape(void*,int*out,size_t count);
// Default native batch context<=8192. Experimental explicit host mode supports
// up to1M with attentionbatch<=256; admission requires separate measured proof.
int ds41_native_engine_prefill(void*,const int32_t*tokens,size_t count,int position,int chunk,int batch,
                              int*next,float*logits,size_t logit_count,float*nll,size_t nll_count,double*times,size_t time_count);
// Optional191: DS41_NATIVE_PREFILL_STREAM_DENSE=1 BEFORE create; default0.
// Direct Pack190 schema24; see pack_stream_dense_v190.hpp for the ordered labels.
int ds41_native_engine_dense_stream_info_v191(void*,uint64_t*out,size_t count);
// Optional252 compact descriptors: explicit env before create, defaultOFF.
// Exact16 values: schema,registered matrices/bytes,enabled,PF uploads/H2Dbytes/
// encodes/capacity/maxtriple/rawuploads,cache entries/H2Dbytes/inversebytes/temp,
// last actual prefill cap/sub. No raw-pointer aliasing.
int ds41_native_engine_output16_info_v252(void*,uint64_t*out,size_t count);
int ds41_native_engine_k3_causal_batch_info_v310(void*,uint64_t*out,size_t count);
// Opaque selected native Handle, not Engine*. count must equal24.
int ds375_native_engine_candidate_info(void*handle,uint64_t*out,int count);
void ds41_native_engine_destroy(void*);
int ds41_native_engine_worker_rows(void*,uint64_t*,size_t);
#ifdef __cplusplus
}
#endif
