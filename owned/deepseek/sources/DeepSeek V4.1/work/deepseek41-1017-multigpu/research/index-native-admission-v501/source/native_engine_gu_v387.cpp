#include "index_pair_admission_v501.hpp"
#include "strata/ds41/native_engine_causal_batch_v310.h"
#include "../../whole-adaptive-engine-v387/source/adaptive_bridge_v387.h"
#include "../../whole-adaptive-engine-v387/source/native_adaptive_lease_v387.hpp"
#include "strata/ds41/engine_causal_batch_v310.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/native_gpu_experts_compact_v252.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
using namespace strata::ds41;
namespace {
thread_local std::string error;
struct Handle{
 std::unique_ptr<Engine>engine;std::mutex mutex;bool verify_graph_reused=false;
 const bool adaptive_requested=ds387::enabled();
 std::shared_ptr<ds387::Lease>adaptive;
 ~Handle()noexcept{
  std::lock_guard<std::mutex>guard(mutex);
  if(adaptive){try{if(!engine)std::terminate();engine->native_adaptive_close_v387(adaptive);adaptive.reset();}catch(...){std::terminate();}}
  engine.reset(); // registered Task hooks/owner resources closed before Enginefree
 }
};
void need(bool ok,const char*why){if(!ok)throw std::invalid_argument(why);}
int admitted_mask_stage(){
    ds501::require_environment();
    return 4; // private pair sentinel; non-pair modes are rejected above
}
void admit_gpu(){
    ds501::validate_all();
    if(cudaSetDevice(0)!=cudaSuccess)throw std::runtime_error("501 cannot select current pair backbone CUDA0");
}
template<class F>int checked(F fn){try{error.clear();fn();return 0;}catch(const std::exception&e){error=e.what();return 1;}catch(...){error="unknown native engine exception";return 2;}}
}
extern "C" {
void* ds41_native_expert_pool_create(){void*result=nullptr;checked([&]{result=new std::shared_ptr<NativeExpertCachePool>(std::make_shared<NativeExpertCachePool>());});return result;}
void ds41_native_expert_pool_destroy(void*pool){delete static_cast<std::shared_ptr<NativeExpertCachePool>*>(pool);}
int ds41_native_engine_attach_expert_pool(void*handle,void*pool){return checked([&]{need(handle&&pool,"invalid expert pool attachment");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);h.engine->native_expert_cache_pool(*static_cast<std::shared_ptr<NativeExpertCachePool>*>(pool));});}
int ds41_native_engine_expert_pool_reused(void*handle,int*out){return checked([&]{need(handle&&out,"invalid expert pool reuse query");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);*out=h.engine->native_expert_cache_reused()?1:0;});}
const char* ds41_native_engine_error(void){return error.c_str();}
void* ds41_native_engine_create(void*checkpoint,const ds41_native_hash*hash,const ds41_native_engine_options*options){
    void*result=nullptr;checked([&]{
        // Validate mask before any CUDA call, including when called incorrectly.
        admitted_mask_stage();
        need(checkpoint&&hash&&options,"missing checkpoint/hash/options");
        need(hash->token_map&&hash->token_count==kVocab&&hash->multipliers&&hash->multiplier_count==8&&hash->primes&&hash->offsets&&hash->modulus_count==48,"invalid native hash shape");
        need(options->context>=1&&options->context<=1048576&&options->cpus&&options->cpu_count>=1&&options->cpu_count<=256&&options->task_library&&options->projection_library,
             "invalid native engine geometry/libraries");
        need((options->a8==0||options->a8==1||options->a8==2)&&(options->graph==0||options->graph==1),"invalid arithmetic/graph flags");
        EngramHash h;h.max_ngram=4;h.n_heads=8;h.pad=hash->pad;h.layers={1,14};h.token_map.assign(hash->token_map,hash->token_map+hash->token_count);
        for(int i=0;i<2;i++){
            h.multipliers.emplace_back(hash->multipliers+i*4,hash->multipliers+(i+1)*4);
            h.primes.emplace_back(hash->primes+i*24,hash->primes+(i+1)*24);h.offsets.emplace_back(hash->offsets+i*24,hash->offsets+(i+1)*24);
        }
        EngineOptions o;o.max_seq=options->context;o.cpu_threads=int(options->cpu_count);o.prefill_chunk=0;o.vram_expert_slots=0;o.ram_budget_gib=0;
        NativeEngineOptions n;n.task_library=options->task_library;n.projection_library=options->projection_library;n.a8=options->a8==1;n.mixed_f32=options->a8==2;n.graph=options->graph;
        n.cpus.assign(options->cpus,options->cpus+options->cpu_count);
        admit_gpu();auto handle=std::make_unique<Handle>();
        handle->engine=std::make_unique<Engine>(std::make_shared<NativeCheckpoint>(*static_cast<NativeCheckpoint*>(checkpoint)),std::move(h),o,n);
        result=handle.release();
    });return result;
}
int ds41_native_engine_step(void*handle,int token,int position,int*next,float*logits,size_t count,uint16_t*hidden,size_t nhidden,int32_t*routes,size_t nroutes,double*times,size_t ntimes){return checked([&]{
    need(handle&&next&&logits&&count==kVocab&&times&&ntimes==4,"invalid engine output buffers");
    need((!hidden&&!routes&&!nhidden&&!nroutes)||(hidden&&routes&&nhidden==size_t(kLayers*kHc*kDim)&&nroutes==size_t(kLayers*kTopK)),"invalid intermediate probe buffers");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);StepDump dump;
    int predicted=h.engine->step(token,position,hidden?&dump:nullptr);
    const auto&values=h.engine->last_logits();need(values.size()==count,"incomplete logits");std::copy(values.begin(),values.end(),logits);
    if(hidden){std::copy(dump.hidden.begin(),dump.hidden.end(),hidden);for(int l=0;l<kLayers;l++)std::copy(dump.routes[l].begin(),dump.routes[l].end(),routes+l*kTopK);}
    const auto&t=h.engine->last_timing();times[0]=t.total_ms;times[1]=t.gpu_ms;times[2]=t.cpu_experts_ms;times[3]=t.engram_ms;*next=predicted;
});}
int ds41_native_engine_compact_prefill_inputs(void*handle,int enabled){return checked([&]{
    need(handle&&(enabled==0||enabled==1),"invalid native compact-input switch");auto&h=*static_cast<Handle*>(handle);
    std::lock_guard<std::mutex>lock(h.mutex);h.engine->compact_native_prefill_inputs(enabled);
});}
int ds41_native_engine_host_prefill(void*handle,int enabled,size_t budget,int expert_batch){return checked([&]{
    need(handle&&(enabled==0||enabled==1),"invalid native host-prefill switch");auto&h=*static_cast<Handle*>(handle);
    std::lock_guard<std::mutex>lock(h.mutex);h.engine->host_native_prefill(enabled,budget,expert_batch);
});}
int ds41_native_engine_gpu_prefill(void*handle,int enabled,int expert_batch,int alias_hc){return checked([&]{
    need(handle&&(enabled==0||enabled==1)&&(alias_hc==0||alias_hc==1),"invalid GPU chunk prefill switch");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    h.engine->gpu_native_prefill(enabled,expert_batch,alias_hc);
});}
int ds41_native_engine_prefill(void*handle,const int32_t*tokens,size_t count,int position,int chunk,int batch,
                              int*next,float*logits,size_t logit_count,float*nll,size_t nll_count,double*times,size_t time_count){return checked([&]{
    need(handle&&tokens&&count>=1&&count<=1048576&&next&&logits&&logit_count==kVocab&&times&&time_count==4,"invalid native prefill buffers");
    need((!nll&&!nll_count)||(nll&&nll_count==count-1),"invalid native prefill NLL buffer");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    h.engine->configure_native_prefill(chunk,batch);std::vector<int>ids(tokens,tokens+count);std::vector<float>loss;
    *next=h.engine->prefill(ids,position,nll?&loss:nullptr);
    if(h.adaptive_requested){
        auto lease=h.engine->native_adaptive_lease_v387();need(bool(lease),"387 prefill did not install actual adaptive owner");
        need(!h.adaptive||h.adaptive==lease,"387 immutable realHandle lease changed");h.adaptive=std::move(lease);
    }

    const auto&values=h.engine->last_logits();need(values.size()==logit_count,"incomplete prefill logits");
    std::copy(values.begin(),values.end(),logits);if(nll){need(loss.size()==nll_count,"incomplete prefill NLL");std::copy(loss.begin(),loss.end(),nll);}
    const auto&t=h.engine->last_prefill();times[0]=t.total_ms;times[1]=t.engram_ms;times[2]=t.stream_wait_ms;times[3]=t.chunks;
});}
int ds41_native_engine_shared_dense(void*handle,int which,uintptr_t*data,size_t*rows,size_t*cols){return checked([&]{
    need(handle&&data&&rows&&cols&&(which==0||which==1),"invalid shared dense view request");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    *data=reinterpret_cast<uintptr_t>(h.engine->native_shared_dense(which));*rows=kVocab;*cols=kDim;
});}
void ds41_native_engine_destroy(void*handle){delete static_cast<Handle*>(handle);}
int ds41_native_engine_verify(void*handle,const int32_t*tokens,size_t count,int position,
                              int32_t*next,float*logits,size_t logit_count,double*times,size_t time_count){return checked([&]{
    need(handle&&tokens&&count>=1&&count<=8&&next&&logits&&logit_count==count*kVocab&&times&&time_count==4,"invalid native verification buffers");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    auto result=h.engine->verify(std::vector<int>(tokens,tokens+count),position,true);
    h.verify_graph_reused=result.graph_reused;
    need(result.next.size()==count&&result.logits.size()==count,"incomplete native verification");
    std::copy(result.next.begin(),result.next.end(),next);
    for(size_t i=0;i<count;i++){
        need(result.logits[i].size()==kVocab,"incomplete native verification logits");
        std::copy(result.logits[i].begin(),result.logits[i].end(),logits+i*kVocab);
    }
    const auto&t=h.engine->last_timing();times[0]=t.total_ms;times[1]=t.gpu_ms;times[2]=t.cpu_experts_ms;times[3]=t.engram_ms;
});}
int ds41_native_engine_commit(void*handle,int keep){return checked([&]{
    need(handle&&keep>=1&&keep<=8,"invalid native commit count");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);h.engine->commit(keep);
});}
int ds41_native_engine_capture_main(void*handle,int enabled){return checked([&]{
    need(handle&&(enabled==0||enabled==1),"invalid native main capture switch");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);h.engine->capture_native_main(enabled);
});}
int ds41_native_engine_main_tail(void*handle,uint16_t*out,size_t capacity,int*first,int*rows){return checked([&]{
    need(handle&&out&&capacity==size_t(128*3*kDim)&&first&&rows,"invalid native main tail buffers");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);int start=0;auto values=h.engine->native_main_tail(start);
    need(values.size()<=capacity&&values.size()%(3*kDim)==0,"invalid native main tail size");
    std::copy(values.begin(),values.end(),out);*first=start;*rows=int(values.size()/(3*kDim));
});}
int ds41_native_engine_prepare_verify_cpu(void*handle,int rows){return checked([&]{
    need(handle&&rows>=1&&rows<=8,"invalid native CPU prepared shape");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);h.engine->prepare_native_verify_cpu(rows);
});}
int ds41_native_engine_verify_graph_reused(void*handle,int*out){return checked([&]{
    need(handle&&out,"missing graph reuse output");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);*out=h.verify_graph_reused?1:0;
});}
int ds41_native_engine_main_recent(void*handle,int maximum_rows,uint16_t*out,size_t capacity,int*first,int*rows){return checked([&]{
    need(handle&&maximum_rows>=1&&maximum_rows<=128&&out&&capacity==size_t(maximum_rows*3*kDim)&&first&&rows,"invalid native recent main buffers");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);int start=0;auto values=h.engine->native_main_tail(start,maximum_rows);
    need(values.size()<=capacity&&values.size()%(3*kDim)==0,"invalid recent main size");
    std::copy(values.begin(),values.end(),out);*first=start;*rows=int(values.size()/(3*kDim));
});}
int ds41_native_engine_worker_rows(void*handle,uint64_t*out,size_t count){return checked([&]{
    need(handle&&out&&count&&count<=256,"invalid worker counter output");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto rows=h.engine->native_worker_rows();
    need(rows.size()==count,"worker counter count differs");std::copy(rows.begin(),rows.end(),out);
});}
int ds41_native_engine_compact_candidate_info(void*handle,uint64_t*out,size_t count){return checked([&]{
    need(handle&&out&&count==4,"invalid compact candidate stats output");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto v=h.engine->native_compact_candidate_info();std::copy(v.begin(),v.end(),out);
});}
int ds41_native_engine_decoder_suffix_info(void*handle,uint64_t*out,size_t count){return checked([&]{
    need(handle&&out&&count==4,"invalid decoder suffix stats output");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto v=h.engine->native_decoder_suffix_info();std::copy(v.begin(),v.end(),out);
});}
int ds41_native_engine_packed_fp8_info(void*handle,uint64_t*out,size_t count){return checked([&]{
    need(handle&&out&&count==3,"invalid packed FP8 stats output");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto v=h.engine->native_packed_fp8_info();std::copy(v.begin(),v.end(),out);
});}
int ds41_native_engine_prefill_profile(void*handle,double*out,size_t count){return checked([&]{
    need(handle&&out&&count==13,"invalid native prefill profile output");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto values=h.engine->native_prefill_profile();
    need(values.size()==count,"native profile inventory differs");std::copy(values.begin(),values.end(),out);
});}
int ds41_native_engine_decoded_expert_cache(void*handle,int enabled){return checked([&]{
    need(handle&&(enabled==0||enabled==1),"invalid decoded expert switch");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);h.engine->cache_native_decoded_expert(enabled);
});}
int ds41_native_engine_prefill_shape(void*handle,int*out,size_t count){return checked([&]{
    need(handle&&out&&count==3,"invalid actual prefill shape output");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);const auto&t=h.engine->last_prefill();
    out[0]=t.chunks;out[1]=t.chunk_tokens;out[2]=t.sub_batch;
});}
int ds41_native_engine_image_inputs(void*handle,const int*positions,size_t count,const uint16_t*features,size_t elements){return checked([&]{
    need(handle&&positions&&features&&count>=1&&count<=32768&&elements==count*kDim,"invalid native image feature inventory");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    h.engine->native_image_inputs(std::vector<int>(positions,positions+count),std::vector<uint16_t>(features,features+elements));
});}

int ds41_native_engine_expert_cache(void*handle,size_t bytes){return checked([&]{need(handle,"missing native engine");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);h.engine->native_expert_cache(bytes);});}
int ds41_native_engine_expert_cache_stats(void*handle,uint64_t*out,size_t count){return checked([&]{need(handle&&out&&count==5,"invalid expert cache stats");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto stats=h.engine->native_expert_cache_statistics();std::copy(stats.begin(),stats.end(),out);});}
int ds41_native_engine_cache_tail_info(void*handle,uint64_t*out,size_t count){return checked([&]{need(handle&&out&&count==5,"invalid cache tail stats");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto stats=h.engine->native_expert_cache_tail_info();std::copy(stats.begin(),stats.end(),out);});}
int ds41_native_engine_output16_info_v252(void*handle,uint64_t*out,size_t count){return checked([&]{
    need(handle&&out&&count==16,"invalid252 output16 staging stats");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    auto info=h.engine->native_output16_info_v252();std::copy(info.begin(),info.end(),out);
});}
int ds41_native_engine_dense_stream_info_v191(void*handle,uint64_t*out,size_t count){return checked([&]{
    need(handle&&out&&count==24,"invalid191 streamed dense stats");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    auto info=h.engine->native_dense_stream_info_v191();std::copy(info.begin(),info.end(),out);
});}
int ds41_native_engine_prefill_memory_info(void*handle,uint64_t*out,size_t count){return checked([&]{need(handle&&out&&count==3,"invalid native prefill memory stats");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto stats=h.engine->native_prefill_memory_info();std::copy(stats.begin(),stats.end(),out);});}
int ds41_native_engine_expert_device_stats(void*handle,uint64_t*out,size_t count){return checked([&]{need(handle&&out&&count==16,"invalid expert device stats");auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);auto stats=h.engine->native_expert_device_statistics();std::fill_n(out,count,uint64_t(0));need(stats.size()<=4,"too many expert devices");for(const auto&row:stats){need(row[0]<4,"invalid expert device index");std::copy(row.begin(),row.end(),out+row[0]*4);}});}

int ds41_native_engine_k3_causal_batch_info_v310(void*handle,uint64_t*out,size_t count){return checked([&]{
    need(handle&&out&&count==7,"invalid310 causal K3 stats");
    auto&h=*static_cast<Handle*>(handle);std::lock_guard<std::mutex>lock(h.mutex);
    auto info=h.engine->native_k3_causal_batch_info_v310();std::copy(info.begin(),info.end(),out);
});}

}

extern "C" int ds41_native_engine_gu_cache_v19(void*value,uint64_t bytes){
 try{need(value,"GU engine handle");auto&h=*static_cast<Handle*>(value);std::lock_guard<std::mutex>lock(h.mutex);h.engine->native_gu_cache_v19(bytes);return 0;}catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int ds41_native_engine_gu_statistics_v19(void*value,uint64_t*out,int n){
 try{need(value&&out&&n==32,"GU statistics buffers");auto&h=*static_cast<Handle*>(value);std::lock_guard<std::mutex>lock(h.mutex);auto v=h.engine->native_gu_statistics_v19();std::copy(v.begin(),v.end(),out);return 0;}catch(const std::exception&e){error=e.what();return 1;}
}

extern "C" int ds375_native_engine_candidate_info(void*handle,uint64_t*out,int count){
    return checked([&]{
        need(handle&&out&&count==24,"375 requires actual native Handle and24 outputs");
        auto&h=*static_cast<Handle*>(handle); // same TU's authoritative existing opaque wrapper.
        std::unique_lock<std::mutex>lock(h.mutex,std::try_to_lock);
        need(lock.owns_lock(),"375 native Engine is busy");
        need(bool(h.engine),"375 missing native Engine");
        const auto values=h.engine->native_candidate_cutoff_info_v375();
        std::copy(values.begin(),values.end(),out);
    });
}

extern "C" int ds387_native_engine_adaptive_arm(void*handle,uint64_t generation,uint64_t format){
 return checked([&]{need(handle,"387 missing native Handle");auto&h=*static_cast<Handle*>(handle);std::unique_lock<std::mutex>lock(h.mutex,std::try_to_lock);need(lock.owns_lock()&&h.engine&&h.adaptive,"387 busy/missing realadaptiveHandle lease");h.engine->native_adaptive_idle_v387(h.adaptive);h.adaptive->arm(generation,format);});
}
extern "C" int ds387_native_engine_adaptive_gap(void*handle,uint64_t generation,uint64_t format,uint64_t nonce,unsigned max,unsigned*disposition){
 return checked([&]{need(handle&&disposition,"387 missing native Handle/disposition");auto&h=*static_cast<Handle*>(handle);std::unique_lock<std::mutex>lock(h.mutex,std::try_to_lock);need(lock.owns_lock()&&h.engine&&h.adaptive,"387 busy/missing realadaptiveHandle lease");h.engine->native_adaptive_idle_v387(h.adaptive);*disposition=h.adaptive->gap(generation,format,nonce,max);});
}
extern "C" int ds387_native_engine_adaptive_receipt(void*handle,void*out,size_t bytes){
 return checked([&]{need(handle&&out&&bytes==sizeof(ds370::Receipt370),"387 typed672B receipt required");auto&h=*static_cast<Handle*>(handle);std::unique_lock<std::mutex>lock(h.mutex,std::try_to_lock);need(lock.owns_lock()&&h.engine&&h.adaptive,"387 busy/missing actualadaptiveowner");if(!h.adaptive->closed())h.engine->native_adaptive_idle_v387(h.adaptive);const auto r=h.adaptive->receipt();std::memcpy(out,&r,sizeof(r));});
}
extern "C" int ds387_native_engine_adaptive_close(void*handle,void*out,size_t bytes){
 return checked([&]{need(handle&&out&&bytes==sizeof(ds370::Receipt370),"387 typed672B terminal receipt required");auto&h=*static_cast<Handle*>(handle);std::unique_lock<std::mutex>lock(h.mutex,std::try_to_lock);need(lock.owns_lock()&&h.engine&&h.adaptive,"387 busy/missing actualadaptiveowner");h.engine->native_adaptive_close_v387(h.adaptive);const auto r=h.adaptive->receipt();std::memcpy(out,&r,sizeof(r));});
}
