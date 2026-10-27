#include "gpu1_generation_off_v489.h"
#include "active_epoch_v427.h"
#include "bulk_home_v418.h"
#include "home_memory_v418.hpp"
#include <condition_variable>
#include "whole_hooks_v299.hpp"
#include "policy_hooks_v386.hpp"
#include "../../gpu-gu-borrowed-post-v18-r3/gu_hooks_v18.hpp"
#include "../../gpu-gu-task-overlap-v6/trace_v6.hpp"
#include <mutex>
#include <cstdlib>
// v118 DOWN-only exact A8 prequant in existing worker post; frozen v66 source math.
// v66: parallelize per32 hidden postprocessing on the borrowed projection worker pool.
// Bounded complete MoE tasks. The projection library remains separately built.
// CPU tests do not load libcuda; driver symbols resolve only at an explicit call.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <sched.h>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cuda.h>
#include <immintrin.h>
#include "../../cpu-pair32-static-v143/source/native_a8_prequant_views_v118.hpp"
extern "C" __m512 _ZGVeN16v_expf(__m512);

namespace {
using ParallelCallback=int(*)(void*,int,int);
using ParallelPost=int(*)(void*,ParallelCallback,void*,int);
using NativeBatch=int(*)(void*,const uintptr_t*,const uintptr_t*,const uintptr_t*,const uintptr_t*,const int*,const int*,int,uint64_t*,int*);
using A8Batch=int(*)(void*,const uintptr_t*,const uintptr_t*,const uintptr_t*,const uintptr_t*,const int*,const int*,const int*,int,uint64_t*,int*);
float bf16(uint16_t u){uint32_t v=uint32_t(u)<<16;float f;std::memcpy(&f,&v,4);return f;}
uint16_t round_bf16(float f){uint32_t v;std::memcpy(&v,&f,4);return uint16_t((v+0x7fff+((v>>16)&1))>>16);}
float fp8_value(int code){return code<8?std::ldexp(float(code),-9):std::ldexp(1.f+float(code&7)/8.f,(code>>3)-7);}
const std::array<float,127> fp8_values=[](){std::array<float,127> v{};for(int i=0;i<127;i++)v[i]=fp8_value(i);return v;}();
float fp8_round(float x){
    const float magnitude=std::min(std::abs(x),448.f);
    int code;
    if(magnitude<0.015625f){
        const float scaled=magnitude*512.f;code=int(scaled);
        const float fraction=scaled-float(code);
        code+=(fraction>.5f||(fraction==.5f&&(code&1)));
    }else{
        uint32_t u;std::memcpy(&u,&magnitude,4);
        code=int((u+0x7ffffu+((u>>20)&1u))>>20)-960;
    }
    return std::copysign(fp8_values[std::min(code,126)],x);
}
// This matches source per32 FP8/E8M0 scale rounding via IEEE bits, rather than
// silently changing DeepSeek's activation group to another architecture's128.
__m512 load_bf16(const uint16_t* p){return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))),16));}
__m512 round_bf16_vec(__m512 x){
 auto u=_mm512_castps_si512(x);u=_mm512_add_epi32(u,_mm512_add_epi32(_mm512_set1_epi32(0x7fff),_mm512_and_si512(_mm512_srli_epi32(u,16),_mm512_set1_epi32(1))));
 return _mm512_castsi512_ps(_mm512_and_si512(u,_mm512_set1_epi32(0xffff0000u)));
}
int quantize(const uint16_t* input,float* output,int columns){
 for(int first=0;first<columns;first+=32){
  __m512 a=load_bf16(input+first),b=load_bf16(input+first+16);
  const auto abs_mask=_mm512_set1_epi32(0x7fffffff);
  auto aa=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a),abs_mask));auto bb=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(b),abs_mask));
  if(_mm512_cmp_ps_mask(aa,_mm512_set1_ps(3.402823466e38f),_CMP_LE_OQ)!=0xffff||_mm512_cmp_ps_mask(bb,_mm512_set1_ps(3.402823466e38f),_CMP_LE_OQ)!=0xffff)return 4;
#ifdef DS41_DIRECT_A8
  _mm512_storeu_ps(output+first,a);_mm512_storeu_ps(output+first+16,b);
#else
  float maximum=std::max(1e-4f,_mm512_reduce_max_ps(_mm512_max_ps(aa,bb)));float raw=maximum*(1.f/448.f);uint32_t u;std::memcpy(&u,&raw,4);
  int exponent=int((u>>23)&255)-127+((u&0x7fffff)!=0);float scale=std::ldexp(1.f,exponent);
  for(int half=0;half<2;half++){
   auto value=half?b:a;auto normalized=_mm512_div_ps(value,_mm512_set1_ps(scale));auto bits=_mm512_castps_si512(normalized);
   auto mag=_mm512_min_ps(_mm512_castsi512_ps(_mm512_and_si512(bits,abs_mask)),_mm512_set1_ps(448.f));auto m=_mm512_castps_si512(mag);
   auto regular=_mm512_sub_epi32(_mm512_srli_epi32(_mm512_add_epi32(m,_mm512_add_epi32(_mm512_set1_epi32(0x7ffff),_mm512_and_si512(_mm512_srli_epi32(m,20),_mm512_set1_epi32(1)))),20),_mm512_set1_epi32(960));
   auto tiny=_mm512_cvttps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(mag,_mm512_set1_ps(512.f)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
   auto code=_mm512_min_epi32(_mm512_mask_blend_epi32(_mm512_cmp_ps_mask(mag,_mm512_set1_ps(0.015625f),_CMP_LT_OQ),regular,tiny),_mm512_set1_epi32(126));
   auto rounded=_mm512_i32gather_ps(code,fp8_values.data(),4);rounded=_mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(rounded),_mm512_and_si512(bits,_mm512_set1_epi32(0x80000000u))));
   _mm512_storeu_ps(output+first+half*16,_mm512_mul_ps(rounded,_mm512_set1_ps(scale)));
  }
#endif
 }return 0;
}
void swiglu(const float* gate,const float* up,uint16_t* out,int n,float limit,float alpha){
 for(int i=0;i<n;i+=16){
  auto g=round_bf16_vec(_mm512_loadu_ps(gate+i)),u=round_bf16_vec(_mm512_loadu_ps(up+i));
  if(limit>0){g=_mm512_min_ps(g,_mm512_set1_ps(limit));u=_mm512_max_ps(_mm512_set1_ps(-limit),_mm512_min_ps(u,_mm512_set1_ps(limit)));}
  auto exp=_ZGVeN16v_expf(_mm512_sub_ps(_mm512_setzero_ps(),g));
  auto h=round_bf16_vec(_mm512_mul_ps(_mm512_mul_ps(_mm512_div_ps(g,_mm512_add_ps(_mm512_set1_ps(1.f),exp)),u),_mm512_set1_ps(alpha)));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),_mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(h),16)));
 }
}


int output_bf16_vector(const float*input,uint16_t*output,int count){
 for(int i=0;i<count;i+=16){
  auto bits=_mm512_castps_si512(_mm512_loadu_ps(input+i));
  const auto exponent=_mm512_set1_epi32(0x7f800000);
  if(_mm512_cmpeq_epi32_mask(_mm512_and_si512(bits,exponent),exponent))return 4;
  auto rounded=_mm512_add_epi32(bits,_mm512_add_epi32(_mm512_set1_epi32(0x7fff),_mm512_and_si512(_mm512_srli_epi32(bits,16),_mm512_set1_epi32(1))));
  auto packed=_mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded,16));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),packed);
 }return 0;
}

struct Task {
    std::vector<uintptr_t> weights,scales;std::vector<int> formats;
    int experts,dim,inter,topk,rows;float limit;
    const uint16_t* x;const int64_t* ids;const float* routing;uint16_t* output;uint64_t* flags;
    std::vector<float> activation,gate,up,hidden,down;
    std::vector<uint16_t> hidden_bf16;
    // Bounded CPU-only storage allocated once with the immutable Task shape.
    // Existing parallel hidden_post rewrites every active32-column block.
    std::vector<int8_t> hidden_codes;
    std::vector<float> hidden_scales;
    std::vector<int32_t> hidden_sums;
    std::vector<int8_t>gu_codes;std::vector<float>gu_scales,gu_result;std::vector<int32_t>gu_sums;std::array<uint64_t,8>gu_stats{};std::array<double,8>gu_phase_us{};bool gu_trace_enabled=false,gu_trace_used=false,gu_trace_snapshot_valid=false;int gu_test_cpu_fail=0;
    std::array<uint64_t,16>whole_stats{};
    std::vector<int>home418;int layer418=-1;
    std::atomic<uint64_t> served{0};std::atomic<int> error{0};
};
struct Receipt418 {std::mutex mutex;std::array<uint64_t,16>v{},epoch427{},generation489{};};
// Private task-level metadata. No clocks, worker writes, GPU calls or numerical work.
// Counters describe attempted host decisions, not successful device execution.
struct GenerationScope489 {
    std::shared_ptr<Receipt418> receipt;
    uint64_t rows=0,positive=0,lookups=0,raw_groups=0,raw_routes=0;
    uint64_t forced_groups=0,forced_routes=0,formed_groups=0,formed_routes=0;
    bool success=false;
    GenerationScope489(std::shared_ptr<Receipt418> r,int n):receipt(std::move(r)),rows(uint64_t(n)){
        if(receipt){std::lock_guard<std::mutex>l(receipt->mutex);auto&v=receipt->generation489;v[3]++;v[15]=1;}
    }
    ~GenerationScope489(){
        if(receipt){std::lock_guard<std::mutex>l(receipt->mutex);auto&v=receipt->generation489;
            v[success?4:5]++;if(success){v[6]+=rows;v[7]+=positive;}
            v[8]+=lookups;v[9]+=raw_groups;v[10]+=raw_routes;v[11]+=forced_groups;v[12]+=forced_routes;
            v[13]+=formed_groups;v[14]+=formed_routes;v[15]=0;
        }
    }
};
std::mutex receipt_registry418_mutex;std::weak_ptr<Receipt418>receipt_registry418;
struct Engine {
    struct Bulk418;std::unique_ptr<Bulk418>bulk418;
    ds370::PolicyHooks370 policy_hooks386{};std::atomic<bool> running386{false};
    DsWholeHooksV299 whole_hooks{};std::mutex whole_mutex;uint64_t whole_epoch=0;bool whole_policy=false;bool gpu1_generation_off489=false;
    DsGuHooksV18 gu_hooks{};std::mutex gu_mutex;uint64_t gu_epoch=0;bool gu_policy=false;
    void* pool;uintptr_t function;int a8,workers,coordinator;ParallelPost post=nullptr;
    Ds41ExactQuantV118 quant_exact=nullptr;Ds41PrequantBatchV118 prequant_batch=nullptr;
    std::array<std::atomic<Task*>,2048> slots{};std::atomic<size_t>slot_count{0};std::vector<std::unique_ptr<Task>> owned;
    std::atomic<bool> stop{false};std::thread thread;std::atomic<int> affinity_error{0};
    std::vector<uint64_t> computed;std::vector<int> actual;
    std::array<std::atomic<uint64_t>,256> total_work{};
    Engine(void* p,uintptr_t f,int m,int w,int cpu):pool(p),function(f),a8(m),workers(w),coordinator(cpu),computed(w),actual(w){
        const char*flag=std::getenv("DS1017_GPU_GU_OVERLAP_V6");if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))throw std::invalid_argument("literal GU overlap flag required");gu_policy=flag&&!std::strcmp(flag,"1");
        const char*whole=std::getenv("DS1017_GPU_WHOLE_EXPERT_OWNER_V299");if(whole&&std::strcmp(whole,"0")&&std::strcmp(whole,"1"))throw std::invalid_argument("literal whole299 flag required");whole_policy=whole&&!std::strcmp(whole,"1");
        const char*off489=std::getenv("DS1017_GPU1_GENERATION_OFF_V489");
        if(off489&&std::strcmp(off489,"0")&&std::strcmp(off489,"1"))throw std::invalid_argument("literal GPU1 generation off489 flag required");
        gpu1_generation_off489=off489&&!std::strcmp(off489,"1");
        if(gpu1_generation_off489&&(!whole_policy||!a8))throw std::invalid_argument("off489 requires whole299 A8 policy");
        Dl_info location{};
        if(!dladdr(reinterpret_cast<void*>(f),&location)||!location.dli_fname)throw std::runtime_error("v66 projection library identity missing");
        void*library=dlopen(location.dli_fname,RTLD_NOW|RTLD_NOLOAD);
        if(!library)throw std::runtime_error("v66 projection library not loaded");
        post=reinterpret_cast<ParallelPost>(dlsym(library,"ds41_wxa8_parallel_post_v66"));
        if(a8){
            quant_exact=reinterpret_cast<Ds41ExactQuantV118>(dlsym(library,"ds41_wxa8_quantize_exact_v118"));
            prequant_batch=reinterpret_cast<Ds41PrequantBatchV118>(dlsym(library,"ds41_wxa8_batch_gemv_prequant_v118"));
        }
        dlclose(library);
        if(!post||(a8&&(!quant_exact||!prequant_batch)))throw std::runtime_error("v118 exact quantizer/prequant batch and v66 post APIs required");
    }
    struct PostContext {Task*task;const std::vector<float>*alpha;Ds41ExactQuantV118 quant_exact;const float*const*gu_sources=nullptr;};
    static int hidden_post(void*raw,int begin,int end){
        auto&context=*static_cast<PostContext*>(raw);auto&t=*context.task;
        const int blocks_per_expert=t.inter/32;
        for(int block=begin;block<end;block++){
            const int expert=block/blocks_per_expert,first=(block%blocks_per_expert)*32;
            const float alpha=(*context.alpha)[expert];if(alpha==0.f)continue;
            auto*hidden=t.hidden_bf16.data()+expert*t.inter+first;
            const float*gate=context.gu_sources?context.gu_sources[expert*2]:t.gate.data()+expert*2*t.inter;const float*up=context.gu_sources?context.gu_sources[expert*2+1]:t.gate.data()+(expert*2+1)*t.inter;
            swiglu(gate+first,up+first,hidden,32,t.limit,alpha);
            if(context.gu_sources&&t.gu_trace_enabled){if(gate!=t.gate.data()+expert*2*t.inter)std::memcpy(t.gate.data()+expert*2*t.inter+first,gate+first,32*sizeof(float));if(up!=t.gate.data()+(expert*2+1)*t.inter)std::memcpy(t.gate.data()+(expert*2+1)*t.inter+first,up+first,32*sizeof(float));}
            int rc=quantize(hidden,t.hidden.data()+expert*t.inter+first,32);if(rc)return rc;
            if(context.quant_exact){
                // Preserve SwiGLU -> BF16 -> original FP8/F32, then invoke the
                // exact frozen-v66 A8 definition on the already rounded block.
                rc=context.quant_exact(t.hidden.data()+expert*t.inter+first,32,
                    t.hidden_codes.data()+expert*t.inter+first,
                    t.hidden_scales.data()+expert*(t.inter/32)+first/32,
                    t.hidden_sums.data()+expert*(t.inter/4)+first/4);
                if(rc)return rc;
            }
        }return 0;
    }
#include "bulk_executor_v427.inc"
    int multiply(Task& t,const std::vector<int>& selected,const float* vectors,bool first,float* output){
        uintptr_t w[12]{},s[12]{},x[12]{},out[12]{};int n[12]{},k[12]{},bits[12]{};int count=0;
        for(int e:selected){
            const int slot=count/(first?2:1);
            for(int p=0;p<(first?2:1);p++){
                const int which=first?p:2,index=e*3+which;
                w[count]=t.weights[index];s[count]=t.scales[index];bits[count]=t.formats[index];
                n[count]=first?t.inter:t.dim;k[count]=first?t.dim:t.inter;
                x[count]=uintptr_t(vectors+(first?0:slot*t.inter));
                out[count]=uintptr_t(output+count*n[count]);count++;
            }
        }
        int rc=a8?reinterpret_cast<A8Batch>(function)(pool,w,s,x,out,n,k,bits,count,computed.data(),actual.data()):
            reinterpret_cast<NativeBatch>(function)(pool,w,s,x,out,n,k,count,computed.data(),actual.data());
        if(!rc)for(int i=0;i<workers;i++)total_work[i].fetch_add(computed[i],std::memory_order_relaxed);
        return rc;
    }
    int run_batched(Task&t){
        if(affinity_error.load())return 6;
        std::vector<int>selected(t.rows*t.topk);std::vector<float>alpha(selected.size());
        for(int row=0;row<t.rows;row++){
            for(int i=0;i<t.topk;i++){
                int64_t id=t.ids[row*t.topk+i];float value=t.routing[row*t.topk+i];
                if(id<0||id>=t.experts||!std::isfinite(value)||value<0)return 1;
                for(int j=0;j<i;j++)if(selected[row*t.topk+j]==id)return 1;
                selected[row*t.topk+i]=int(id);alpha[row*t.topk+i]=value;
            }
            for(int i=0;i<t.topk;i++)for(int j=i+1;j<t.topk;j++)if(selected[row*t.topk+j]<selected[row*t.topk+i]){
                std::swap(selected[row*t.topk+j],selected[row*t.topk+i]);std::swap(alpha[row*t.topk+j],alpha[row*t.topk+i]);}
            int rc=quantize(t.x+row*t.dim,t.activation.data()+row*t.dim,t.dim);if(rc)return rc;
        }
        if(bulk418){int rc=bulk_all418(t,selected,alpha);if(rc)return rc;return output_bf16_vector(t.down.data(),t.output,t.rows*t.topk*t.dim);}
        auto multiply_all=[&](bool first){
            uintptr_t w[96]{},s[96]{},x[96]{},out[96]{};int n[96]{},k[96]{},bits[96]{};int count=0;
            Ds41A8ViewV118 views[96]{};
            for(int i=0;i<int(selected.size());i++)for(int p=0;p<(first?2:1);p++){
                if(alpha[i]==0.f)continue;
                const int projection=selected[i]*3+(first?p:2);w[count]=t.weights[projection];s[count]=t.scales[projection];bits[count]=t.formats[projection];
                n[count]=first?t.inter:t.dim;k[count]=first?t.dim:t.inter;
                x[count]=uintptr_t(first?t.activation.data()+(i/t.topk)*t.dim:t.hidden.data()+i*t.inter);
                out[count]=uintptr_t(first?t.gate.data()+(i*2+p)*t.inter:t.down.data()+i*t.dim);
                if(!first&&a8)views[count]={t.hidden_codes.data()+i*t.inter,
                    t.hidden_scales.data()+i*(t.inter/32),t.hidden_sums.data()+i*(t.inter/4),t.inter};
                count++;
            }
            if(!count)return 0;
            int rc;
            if(a8&&!first)rc=prequant_batch(pool,w,s,x,out,n,k,bits,count,views,computed.data(),actual.data());
            else rc=a8?reinterpret_cast<A8Batch>(function)(pool,w,s,x,out,n,k,bits,count,computed.data(),actual.data()):reinterpret_cast<NativeBatch>(function)(pool,w,s,x,out,n,k,count,computed.data(),actual.data());
            if(!rc)for(int i=0;i<workers;i++)total_work[i].fetch_add(computed[i],std::memory_order_relaxed);
            return rc;
        };
        int rc=multiply_all(true);if(rc)return rc;
        for(int i=0;i<int(selected.size());i++)if(alpha[i]==0.f)std::fill_n(t.down.data()+i*t.dim,t.dim,0.f);
        PostContext post_context{&t,&alpha,quant_exact};
        rc=post(pool,hidden_post,&post_context,int(selected.size())*(t.inter/32));if(rc)return rc;
        rc=multiply_all(false);if(rc)return rc;
        return output_bf16_vector(t.down.data(),t.output,t.rows*t.topk*t.dim);
    }
    int attach_gu(const DsGuHooksV18*h){
        std::unique_lock<std::mutex>lock(gu_mutex,std::try_to_lock);if(!lock.owns_lock())return 2;
        if(h&&whole_hooks.owner)return 1;
        if(h&&(h->struct_size!=sizeof(*h)||h->version!=DsGuVersionV18||!h->owner||!h->retain||!h->release||!h->contains||!h->begin||!h->finish||!h->release_view||!h->abort||!gu_policy||!a8))return 1;
        DsGuHooksV18 next{};if(h){next=*h;next.retain(next.owner);}auto previous=gu_hooks;gu_hooks=next;if(previous.owner)previous.release(previous.owner);return 0;
    }
    int run_overlap(Task&t){
        auto stamp=std::chrono::steady_clock::now();int phase=0;auto mark=[&](){auto now=std::chrono::steady_clock::now();t.gu_phase_us[phase++]=std::chrono::duration<double,std::micro>(now-stamp).count();stamp=now;};
        std::vector<int>selected(t.rows*t.topk);std::vector<float>alpha(selected.size());
        for(int row=0;row<t.rows;row++){
            for(int i=0;i<t.topk;i++){
                int64_t id=t.ids[row*t.topk+i];float value=t.routing[row*t.topk+i];
                if(id<0||id>=t.experts||!std::isfinite(value)||value<0)return 1;
                for(int j=0;j<i;j++)if(selected[row*t.topk+j]==id)return 1;
                selected[row*t.topk+i]=int(id);alpha[row*t.topk+i]=value;
            }
            for(int i=0;i<t.topk;i++)for(int j=i+1;j<t.topk;j++)if(selected[row*t.topk+j]<selected[row*t.topk+i]){
                std::swap(selected[row*t.topk+j],selected[row*t.topk+i]);std::swap(alpha[row*t.topk+j],alpha[row*t.topk+i]);}
            int rc=quantize(t.x+row*t.dim,t.activation.data()+row*t.dim,t.dim);if(rc)return rc;
        }
        std::array<int,384>count{};std::array<std::array<int,6>,384>slot{};std::array<bool,384>gpu{};
        for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f){int e=selected[i];slot[e][count[e]++]=i;}
        DsGuCommandV18 cmd{};cmd.struct_size=sizeof(cmd);cmd.version=DsGuVersionV18;cmd.epoch=++gu_epoch;cmd.rows=t.rows;cmd.dim=t.dim;cmd.inter=t.inter;
        for(int e=0;e<t.experts&&cmd.groups<36;e++)if(count[e]==1||count[e]==2||count[e]==6){
            if(t.formats[e*3]!=0||t.formats[e*3+1]!=0)continue;
            int present=gu_hooks.contains(gu_hooks.owner,e,t.weights[e*3],t.weights[e*3+1]);if(present<0)return 9;if(!present)continue;
            gpu[e]=true;auto&g=cmd.group[cmd.groups++];g.expert=e;g.count=count[e];g.output_words=cmd.output_words;g.gate_key=t.weights[e*3];g.up_key=t.weights[e*3+1];cmd.output_words+=2*g.count*t.inter;
            for(int j=0;j<count[e];j++){g.query[j]=slot[e][j]/t.topk;g.route_slot[j]=slot[e][j];}
        }
        if(!cmd.groups){t.gu_stats[7]++;return run_batched(t);}
        std::array<bool,6>active{};for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f)active[i/t.topk]=true;
        for(int q=0;q<t.rows;q++)if(active[q]){
            int rc=quant_exact(t.activation.data()+q*t.dim,t.dim,t.gu_codes.data()+q*t.dim,t.gu_scales.data()+q*(t.dim/32),t.gu_sums.data()+q*(t.dim/4));if(rc)return rc;
            cmd.view[q]={t.gu_codes.data()+q*t.dim,t.gu_scales.data()+q*(t.dim/32),t.gu_sums.data()+q*(t.dim/4),t.dim};
        }
        uintptr_t w[96]{},s[96]{},x[96]{},out[96]{};int n[96]{},k[96]{},bits[96]{};Ds41A8ViewV118 views[96]{};int entries=0;
        for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f&&!gpu[selected[i]])for(int p=0;p<2;p++){
            int projection=selected[i]*3+p;w[entries]=t.weights[projection];s[entries]=t.scales[projection];bits[entries]=t.formats[projection];n[entries]=t.inter;k[entries]=t.dim;x[entries]=uintptr_t(t.activation.data()+(i/t.topk)*t.dim);out[entries]=uintptr_t(t.gate.data()+(i*2+p)*t.inter);views[entries]=cmd.view[i/t.topk];entries++;
        }
        struct Pending {DsGuHooksV18*h;void*ticket=nullptr;uint64_t epoch=0;~Pending(){if(ticket&&h->abort(h->owner,ticket,epoch))std::terminate();}}pending{&gu_hooks,nullptr,cmd.epoch};
        t.gu_stats[0]=cmd.epoch;t.gu_stats[1]=cmd.groups;t.gu_stats[2]=cmd.output_words/(2*t.inter);t.gu_stats[3]=entries;
        mark();int rc=gu_hooks.begin(gu_hooks.owner,&cmd,&pending.ticket);if(rc||!pending.ticket)return rc?rc:9;t.gu_stats[5]++;mark();
        if(t.gu_test_cpu_fail){t.gu_test_cpu_fail=0;return 91;}
        if(entries){rc=prequant_batch(pool,w,s,x,out,n,k,bits,entries,views,computed.data(),actual.data());t.gu_stats[4]++;if(!rc)for(int i=0;i<workers;i++)total_work[i].fetch_add(computed[i],std::memory_order_relaxed);}
        if(rc)return rc;
        const float*borrowed_result=nullptr;mark();rc=gu_hooks.finish(gu_hooks.owner,pending.ticket,cmd.epoch,&borrowed_result,cmd.output_words);if(rc||!borrowed_result)return rc?rc:9;t.gu_stats[6]++;t.gu_trace_used=true;mark();
        std::array<const float*,96>gu_sources{};for(int i=0;i<int(selected.size())*2;i++)gu_sources[i]=t.gate.data()+i*t.inter;
        for(unsigned e=0;e<cmd.groups;e++){const auto&g=cmd.group[e];for(unsigned j=0;j<g.count;j++)for(int p=0;p<2;p++)gu_sources[g.route_slot[j]*2+p]=borrowed_result+g.output_words+(p*g.count+j)*t.inter;}
        for(int i=0;i<int(selected.size());i++)if(alpha[i]==0.f)std::fill_n(t.down.data()+i*t.dim,t.dim,0.f);
        mark();PostContext post_context{&t,&alpha,quant_exact,gu_sources.data()};rc=post(pool,hidden_post,&post_context,int(selected.size())*(t.inter/32));if(rc)return rc;t.gu_trace_snapshot_valid=t.gu_trace_enabled;rc=gu_hooks.release_view(gu_hooks.owner,pending.ticket,cmd.epoch);if(rc)return rc;pending.ticket=nullptr;mark();
        // Literal original DN batch, original ordering/views and output conversion.
        entries=0;for(int i=0;i<int(selected.size());i++){
            if(alpha[i]==0.f)continue;const int projection=selected[i]*3+2;w[entries]=t.weights[projection];s[entries]=t.scales[projection];bits[entries]=t.formats[projection];n[entries]=t.dim;k[entries]=t.inter;x[entries]=uintptr_t(t.hidden.data()+i*t.inter);out[entries]=uintptr_t(t.down.data()+i*t.dim);views[entries]={t.hidden_codes.data()+i*t.inter,t.hidden_scales.data()+i*(t.inter/32),t.hidden_sums.data()+i*(t.inter/4),t.inter};entries++;
        }
        if(entries){rc=prequant_batch(pool,w,s,x,out,n,k,bits,entries,views,computed.data(),actual.data());if(rc)return rc;for(int i=0;i<workers;i++)total_work[i].fetch_add(computed[i],std::memory_order_relaxed);}
        mark();rc=output_bf16_vector(t.down.data(),t.output,t.rows*t.topk*t.dim);mark();return rc;
    }
    bool idle386()const{
        if(running386.load(std::memory_order_acquire))return false;
        for(size_t i=0,n=slot_count.load(std::memory_order_acquire);i<n;i++){
            auto*t=slots[i].load(std::memory_order_acquire);
            if(t&&__atomic_load_n(t->flags,__ATOMIC_ACQUIRE)==1)return false;
        }
        return true;
    }
    int attach_policy386(const ds370::PolicyHooks370*h){
        std::unique_lock<std::mutex>lock(whole_mutex,std::try_to_lock);
        if(!lock.owns_lock()||!idle386())return 2;
        if(h&&(policy_hooks386.owner||h->struct_bytes!=sizeof(*h)||h->version!=370||!h->owner||!h->retain||!h->release||!h->before||!h->after||!whole_hooks.owner||h->owner!=whole_hooks.owner||!whole_policy||!a8))return 1;
        ds370::PolicyHooks370 next{};if(h){next=*h;next.retain(next.owner);}
        auto previous=policy_hooks386;policy_hooks386=next;
        if(previous.owner)previous.release(previous.owner);return 0;
    }
    int attach_whole(const DsWholeHooksV299*h){
        std::unique_lock<std::mutex>lock(whole_mutex,std::try_to_lock);if(!lock.owns_lock())return 2;
        if(h&&(h->struct_size!=sizeof(*h)||h->version!=DsWholeVersionV299||!h->owner||!h->retain||!h->release||!h->contains||!h->begin||!h->finish||!h->release_view||!h->abort||!whole_policy||!a8||gu_hooks.owner))return 1;
        if(!idle386()||(policy_hooks386.owner&&(!h||h->owner!=policy_hooks386.owner)))return 2;
        DsWholeHooksV299 next{};if(h){next=*h;next.retain(next.owner);}auto previous=whole_hooks;whole_hooks=next;if(previous.owner)previous.release(previous.owner);return 0;
    }
    int run_whole(Task&t){
        std::unique_lock<std::mutex>lock(whole_mutex,std::try_to_lock);if(!lock.owns_lock())return 9;
        GenerationScope489 generation489(bulk418?bulk418->receipt:nullptr,t.rows);
        std::vector<int>selected(t.rows*t.topk);std::vector<float>alpha(selected.size());
        std::array<int,384>count{};std::array<std::array<int,6>,384>slot{};std::array<bool,384>gpu{};
        // Literal per-row validation and sort from18. No routing/alpha change.
        for(int row=0;row<t.rows;row++){
            for(int i=0;i<t.topk;i++){
                int64_t id=t.ids[row*t.topk+i];float value=t.routing[row*t.topk+i];
                if(id<0||id>=t.experts||!std::isfinite(value)||value<0)return 1;
                for(int j=0;j<i;j++)if(selected[row*t.topk+j]==id)return 1;
                selected[row*t.topk+i]=int(id);alpha[row*t.topk+i]=value;
            }
            for(int i=0;i<t.topk;i++)for(int j=i+1;j<t.topk;j++)if(selected[row*t.topk+j]<selected[row*t.topk+i]){
                std::swap(selected[row*t.topk+j],selected[row*t.topk+i]);std::swap(alpha[row*t.topk+j],alpha[row*t.topk+i]);}
        }
        for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f){int e=selected[i];slot[e][count[e]++]=i;}
        for(int e=0;e<t.experts;e++)generation489.positive+=uint64_t(count[e]);
        // Counts are literal positive-alpha counts of ALL experts, before ANY
        // resident lookup. Expert0 gate is immutable layer identity, not a hit.
        const bool observe386=policy_hooks386.owner!=nullptr;
        if(observe386){
            std::array<uint8_t,384>counts386{};
            for(int e=0;e<384;e++){if(count[e]<0||count[e]>6)return 1;counts386[e]=uint8_t(count[e]);}
            int status=policy_hooks386.before(policy_hooks386.owner,t.weights[0],unsigned(t.rows),counts386.data(),384);
            if(status)return status;
        }
        // All original numerical work and its Pending ticket live INSIDE this
        // lambda. Destruction/abort precedes the matching after callback.
        int status386=9;
        try{status386=[&]()->int{
        DsWholeCommandV299 cmd{};cmd.struct_size=sizeof(cmd);cmd.version=DsWholeVersionV299;cmd.epoch=++whole_epoch;cmd.rows=t.rows;cmd.dim=t.dim;cmd.inter=t.inter;cmd.route_words=t.rows*t.topk*t.dim;
        for(int e=0;e<t.experts&&cmd.groups<36;e++)if(count[e]>=1&&count[e]<=6){
            if(t.formats[e*3]||t.formats[e*3+1]||t.formats[e*3+2])continue;
            int present=whole_hooks.contains(whole_hooks.owner,e,t.weights[e*3],t.weights[e*3+1],t.weights[e*3+2]);generation489.lookups++;if(present<0)return 9;
            if(present){generation489.raw_groups++;generation489.raw_routes+=uint64_t(count[e]);}
            // Preserve the real typed lookup and its validation BEFORE ablation.
            // With no formed GPU groups the literal all-miss CPU path below runs.
            if(gpu1_generation_off489){if(present){generation489.forced_groups++;generation489.forced_routes+=uint64_t(count[e]);}continue;}
            if(!present)continue;
            gpu[e]=true;auto&g=cmd.group[cmd.groups++];g.expert=e;g.count=count[e];g.gate_key=t.weights[e*3];g.up_key=t.weights[e*3+1];g.down_key=t.weights[e*3+2];
            for(int j=0;j<count[e];j++){int r=slot[e][j];g.query[j]=r/t.topk;g.route_slot[j]=r;g.alpha[j]=alpha[r];}
        }
        generation489.formed_groups=cmd.groups;for(unsigned g=0;g<cmd.groups;g++)generation489.formed_routes+=cmd.group[g].count;
        if(!cmd.groups){t.whole_stats[1]++;return run_batched(t);}
        // Original FP8 boundary then actual CPU118 A8, shared by both consumers.
        std::array<bool,6>active{};for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f)active[i/t.topk]=true;
        for(int q=0;q<t.rows;q++){
            int rc=quantize(t.x+q*t.dim,t.activation.data()+q*t.dim,t.dim);if(rc)return rc;
            if(active[q]){rc=quant_exact(t.activation.data()+q*t.dim,t.dim,t.gu_codes.data()+q*t.dim,t.gu_scales.data()+q*(t.dim/32),t.gu_sums.data()+q*(t.dim/4));if(rc)return rc;cmd.view[q]={t.gu_codes.data()+q*t.dim,t.gu_scales.data()+q*(t.dim/32),t.gu_sums.data()+q*(t.dim/4),t.dim};}
        }
        struct Pending {DsWholeHooksV299*h;void*ticket=nullptr;uint64_t epoch=0;~Pending(){if(ticket&&h->abort(h->owner,ticket,epoch))std::terminate();}}pending{&whole_hooks,nullptr,cmd.epoch};
        int rc=whole_hooks.begin(whole_hooks.owner,&cmd,&pending.ticket);
        if(rc==DsWholeDeclinedBeforeSubmitV299&&!pending.ticket){t.whole_stats[2]++;return run_batched(t);}
        if(rc||!pending.ticket)return rc?rc:9;
        if(t.gu_test_cpu_fail){t.gu_test_cpu_fail=0;return 91;}
        t.whole_stats[0]=cmd.epoch;t.whole_stats[3]+=cmd.groups;t.gu_trace_used=true;t.gu_trace_snapshot_valid=false;
        if(bulk418){
            auto cpu_alpha=alpha;uint64_t misses=0;for(size_t i=0;i<alpha.size();i++){if(gpu[selected[i]]||alpha[i]==0.f){cpu_alpha[i]=0.f;std::fill_n(t.down.data()+i*t.dim,t.dim,0.f);}else misses++;}
            rc=bulk418->run(t,selected,cpu_alpha);if(rc)return rc;
            t.whole_stats[4]+=2*misses;t.whole_stats[5]+=misses;if(misses)t.whole_stats[8]++;else t.whole_stats[9]++;
        }else{
        uintptr_t w[96]{},s[96]{},x[96]{},out[96]{};int n[96]{},k[96]{},bits[96]{};Ds41A8ViewV118 views[96]{};int entries=0;
        for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f&&!gpu[selected[i]])for(int p=0;p<2;p++){
            int projection=selected[i]*3+p;w[entries]=t.weights[projection];s[entries]=t.scales[projection];bits[entries]=t.formats[projection];n[entries]=t.inter;k[entries]=t.dim;x[entries]=uintptr_t(t.activation.data()+(i/t.topk)*t.dim);out[entries]=uintptr_t(t.gate.data()+(i*2+p)*t.inter);views[entries]=cmd.view[i/t.topk];entries++;
        }
        if(entries){rc=prequant_batch(pool,w,s,x,out,n,k,bits,entries,views,computed.data(),actual.data());if(rc)return rc;for(int i=0;i<workers;i++)total_work[i].fetch_add(computed[i],std::memory_order_relaxed);}
        t.whole_stats[4]+=entries;
        auto cpu_alpha=alpha;for(int i=0;i<int(selected.size());i++)if(alpha[i]==0.f||gpu[selected[i]]){cpu_alpha[i]=0.f;std::fill_n(t.down.data()+i*t.dim,t.dim,0.f);}
        // CPU hits are excluded only from work, not original mathematical routes.
        // Literal18 hidden_post gets original alpha on each CPU miss; GPU routes
        // already run the qualified276 hidden chain on independent device streams.
        PostContext pc{&t,&cpu_alpha,quant_exact};if(entries){rc=post(pool,hidden_post,&pc,int(selected.size())*(t.inter/32));if(rc)return rc;t.whole_stats[8]++;}else t.whole_stats[9]++;
        entries=0;for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f&&!gpu[selected[i]]){
            int projection=selected[i]*3+2;w[entries]=t.weights[projection];s[entries]=t.scales[projection];bits[entries]=t.formats[projection];n[entries]=t.dim;k[entries]=t.inter;x[entries]=uintptr_t(t.hidden.data()+i*t.inter);out[entries]=uintptr_t(t.down.data()+i*t.dim);views[entries]={t.hidden_codes.data()+i*t.inter,t.hidden_scales.data()+i*(t.inter/32),t.hidden_sums.data()+i*(t.inter/4),t.inter};entries++;
        }
        if(entries){rc=prequant_batch(pool,w,s,x,out,n,k,bits,entries,views,computed.data(),actual.data());if(rc)return rc;for(int i=0;i<workers;i++)total_work[i].fetch_add(computed[i],std::memory_order_relaxed);}
        t.whole_stats[5]+=entries;
        }
        rc=output_bf16_vector(t.down.data(),t.output,t.rows*t.topk*t.dim);if(rc)return rc;
        const uint16_t*borrowed=nullptr;rc=whole_hooks.finish(whole_hooks.owner,pending.ticket,cmd.epoch,&borrowed,cmd.route_words);if(rc||!borrowed)return rc?rc:9;
        for(int i=0;i<int(selected.size());i++)if(alpha[i]!=0.f&&gpu[selected[i]]){std::memcpy(t.output+i*t.dim,borrowed+i*t.dim,t.dim*sizeof(uint16_t));t.whole_stats[6]++;}
        rc=whole_hooks.release_view(whole_hooks.owner,pending.ticket,cmd.epoch);if(rc)return rc;pending.ticket=nullptr;t.whole_stats[7]++;return 0;
        }();}catch(...){
            if(observe386)(void)policy_hooks386.after(policy_hooks386.owner,9);
            throw; //Original worker converts the exception to Task error9.
        }
        if(observe386){int after386=policy_hooks386.after(policy_hooks386.owner,status386);if(!status386)status386=after386;}
        generation489.success=status386==0;
        return status386;
    }
    int run(Task&t){
        t.gu_stats.fill(0);t.gu_trace_used=false;t.gu_trace_snapshot_valid=false;
        if(whole_policy&&whole_hooks.owner&&a8&&t.rows<=6&&t.topk==6&&t.dim==5120&&t.inter==2304&&t.limit==10.f)return run_whole(t);
        if(!gu_policy||!gu_hooks.owner||!a8||t.rows>6||t.topk!=6||t.dim!=5120||t.inter!=2304)return run_batched(t);
        std::unique_lock<std::mutex>lock(gu_mutex,std::try_to_lock);if(!lock.owns_lock())return 9;
        // Missing-cache/unsupported-count decision BEFORE source or A8 quantization.
        std::array<int,384>count{};for(int i=0;i<t.rows*t.topk;i++){
            if(t.ids[i]<0||t.ids[i]>=t.experts||!std::isfinite(t.routing[i])||t.routing[i]<0)return run_batched(t);if(t.routing[i]!=0.f)count[t.ids[i]]++;
        }
        bool any=false;for(int e=0;e<t.experts;e++)if((count[e]==1||count[e]==2||count[e]==6)&&t.formats[e*3]==0&&t.formats[e*3+1]==0){int present=gu_hooks.contains(gu_hooks.owner,e,t.weights[e*3],t.weights[e*3+1]);if(present<0)return 9;any|=present!=0;}
        if(!any){t.gu_stats[7]++;return run_batched(t);}return run_overlap(t);
    }
    void start(){thread=std::thread([this]{
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(coordinator,&mask);
        if(sched_setaffinity(0,sizeof(mask),&mask))affinity_error=6;
        auto last=std::chrono::steady_clock::now();unsigned polls=0;
        while(!stop.load(std::memory_order_relaxed)){
            bool active=false;
            for(size_t index=0,index_limit=slot_count.load(std::memory_order_acquire);index<index_limit;index++){Task* task=slots[index].load(std::memory_order_acquire);if(!task)continue;
                if(__atomic_load_n(task->flags,__ATOMIC_ACQUIRE)!=1)continue;
                __atomic_store_n(task->flags,uint64_t(0),__ATOMIC_RELEASE);
                running386.store(true,std::memory_order_release);
                int rc=9;try{rc=run(*task);}catch(...){rc=9;}
                running386.store(false,std::memory_order_release);
                task->error=rc;
                if(rc)std::fill(task->output,task->output+task->rows*task->topk*task->dim,uint16_t(0x7fc0));
                task->served++;__atomic_store_n(task->flags+1,uint64_t(1),__ATOMIC_RELEASE);active=true;
            }
            if(active){last=std::chrono::steady_clock::now();polls=0;}
            else if(++polls%1024==0&&std::chrono::steady_clock::now()-last>std::chrono::milliseconds(50))
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            else __builtin_ia32_pause();
        }
    });}
    ~Engine(){stop=true;if(thread.joinable())thread.join();bulk418.reset();if(gu_hooks.owner)gu_hooks.release(gu_hooks.owner);if(policy_hooks386.owner)policy_hooks386.release(policy_hooks386.owner);if(whole_hooks.owner)whole_hooks.release(whole_hooks.owner);}
};
using MemOp=int(*)(void*,unsigned long long,unsigned long long,unsigned int);
using HostMap=int(*)(unsigned long long*,void*,unsigned int);
using HostFunc=decltype(&cuLaunchHostFunc);
void* driver=nullptr;MemOp write64=nullptr,wait64=nullptr;HostMap map_host=nullptr;HostFunc host_func=nullptr;
bool resolve_driver(){
    if(driver)return write64&&wait64&&map_host&&host_func;
    driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);if(!driver)return false;
    auto symbol=[](const char* name){void* value=dlsym(driver,(std::string(name)+"_v2").c_str());return value?value:dlsym(driver,name);};
    write64=reinterpret_cast<MemOp>(symbol("cuStreamWriteValue64"));wait64=reinterpret_cast<MemOp>(symbol("cuStreamWaitValue64"));
    map_host=reinterpret_cast<HostMap>(symbol("cuMemHostGetDevicePointer"));
    // CUDA13.4's _v2 takes an additional syncMode argument. Bind the exact
    // three-argument legacy ABI declared by cuda.h, never guess a versioned ABI.
    host_func=reinterpret_cast<HostFunc>(dlsym(driver,"cuLaunchHostFunc"));
    return write64&&wait64&&map_host&&host_func;
}
void submit_host(void* value){auto& t=*static_cast<Task*>(value);__atomic_store_n(t.flags+1,uint64_t(0),__ATOMIC_RELEASE);__atomic_store_n(t.flags,uint64_t(1),__ATOMIC_RELEASE);}
void wait_host(void* value){auto& t=*static_cast<Task*>(value);while(__atomic_load_n(t.flags+1,__ATOMIC_ACQUIRE)!=1)std::this_thread::sleep_for(std::chrono::microseconds(10));}
}
extern "C" {
int ds41_moe_bf16_output_vec(const float*input,uint16_t*out,int n){if(!input||!out||n<16||n%16||n>393216)return 1;return output_bf16_vector(input,out,n);}
int ds41_moe_task_batch_rows(){return 8;}
void* ds41_moe_engine_create(void* pool,uintptr_t function,int a8,int workers,int cpu){
    const char*bulk=std::getenv("DS1017_CPU_NUMA_BULK_V418");if(bulk&&!std::strcmp(bulk,"1"))return nullptr;
    if(!pool||!function||(a8!=0&&a8!=1)||workers<1||workers>256||cpu<0||cpu>=CPU_SETSIZE)return nullptr;
    try{auto e=std::make_unique<Engine>(pool,function,a8,workers,cpu);return e.release();}catch(...){return nullptr;}
}
void ds41_moe_engine_destroy(void* value){delete static_cast<Engine*>(value);}
void* ds41_moe_task_create(void* value,const uintptr_t* weights,const uintptr_t* scales,const int* bits,
    int experts,int dim,int inter,int topk,int rows,float limit,const uint16_t* x,const int64_t* ids,const float* routing,uint16_t* output,uint64_t* flags){
    if(!value||!weights||!scales||!bits||!x||!ids||!routing||!output||!flags||uintptr_t(flags)%8||
       experts<1||experts>384||dim<32||dim>8192||inter<32||inter>8192||dim%32||inter%32||topk<1||topk>6||topk>experts||rows<1||rows>8||!std::isfinite(limit)||limit<0)return nullptr;
    auto& engine=*static_cast<Engine*>(value);if(engine.owned.size()>=2048)return nullptr;
    for(int e=0;e<experts;e++)for(int p=0;p<3;p++){
        int i=e*3+p,b=bits[i];int n=p==2?dim:inter,k=p==2?inter:dim;
        if(!weights[i]||(!engine.a8&&b!=0)||(b==0&&!scales[i])||(b!=0&&b!=2&&b!=3&&b!=4&&b!=6&&b!=8))return nullptr;
        uint64_t size=uint64_t(n)*(b==0?k/2+k/32:k*b/8+k/32*2);if(size>64*1024*1024)return nullptr;
    }
    try{
        auto t=std::make_unique<Task>();t->weights.assign(weights,weights+experts*3);t->scales.assign(scales,scales+experts*3);t->formats.assign(bits,bits+experts*3);
        t->experts=experts;t->dim=dim;t->inter=inter;t->topk=topk;t->rows=rows;t->limit=limit;t->x=x;t->ids=ids;t->routing=routing;t->output=output;t->flags=flags;
        t->activation.resize(rows*dim);t->gate.resize(rows*topk*2*inter);t->hidden.resize(rows*topk*inter);t->hidden_bf16.resize(rows*topk*inter);t->down.resize(rows*topk*dim);
        if(engine.a8){t->hidden_codes.resize(rows*topk*inter);t->hidden_scales.resize(rows*topk*(inter/32));t->hidden_sums.resize(rows*topk*(inter/4));}
        if(engine.gu_policy||engine.whole_policy||engine.bulk418){t->gu_codes.resize(rows*dim);t->gu_scales.resize(rows*(dim/32));t->gu_sums.resize(rows*(dim/4));t->gu_result.resize(rows*topk*2*inter);}
        if(engine.bulk418&&engine.bulk418->validate_task(*t))return nullptr;
        Task* pointer=t.get();engine.owned.push_back(std::move(t));engine.slots[engine.owned.size()-1].store(pointer,std::memory_order_release);engine.slot_count.store(engine.owned.size(),std::memory_order_release);return pointer;
    }catch(...){return nullptr;}
}
int ds41_moe_task_run(void* e,void* t){if(!e||!t)return 1;try{return static_cast<Engine*>(e)->run(*static_cast<Task*>(t));}catch(...){return 9;}}
int ds41_moe_task_status(void* value,uint64_t* served){if(!value||!served)return 1;auto& t=*static_cast<Task*>(value);*served=t.served.load();return t.error.load();}
int ds41_moe_engine_work(void* value,uint64_t* counts,int count){if(!value||!counts)return 1;auto& e=*static_cast<Engine*>(value);if(count!=e.workers)return 1;for(int i=0;i<count;i++)counts[i]=e.total_work[i].load();return e.affinity_error.load();}
int ds41_moe_fp8_quant(const uint16_t* x,float* out,int columns){if(!x||!out||columns<32||columns>8192||columns%32)return 1;return quantize(x,out,columns);}
int ds41_moe_map_host(uintptr_t host,uint64_t* device){if(!host||!device||!resolve_driver())return 801;unsigned long long result=0;int rc=map_host(&result,reinterpret_cast<void*>(host),0);if(!rc)*device=result;return rc;}
int ds41_moe_stream_submit(uintptr_t,uint64_t,uint64_t){return 801;}
int ds41_moe_stream_wait(uintptr_t,uint64_t){return 801;}
int ds41_moe_host_submit(uintptr_t,void*){return 801;}
int ds41_moe_host_wait(uintptr_t,void*){return 801;}
}

extern "C" int ds41_moe_trace_v6(void*value,DsTraceV6*out){
 if(!value||!out)return 1;auto&t=*static_cast<Task*>(value);
 if(t.gu_trace_used&&!t.gu_trace_snapshot_valid)return 1;
 *out={sizeof(DsTraceV6),6,uint32_t(t.rows),uint32_t(t.topk),uint32_t(t.dim),uint32_t(t.inter),t.activation.data(),t.gate.data(),t.down.data(),t.hidden_bf16.data(),t.output,t.hidden_codes.data(),t.hidden_scales.data(),t.hidden_sums.data(),{t.flags[0],t.flags[1]},t.served.load(),t.error.load(),0};return 0;
}

extern "C" int ds41_moe_gu_attach_v18(void*value,const DsGuHooksV18*h){if(!value)return 1;try{return static_cast<Engine*>(value)->attach_gu(h);}catch(...){return 9;}}
extern "C" int ds41_moe_gu_stats_v6(void*value,uint64_t*out,int count){if(!value||!out||count!=8)return 1;auto&t=*static_cast<Task*>(value);std::copy(t.gu_stats.begin(),t.gu_stats.end(),out);return 0;}

extern "C" int ds41_moe_gu_test_cpu_fail_v6(void*value){const char*f=std::getenv("DS1017_GPU_GU_TEST_FAULTS_V6");if(!value||!f||std::strcmp(f,"1"))return 1;static_cast<Task*>(value)->gu_test_cpu_fail=1;return 0;}

extern "C" int ds41_moe_gu_phase_v15(void*value,double*out,int n){if(!value||!out||n!=8)return 1;auto&t=*static_cast<Task*>(value);std::copy(t.gu_phase_us.begin(),t.gu_phase_us.end(),out);return 0;}

extern "C" int ds41_moe_gu_trace_enable_v18(void*value,int enable){if(!value||(enable!=0&&enable!=1))return 1;static_cast<Task*>(value)->gu_trace_enabled=enable!=0;return 0;}

extern "C" int ds41_moe_whole_attach_v299(void*e,const DsWholeHooksV299*h){if(!e)return 1;try{return static_cast<Engine*>(e)->attach_whole(h);}catch(...){return 9;}}
extern "C" int ds41_moe_whole_stats_v299(void*t,uint64_t*out,int n){if(!t||!out||n!=16)return 1;auto&v=*static_cast<Task*>(t);std::copy(v.whole_stats.begin(),v.whole_stats.end(),out);return 0;}

extern "C" int ds386_moe_policy_attach(void*engine,const ds370::PolicyHooks370*hooks)noexcept{
 if(!engine)return 1;try{return static_cast<Engine*>(engine)->attach_policy386(hooks);}catch(...){return 9;}
}

extern "C" void*ds418_moe_engine_create(const Ds418Pools*b){
 if(!b||b->struct_bytes!=sizeof(*b)||b->version!=418||!b->pool[0]||!b->pool[1]||b->pool[0]==b->pool[1]||!b->gemv_function||b->workers[0]<1||b->workers[0]>127||b->workers[1]<1||b->workers[1]>126||b->coordinator[0]!=127||b->coordinator[1]!=254||b->reserved[0]||b->reserved[1])return nullptr;
 try{auto e=std::make_unique<Engine>(b->pool[1],b->gemv_function,1,b->workers[0]+b->workers[1],254);e->enable_bulk418(*b);{std::lock_guard<std::mutex>l(e->bulk418->receipt->mutex);e->bulk418->receipt->generation489={489,16,uint64_t(e->gpu1_generation_off489)};}return e.release();}catch(...){return nullptr;}
}
extern "C" int ds418_moe_engine_info(void*value,uint64_t*out,int n){
 if(!value||!out||n!=16)return 1;auto&e=*static_cast<Engine*>(value);if(!e.bulk418)return 2;auto r=e.bulk418->receipt;std::lock_guard<std::mutex>l(r->mutex);if(r->v[14])return 2;std::copy(r->v.begin(),r->v.end(),out);return 0;
}
extern "C" int ds418_moe_process_info(uint64_t*out,int n){
 if(!out||n!=16)return 1;std::shared_ptr<Receipt418>r;{std::lock_guard<std::mutex>l(receipt_registry418_mutex);r=receipt_registry418.lock();}if(!r)return 2;
 std::lock_guard<std::mutex>l(r->mutex);if(r->v[14])return 2;std::copy(r->v.begin(),r->v.end(),out);return 0;
}

extern "C" int ds427_moe_epoch_info(uint64_t*out,int n){
 if(!out||n!=16)return 1;std::shared_ptr<Receipt418>r;{std::lock_guard<std::mutex>l(receipt_registry418_mutex);r=receipt_registry418.lock();}if(!r)return 2;
 std::lock_guard<std::mutex>l(r->mutex);if(r->v[14])return 2;std::copy(r->epoch427.begin(),r->epoch427.end(),out);return 0;
}

// Existing N1 weak receipt identifies the ACTUAL split-home Task engine. A strong
// borrow is released only after the registry lock has been dropped. No Engine cast.
extern "C" int ds489_moe_gpu1_generation_info(uint64_t*out,int n){
 if(!out||n!=16)return 1;std::fill_n(out,16,uint64_t(0));
 std::shared_ptr<Receipt418>r;{std::lock_guard<std::mutex>l(receipt_registry418_mutex);r=receipt_registry418.lock();}if(!r)return 2;
 std::lock_guard<std::mutex>l(r->mutex);if(r->v[14]||r->generation489[15]||r->generation489[0]!=489)return 2;
 std::copy(r->generation489.begin(),r->generation489.end(),out);return 0;
}
