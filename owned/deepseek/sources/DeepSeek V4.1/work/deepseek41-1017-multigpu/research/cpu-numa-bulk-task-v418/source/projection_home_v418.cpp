#include "bulk_home_v418.h"
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <new>
#include <climits>
#include <cerrno>
#include <cstdio>
// v118 DOWN-only exact Task-owned A8 views; copied from frozen v66, not v115.
// v66: reuse the existing physical worker pool for independent per32 SwiGLU/FP8 postprocessing.
#include <array>
#include "native_a8_prequant_views_v118.hpp"
// Variant v2: deduplicate activation rounding only within one immutable batch.
// Batch several independent projections with the proven native row arithmetic.
// The barrier is per batch, not per expert/projection. No model state is owned.
#include "cpu_mxfp4.cpp"

// Separate W4/W2/3/6/8-A8 experiment. Per32 activation rounding is intentional;
// it is NEVER source calibration or a quality-admitted production path.
#define vector_row own_v2_vector_row
#define scalar_row own_v2_scalar_row
#include "own_uniform_cpu_v2.cpp"
#undef scalar_row
#undef vector_row
#include <cfenv>
#include <string>
#include <stdexcept>
#include "native_packed_output16_prefetch_many_v125.hpp"
#include "../../cpu-pair32-static-v143/source/pair_pool_v143.h"
#include "pair_leaf_v143.hpp"

namespace {
using ParallelCallback=int(*)(void*,int,int);
using RowFunction=float(*)(const uint8_t*,const uint8_t*,const int8_t*,const float*,const int*,int);
template<int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl")))
__m256i small_code_bytes(const uint8_t*p){
    static constexpr auto mask=[](){std::array<uint8_t,64>v{};for(int i=0;i<32;i++){v[2*i]=i*Bits/8;v[2*i+1]=i*Bits/8+1;}return v;}();
    static constexpr auto shifts=[](){std::array<uint16_t,32>v{};for(int i=0;i<32;i++)v[i]=(i*Bits)%8;return v;}();
    __m128i packed;
    if constexpr(Bits==2)packed=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    else if constexpr(Bits==3){uint32_t tail;std::memcpy(&tail,p+8,4);packed=_mm_insert_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)),tail,2);}
    else packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    auto words=_mm512_shuffle_epi8(_mm512_broadcast_i32x4(packed),_mm512_loadu_si512(mask.data()));
    auto codes=_mm512_and_si512(_mm512_srlv_epi16(words,_mm512_loadu_si512(shifts.data())),_mm512_set1_epi16((1<<Bits)-1));
    return _mm512_cvtepi16_epi8(codes);
}
template<int bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,ssse3,fma")))
float wxa8_row(const uint8_t* w,const uint8_t* s,const int8_t* x,const float* xs,const int* xsum,int columns) {
    const __m128i native_table=_mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0);
    const int zero=bits==0?12:((1<<(bits-1))-1);
    __m256 accumulated=_mm256_setzero_ps();
    for(int first=0;first<columns;first+=32) {
        __m256i weights;float scale;
        if constexpr(bits==0) {
            const uint8_t code=s[first/32];if(code==255)return NAN;
            scale=e8m0(code)*.5f;
            const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+first/2));
            const __m128i mask=_mm_set1_epi8(15);
            const __m128i low=_mm_shuffle_epi8(native_table,_mm_and_si128(packed,mask));
            const __m128i high=_mm_shuffle_epi8(native_table,_mm_and_si128(_mm_srli_epi16(packed,4),mask));
            weights=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(low,high)),_mm_unpackhi_epi8(low,high),1);
        } else {
            if constexpr(bits==2||bits==3||bits==4)weights=small_code_bytes<bits>(w+first*bits/8);
            else if constexpr(bits==8)weights=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+first));
            else{__m512i low,high;unpack32_simd<bits>(w+first*bits/8,low,high);weights=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm512_cvtepi32_epi8(low)),_mm512_cvtepi32_epi8(high),1);}
            if(_mm256_cmpeq_epi8_mask(weights,_mm256_set1_epi8((1<<bits)-1)))return NAN;
            scale=scale_value(w,columns*bits/8,first/32);
            if(!std::isfinite(scale)||scale<0)return NAN;
        }
        const __m256i activation=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+first));
        const __m256i dot=_mm256_dpbusd_epi32(_mm256_setzero_si256(),weights,activation);
        const __m256i bias=_mm256_mullo_epi32(_mm256_set1_epi32(zero),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xsum+first/4)));
        const __m256i exact=_mm256_sub_epi32(dot,bias);
        accumulated=_mm256_fmadd_ps(_mm256_cvtepi32_ps(exact),_mm256_set1_ps(scale*xs[first/32]),accumulated);
    }
    __m128 sum=_mm_add_ps(_mm256_castps256_ps128(accumulated),_mm256_extractf128_ps(accumulated,1));
    sum=_mm_hadd_ps(sum,sum);sum=_mm_hadd_ps(sum,sum);return _mm_cvtss_f32(sum);
}
RowFunction row_function(int bits) {
    switch(bits) {
        case 0:return wxa8_row<0>;case 2:return wxa8_row<2>;case 3:return wxa8_row<3>;
        case 4:return wxa8_row<4>;case 6:return wxa8_row<6>;case 8:return wxa8_row<8>;
        default:return nullptr;
    }
}
}

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <array>

namespace {
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
int quantize32_vector(const float*x,int columns,int8_t*out,float*scales,int*sums){
    for(int first=0;first<columns;first+=32){
        const __m512 a=_mm512_loadu_ps(x+first),b=_mm512_loadu_ps(x+first+16);
        const __m512i mask=_mm512_set1_epi32(0x7fffffff);
        const __m512 aa=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a),mask)),bb=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(b),mask));
        if(_mm512_cmp_ps_mask(aa,_mm512_set1_ps(3.402823466e38f),_CMP_LE_OQ)!=0xffff||_mm512_cmp_ps_mask(bb,_mm512_set1_ps(3.402823466e38f),_CMP_LE_OQ)!=0xffff)return 4;
        const float maximum=_mm512_reduce_max_ps(_mm512_max_ps(aa,bb)),scale=maximum/127.f;
        if(!std::isfinite(scale)||(maximum>0&&scale==0))return 4;
        const __m512 divisor=_mm512_set1_ps(scale>0?scale:1.f);
        auto round=[&](__m512 value) __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"))) {return _mm512_cvttps_epi32(_mm512_min_ps(_mm512_set1_ps(127.f),_mm512_max_ps(_mm512_set1_ps(-127.f),_mm512_roundscale_ps(_mm512_div_ps(value,divisor),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC))));};
        __m128i lo=_mm512_cvtepi32_epi8(round(a)),hi=_mm512_cvtepi32_epi8(round(b));
        __m256i packed=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);_mm256_storeu_si256(reinterpret_cast<__m256i*>(out+first),packed);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums+first/4),_mm256_dpbusd_epi32(_mm256_setzero_si256(),_mm256_set1_epi8(1),packed));scales[first/32]=scale;
    }return 0;
}
constexpr size_t PairBytes143=6635520,PairAlign143=size_t(2)<<20;
bool paired_mode143(){const char*v=std::getenv("DS1017_CPU_PAIR32_LAYOUT_V143");return v&&!std::strcmp(v,"139");}
bool paired_shape143(int rows,int columns){return(rows==2304&&columns==5120)||(rows==5120&&columns==2304);}
struct PairImage143 {const uint8_t*data=nullptr;size_t bytes=0;int rows=0,columns=0,bits=0;};
struct Projection {
    const uint8_t *weights,*scales;
    const float *x;
    float *output;
    int rows,columns,first,bits,stride;
    RowFunction calculate;
    size_t activation_owner;
    std::vector<int8_t> quantized;
    std::vector<float> activation_scales;
    std::vector<int> activation_sums;
    std::shared_ptr<PairImage143>cached;
    std::vector<int>same;
    int numa_node=-1;
    Ds41A8ViewV118 activation_view{};
};

class Wxa8BatchPool {
    std::mutex command_mutex,state_mutex;
    std::condition_variable ready_cv,job_cv,done_cv;
    std::vector<std::thread> workers;
    std::vector<Projection> projections;
    std::atomic<bool>stopping{false};
    struct alignas(64) Completion{std::atomic<uint64_t>epoch{0};};
    std::array<Completion,256>completions{};
    int ready=0,pending=0,total_rows=0;
    std::array<int,2> local_rows{},local_workers{};
    std::vector<int> local_ranks;
    std::unordered_map<const uint8_t*,int> cache_nodes;
    std::atomic<uint64_t>generation{0};
    std::atomic<int> error{0};
    std::unordered_map<const uint8_t*,std::shared_ptr<PairImage143>>packed_cache;size_t cache_bytes=0,cache_budget=512*1024*1024;
    ParallelCallback post_function=nullptr;void*post_context=nullptr;int post_blocks=0;
    uint64_t* done=nullptr;
    int* actual=nullptr;
    uint64_t owner_generation143=0,format_epoch143=0;
    const int home416; // -1: literal143 global pool; 0/1: explicit component-only domain.
public:
    std::vector<int> cpus;
    int borrow(const void*w,const Ds143PairDescriptor&d,int node){
        std::lock_guard<std::mutex>lock(command_mutex);
        if((home416>=0&&node!=home416)||node<0||node>1||!w||w!=d.data||d.layout_tag!=139||d.bits||!paired_shape143(d.rows,d.columns)||d.bytes!=PairBytes143||uintptr_t(d.data)%PairAlign143||packed_cache.count(static_cast<const uint8_t*>(w))||d.bytes>cache_budget-cache_bytes)return 1;
        if(!d.owner_generation||!d.bank_format_epoch||(owner_generation143&&owner_generation143!=d.owner_generation)||(format_epoch143&&format_epoch143!=d.bank_format_epoch))return 1;
        auto tile=std::make_shared<PairImage143>();tile->data=static_cast<const uint8_t*>(d.data);tile->bytes=d.bytes;tile->rows=d.rows;tile->columns=d.columns;
        packed_cache.emplace(static_cast<const uint8_t*>(w),std::move(tile));cache_bytes+=d.bytes;cache_nodes[static_cast<const uint8_t*>(w)]=node;owner_generation143=d.owner_generation;format_epoch143=d.bank_format_epoch;return 0;
    }
    static void dispose416(Wxa8BatchPool*p){
        if(!p)return;
        if(p->home416<0){delete p;return;}
        const size_t bytes=(sizeof(Wxa8BatchPool)+4095)&~size_t(4095);
        p->~Wxa8BatchPool();if(munmap(p,bytes))std::terminate();
    }
    int info416(uint64_t*out,int n){
        if(!out||n!=8)return 1;
        std::unique_lock<std::mutex>lock(command_mutex,std::try_to_lock);if(!lock.owns_lock())return 2;
        const uint64_t v[8]={418,uint64_t(home416+1),cpus.size(),uint64_t(local_workers[0]),uint64_t(local_workers[1]),generation.load(std::memory_order_acquire),packed_cache.size(),cache_bytes};
        std::copy(v,v+8,out);return 0;
    }
    int prime(const void*,const void*,int,int,int){return 16;} // NO untagged/raw producer fallback.
    int info143(uint64_t*out,int words){
        if(!out||words!=12)return 1;std::unique_lock<std::mutex>lock(command_mutex,std::try_to_lock);if(!lock.owns_lock())return 2;
        out[0]=143;out[1]=139;out[2]=32;out[3]=576;out[4]=288;out[5]=1;out[6]=cpus.size();out[7]=packed_cache.size();out[8]=cache_bytes;out[9]=owner_generation143;out[10]=format_epoch143;out[11]=0;return 0;
    }
    Wxa8BatchPool(const int* ids,int count,int home=-1):home416(home),cpus(ids,ids+count){for(int cpu:cpus){if(cpu<0||cpu>=256)throw std::invalid_argument("physical CPU only");local_ranks.push_back(local_workers[cpu/128]++);}}
    int initialize() {
        if(!paired_mode143())return 16;
        const char*verified=std::getenv("DS41_CPU_WHOLE_LOCAL_VERIFIED");
        if(!verified||std::strcmp(verified,"1"))return 13;
        if(home416<0){if(!local_workers[0]||!local_workers[1])return 13;}
        else if(home416>1||!local_workers[home416]||local_workers[1-home416])return 13;
        if(const char*value=std::getenv("DS41_CPU_BLOCK_CACHE_GIB")){char*end=nullptr;long n=std::strtol(value,&end,10);if(!*value||*end||n<1||n>320)return 1;cache_budget=size_t(n)<<30;}
        if(!__builtin_cpu_supports("avx512vbmi")||!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512bw")||!__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("avx512vnni"))return 3;
        try {for(size_t i=0;i<cpus.size();i++)workers.emplace_back([this,i]{run(i);});}
        catch(...){shutdown();return 9;}
        std::unique_lock<std::mutex> lock(state_mutex);
        ready_cv.wait(lock,[this]{return ready==int(cpus.size());});
        return error.load();
    }
    void run(size_t worker) {
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpus[worker],&mask);
        if(sched_setaffinity(0,sizeof(mask),&mask)||sched_getcpu()!=cpus[worker])error=6;
        uint64_t observed=0;
        {std::lock_guard<std::mutex> lock(state_mutex);ready++;ready_cv.notify_one();}
        auto idle_since=std::chrono::steady_clock::now();unsigned polls=0;
        for(;;){
            uint64_t next=generation.load(std::memory_order_acquire);
            while(next==observed){
                if(stopping.load(std::memory_order_acquire))return;
                if(++polls%256==0 && std::chrono::steady_clock::now()-idle_since>std::chrono::milliseconds(10)){
                    std::unique_lock<std::mutex> lock(state_mutex);
                    job_cv.wait(lock,[this,observed]{return stopping.load(std::memory_order_acquire)||generation.load(std::memory_order_acquire)!=observed;});
                }else __builtin_ia32_pause();
                next=generation.load(std::memory_order_acquire);
            }
            if(stopping.load(std::memory_order_acquire))return;
            observed=next;
            const int node=cpus[worker]/128,rank=local_ranks[worker],count_local=local_workers[node];
            const int first=32*int(uint64_t(local_rows[node]/32)*rank/count_local);
            const int last=32*int(uint64_t(local_rows[node]/32)*(rank+1)/count_local);
            uint64_t count=0;
            if(post_function){
                const int begin=int(uint64_t(post_blocks)*worker/cpus.size());
                const int end=int(uint64_t(post_blocks)*(worker+1)/cpus.size());
                if(begin<end){int rc=post_function(post_context,begin,end);if(rc)error=rc;}
            }else if(!error.load())for(const auto& p:projections){
                if(p.numa_node!=node)continue;
                const int lo=std::max(first,p.first)-p.first;
                const int hi=std::min(last,p.first+p.rows)-p.first;
                if(p.same.empty()||lo>=hi)continue;
                Ds41A8ViewV118 view[6];float*out[6];int n=0;
                for(int index:p.same){const auto&q=projections[index];const auto&a=projections[q.activation_owner];view[n]=a.activation_view;out[n]=q.output;n++;}
                for(int row=lo;row<hi;row+=32){
                    const auto*blocks=p.cached->data+size_t(row/32)*(p.columns/32)*576;
#define DS143_CASE(N) case N:ds143_pair_leaf::ds1017_pair32_kernel_v115<N>(blocks,p.columns,view,out,row,true,true);break
                    switch(n){DS143_CASE(1);DS143_CASE(2);DS143_CASE(3);DS143_CASE(4);DS143_CASE(5);DS143_CASE(6);default:error=1;}
#undef DS143_CASE
                    for(int t=0;t<n;t++)for(int j=0;j<32;j++)if(!std::isfinite(out[t][row+j]))error=5;
                    count+=32*n;
                }
            }
            if(done)done[worker]=count;const int actual_cpu=sched_getcpu();if(actual)actual[worker]=actual_cpu;
            if(actual_cpu!=cpus[worker])error=6;
            completions[worker].epoch.store(observed,std::memory_order_release);
            idle_since=std::chrono::steady_clock::now();polls=0;
        }
    }
    int parallel(ParallelCallback callback,void*context,int blocks){
        std::lock_guard<std::mutex> command(command_mutex);
        if(!callback||!context||blocks<1||blocks>1048576)return 1;
        if(error.load())return error.load();
        std::unique_lock<std::mutex> lock(state_mutex);if(stopping)return 10;
        post_function=callback;post_context=context;post_blocks=blocks;
        // post ABI supplies no GEMV progress buffers; prior caller may have released them.
        done=nullptr;actual=nullptr;
        const uint64_t current=generation.fetch_add(1,std::memory_order_release)+1;
        job_cv.notify_all();lock.unlock();
        for(size_t worker=0;worker<cpus.size();worker++){unsigned spins=0;while(completions[worker].epoch.load(std::memory_order_acquire)!=current){if(++spins%4096==0)std::this_thread::yield();else __builtin_ia32_pause();}}
        post_function=nullptr;post_context=nullptr;post_blocks=0;return error.load();
    }
    int multiply(const uintptr_t* w,const uintptr_t* s,const uintptr_t* x,const uintptr_t* out,
                 const int* rows,const int* cols,const int* bits,int count,uint64_t* completed,int* actual_cpus,const Ds41A8ViewV118* views=nullptr){
        std::lock_guard<std::mutex> command(command_mutex);
        if(!w||!s||!x||!out||!rows||!cols||!bits||!completed||!actual_cpus||count<1||count>96)return 1;
        if(error.load())return error.load();
        if(std::fegetround()!=FE_TONEAREST)return 4;
        std::vector<Projection> next;
        int total=0;std::array<int,2> totals{};
        uint64_t packed_bytes=0;
        for(int i=0;i<count;i++){
            RowFunction calculate=row_function(bits[i]);
            if(!calculate||bits[i]!=0||!w[i]||!s[i]||!x[i]||!out[i]||!paired_shape143(rows[i],cols[i])||s[i]!=w[i]+512)return 1;
            const int stride=bits[i]==0?cols[i]/2:cols[i]*bits[i]/8+cols[i]/32*2;
            const uint64_t bytes=uint64_t(rows[i])*stride;
            if(bytes>64*1024*1024)return 2;
            packed_bytes+=bytes+(bits[i]==0?uint64_t(rows[i])*cols[i]/32:0);
            if(packed_bytes>size_t(4)*1024*1024*1024||total+rows[i]>524288)return 2;
            Projection p{reinterpret_cast<const uint8_t*>(w[i]),reinterpret_cast<const uint8_t*>(s[i]),
                reinterpret_cast<const float*>(x[i]),reinterpret_cast<float*>(out[i]),rows[i],cols[i],total,
                bits[i],stride,calculate,next.size(),{},{},{}};
            auto existing=std::find_if(next.begin(),next.end(),[&](const Projection&q){return q.x==p.x&&q.columns==p.columns;});
            if(views&&(!views[i].codes||!views[i].scales||!views[i].sums||views[i].columns!=p.columns))return 1;
            if(existing!=next.end()){
                p.activation_owner=existing->activation_owner;
                if(views){const auto&v=next[p.activation_owner].activation_view;
                    if(v.codes!=views[i].codes||v.scales!=views[i].scales||v.sums!=views[i].sums||v.columns!=views[i].columns)return 1;}
            }else if(views){
                // Task owns these already rounded codes/scales/sums. No copy,
                // resize, packing or coordinator quantization is performed.
                p.activation_view=views[i];
            }else{
                p.quantized.resize(p.columns);p.activation_scales.resize(p.columns/32);p.activation_sums.resize(p.columns/4);
                if(quantize32_vector(p.x,p.columns,p.quantized.data(),p.activation_scales.data(),p.activation_sums.data()))return 4;
                p.activation_view={p.quantized.data(),p.activation_scales.data(),p.activation_sums.data(),p.columns};
            }
            auto cached=packed_cache.find(p.weights);if(cached!=packed_cache.end()){if(cached->second->rows!=p.rows||cached->second->columns!=p.columns||cached->second->bits!=p.bits)return 1;p.cached=cached->second;}
            auto located=cache_nodes.find(p.weights);if(!p.cached||located==cache_nodes.end())return 14;p.numa_node=located->second;
            int group=-1;
            if(p.cached)for(int j=0;j<int(next.size());j++)if(!next[j].same.empty()&&next[j].cached==p.cached){if(next[j].same.size()>=6)return 1;group=j;break;}
            if(group<0){p.same.push_back(i);p.first=totals[p.numa_node];totals[p.numa_node]+=p.rows;total+=p.rows;}else{next[group].same.push_back(i);p.first=-p.rows;}
            next.push_back(std::move(p));
        }
        // No alias fallback: modelTask outputs are disjoint. Refuse all
        // overlapping output intervals, including shifted and grouped aliases.
        for(size_t i=0;i<next.size();i++){
            const uintptr_t lo=uintptr_t(next[i].output),hi=lo+size_t(next[i].rows)*sizeof(float);if(hi<lo)return 1;
            for(size_t j=0;j<i;j++){const uintptr_t a=uintptr_t(next[j].output),b=a+size_t(next[j].rows)*sizeof(float);if(b<a||(lo<b&&a<hi))return 1;}
        }
        std::unique_lock<std::mutex> lock(state_mutex);
        if(stopping)return 10;
        projections=std::move(next);total_rows=total;local_rows=totals;done=completed;actual=actual_cpus;
        const uint64_t current=generation.fetch_add(1,std::memory_order_release)+1;job_cv.notify_all();lock.unlock();
        for(size_t worker=0;worker<cpus.size();worker++){unsigned spins=0;while(completions[worker].epoch.load(std::memory_order_acquire)!=current){if(++spins%4096==0)std::this_thread::yield();else __builtin_ia32_pause();}}
        return error.load();
    }
    void shutdown(){
        std::lock_guard<std::mutex> command(command_mutex);
        {std::lock_guard<std::mutex> lock(state_mutex);stopping=true;job_cv.notify_all();}
        for(auto& w:workers)if(w.joinable())w.join();workers.clear();
    }
    ~Wxa8BatchPool(){shutdown();std::fprintf(stderr,"CPU_BLOCK_CACHE projections=%zu bytes=%zu budget=%zu\n",packed_cache.size(),cache_bytes,cache_budget);}
};
}

extern "C" void* ds41_wxa8_batch_create(const int* cpus,int count,int* status){
    if(!status)return nullptr;
    *status=1;
    if(!cpus||count<1||count>256)return nullptr;
    for(int i=0;i<count;i++){
        if(cpus[i]<0||cpus[i]>=CPU_SETSIZE)return nullptr;
        for(int j=0;j<i;j++)if(cpus[j]==cpus[i])return nullptr;
    }
    Wxa8BatchPool* pool=nullptr;
    try{pool=new Wxa8BatchPool(cpus,count);*status=pool->initialize();
        if(*status){delete pool;return nullptr;}return pool;
    }catch(...){if(pool)delete pool;*status=9;return nullptr;}
}
extern "C" int ds41_wxa8_batch_gemv(void* handle,const uintptr_t* w,const uintptr_t* s,
    const uintptr_t* x,const uintptr_t* output,const int* rows,const int* columns,const int* bits,int count,uint64_t* done,int* actual){
    if(!handle)return 8;
    try{return static_cast<Wxa8BatchPool*>(handle)->multiply(w,s,x,output,rows,columns,bits,count,done,actual);}
    catch(...){return 9;}
}
extern "C" void ds41_wxa8_batch_destroy(void* handle){Wxa8BatchPool::dispose416(static_cast<Wxa8BatchPool*>(handle));}

extern "C" int ds41_wxa8_prime_v2(void*pool,const void*w,const void*s,int rows,int columns,int bits){if(!pool)return 1;try{return static_cast<Wxa8BatchPool*>(pool)->prime(w,s,rows,columns,bits);}catch(...){return 9;}}

namespace {
std::mutex borrowed_mutex;std::unordered_map<std::string,Ds143PairDescriptor>borrowed_tiles;
std::unordered_map<const void*,std::string>borrowed_keys143;
uint64_t registry_owner143=0,registry_epoch143=0;
thread_local std::string packed_error;
bool descriptor_name143(const char*name,const Ds143PairDescriptor&d){
    if(!name||!strnlen(name,160)||strnlen(name,160)>=160)return false;
    int l=-1,e=-1,end=-1;char role[4]{};
    if(std::sscanf(name,"layers.%d.ffn.experts.%d.%3[^.].weight%n",&l,&e,role,&end)!=3||end<0||name[end])return false;
    const int r=!std::strcmp(role,"w1")?0:!std::strcmp(role,"w3")?1:!std::strcmp(role,"w2")?2:-1;
    return l==d.layer&&e==d.expert&&r==d.role&&l>=0&&l<40&&e>=0&&e<384&&r>=0&&((r==2&&d.rows==5120&&d.columns==2304)||(r!=2&&d.rows==2304&&d.columns==5120));
}
}
extern "C" const char*ds41_wxa8_cache_error(){return packed_error.c_str();}
extern "C" void*ds41_wxa8_cache_make(const void*,const void*,int,int,int,int,void**,size_t*){packed_error="143 forbids untagged/raw cache_make; explicit paired139 producer required";return nullptr;}
extern "C" void ds41_wxa8_cache_free(void*){} // This privateDSO never creates image handles.
extern "C" int ds41_wxa8_cache_register(const char*,void*,size_t,int,int,int){return 16;}
extern "C" int ds143_pair32_cache_register(const char*name,const Ds143PairDescriptor*d){
    try{
    if(!paired_mode143())return 16;
    if(!d||d->struct_bytes!=sizeof(*d)||d->layout_tag!=139||d->bits||!d->data||uintptr_t(d->data)%PairAlign143||d->bytes!=PairBytes143||!paired_shape143(d->rows,d->columns)||d->reserved||!d->owner_generation||!d->bank_format_epoch||!descriptor_name143(name,*d))return 1;
    std::lock_guard<std::mutex>lock(borrowed_mutex);
    if(borrowed_tiles.count(name)||borrowed_keys143.count(d->data))return 2;
    if((registry_owner143&&registry_owner143!=d->owner_generation)||(registry_epoch143&&registry_epoch143!=d->bank_format_epoch))return 1;
    // Validate only original u16 E8 footers, never decode numerical weights.
    const auto*p=static_cast<const uint8_t*>(d->data);const int groups=d->columns/32;
    for(int pair=0;pair<d->rows/32;pair++)for(int g=0;g<groups;g++)for(int k=0;k<32;k++){
        uint16_t word;std::memcpy(&word,p+(size_t(pair)*groups+g)*576+512+k*2,2);if(word>254)return 1;
    }
    auto inserted=borrowed_tiles.emplace(name,*d);
    try{borrowed_keys143.emplace(d->data,name);}catch(...){borrowed_tiles.erase(inserted.first);throw;}
    registry_owner143=d->owner_generation;registry_epoch143=d->bank_format_epoch;return 0;
    }catch(...){return 9;}
}
extern "C" int ds41_wxa8_prime_v3(void*pool,const char*name,const void*w,const void*s,int rows,int columns,int bits){
    try{
    if(!pool||!name)return 1;
    std::lock_guard<std::mutex>lock(borrowed_mutex);auto found=borrowed_tiles.find(name);if(found==borrowed_tiles.end())return 11;
    const auto&d=found->second;
    if(d.rows!=rows||d.columns!=columns||bits!=0||w!=d.data||s!=static_cast<const uint8_t*>(d.data)+512)return 12;
    return static_cast<Wxa8BatchPool*>(pool)->borrow(w,d,(d.layer+d.expert)%2);
    }catch(...){return 9;}
}
extern "C" int ds143_pair32_pool_info(void*p,uint64_t*out,int words){if(!p)return 8;try{return static_cast<Wxa8BatchPool*>(p)->info143(out,words);}catch(...){return 9;}}

extern "C" int ds41_wxa8_parallel_post_v66(void*pool,ParallelCallback callback,void*context,int blocks){if(!pool)return 8;try{return static_cast<Wxa8BatchPool*>(pool)->parallel(callback,context,blocks);}catch(...){return 9;}}

// The sole A8 numerical definition is the unchanged frozen-v66 helper above.
// Existing hidden_post workers call it after original BF16/SwiGLU and FP8/F32.
extern "C" int ds41_wxa8_quantize_exact_v118(const float*x,int columns,int8_t*codes,float*scales,int32_t*sums){
    if(!x||!codes||!scales||!sums||columns<32||columns>65536||columns%32)return 1;
    if(std::fegetround()!=FE_TONEAREST)return 4;
    return quantize32_vector(x,columns,codes,scales,sums);
}
namespace {
std::mutex prequant_stats_mutex;
uint64_t prequant_calls=0,prequant_entries=0,prequant_columns=0;
}
extern "C" int ds41_wxa8_batch_gemv_prequant_v118(void*handle,const uintptr_t*w,const uintptr_t*s,
    const uintptr_t*x,const uintptr_t*output,const int*rows,const int*columns,const int*bits,int count,
    const Ds41A8ViewV118*views,uint64_t*done,int*actual){
    if(!handle)return 8;
    if(!views)return 1;
    try{
        const int rc=static_cast<Wxa8BatchPool*>(handle)->multiply(w,s,x,output,rows,columns,bits,count,done,actual,views);
        if(!rc){uint64_t cols=0;for(int i=0;i<count;i++)cols+=columns[i];
            std::lock_guard<std::mutex>lock(prequant_stats_mutex);prequant_calls++;prequant_entries+=count;prequant_columns+=cols;}
        return rc;
    }catch(...){return 9;}
}
extern "C" int ds41_wxa8_prequant_info_v118(uint64_t*out,int count,int reset){
    if(!out||count<4||(reset!=0&&reset!=1))return 1;
    std::lock_guard<std::mutex>lock(prequant_stats_mutex);
    out[0]=1;out[1]=prequant_calls;out[2]=prequant_entries;out[3]=prequant_columns;
    if(reset)prequant_calls=prequant_entries=prequant_columns=0;
    return 0;
}

// DSO identity for the post-publication lifetime repair; no execution/speed claim.
extern "C" int ds41_wxa8_post_lifetime_info_v94(uint64_t*out,int words){
    if(!out||words!=2)return 1;
    out[0]=94;out[1]=1;return 0;
}

// Component-only local ownership. Original ds41_* constructors remain global143.
extern "C" void* ds418_home_pool_create(const int*ids,int count,int home,int*status){
    if(!status)return nullptr;*status=1;
    const char*flag=std::getenv("DS1017_CPU_NUMA_BULK_V418");
    if(!flag||std::strcmp(flag,"1")){*status=16;return nullptr;}
    if(!ids||count<1||count>128||home<0||home>1)return nullptr;
    for(int i=0;i<count;i++){if(ids[i]<0||ids[i]>=256||ids[i]/128!=home)return nullptr;for(int j=0;j<i;j++)if(ids[i]==ids[j])return nullptr;}
    Wxa8BatchPool*p=nullptr;
    const size_t bytes=(sizeof(Wxa8BatchPool)+4095)&~size_t(4095);
    void*storage=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(storage==MAP_FAILED){*status=9;return nullptr;}
    unsigned long mask=1ul<<home;
    if(syscall(SYS_mbind,storage,bytes,MPOL_BIND,&mask,sizeof(mask)*CHAR_BIT,MPOL_MF_STRICT)){
        const int saved=errno;std::fprintf(stderr,"CPU418 pool mbind errno=%d (%s) node=%d mask=%lu maxnode=%zu flags=%d bytes=%zu address_mod4096=%zu\n",saved,std::strerror(saved),home,mask,sizeof(mask)*CHAR_BIT,MPOL_MF_STRICT,bytes,uintptr_t(storage)%4096);
        munmap(storage,bytes);*status=13;return nullptr;
    }
    try{p=new(storage) Wxa8BatchPool(ids,count,home);*status=p->initialize();if(*status){Wxa8BatchPool::dispose416(p);return nullptr;}
        std::vector<void*>pages;for(size_t offset=0;offset<bytes;offset+=4096)pages.push_back(static_cast<uint8_t*>(storage)+offset);
        std::vector<int>nodes(pages.size(),-1);if(syscall(SYS_move_pages,0,pages.size(),pages.data(),nullptr,nodes.data(),0)||std::any_of(nodes.begin(),nodes.end(),[home](int node){return node!=home;})){
            *status=13;Wxa8BatchPool::dispose416(p);return nullptr;
        }return p;}
    catch(...){if(p)Wxa8BatchPool::dispose416(p);else munmap(storage,bytes);*status=9;return nullptr;}
}
extern "C" int ds418_home_pool_info(void*p,uint64_t*out,int n){
    if(!p)return 8;try{return static_cast<Wxa8BatchPool*>(p)->info416(out,n);}catch(...){return 9;}
}

extern "C" int ds418_home_pool_cpus(void*handle,int*out,int count){
    if(!handle||!out)return 1;
    auto*p=static_cast<Wxa8BatchPool*>(handle);std::array<uint64_t,8>info{};
    if(p->info416(info.data(),8)||info[1]<1||info[1]>2||count!=int(p->cpus.size()))return 1;
    std::copy(p->cpus.begin(),p->cpus.end(),out);return 0;
}
extern "C" int ds418_paired_descriptor(const void*key,Ds143PairDescriptor*out){
    if(!key||!out)return 1;
    try{std::lock_guard<std::mutex>lock(borrowed_mutex);auto k=borrowed_keys143.find(key);if(k==borrowed_keys143.end())return 11;
        auto d=borrowed_tiles.find(k->second);if(d==borrowed_tiles.end()||d->second.data!=key)return 11;*out=d->second;return 0;
    }catch(...){return 9;}
}
