// V2 SIMD unpack: independent candidate component, identical v1 F32 dot order.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <sched.h>
#include <thread>
#include <vector>

namespace {
bool geometry(int rows,int columns,int bits) {
    return rows>0 && rows<=65536 && columns>=32 && columns<=65536 && columns%32==0 &&
        (bits==2||bits==3||bits==4||bits==6||bits==8) &&
        uint64_t(rows)*(uint64_t(columns)*bits/8+columns/32*2)<=64*1024*1024;
}
float scale_value(const uint8_t* row,int code_bytes,int group) {
    const uint32_t word=uint32_t(row[code_bytes+2*group])|(uint32_t(row[code_bytes+2*group+1])<<8);
    uint32_t bits=word<<16;float value;std::memcpy(&value,&bits,sizeof(value));return value;
}
bool codes32(const uint8_t* row,int first,int bits,int* values) {
    const int mask=(1<<bits)-1,zero=(1<<(bits-1))-1;
    for(int lane=0;lane<32;lane++) {
        const int bit=(first+lane)*bits,byte=bit/8;
        // The final code can read its adjacent scale byte; two-byte reads never
        // extend outside the interleaved row, including the width32 boundary.
        const int word=int(row[byte])|(int(row[byte+1])<<8);
        const int code=(word>>(bit%8))&mask;
        if(code==mask)return false;
        values[lane]=code-zero;
    }
    return true;
}
// Integer SIMD unpacking retains the v1 F32 FMA order. No BF16 or int8
// activation approximation is introduced. Short groups never over-read.
template<int bits>
__attribute__((target("avx512f,avx512bw,ssse3,fma")))
void unpack32_simd(const uint8_t* group,__m512i&low,__m512i&high) {
    if(bits==8) {
        low=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(group)));
        high=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(group+16)));
    } else if(bits==2||bits==4) {
        const __m128i packed=bits==2?_mm_loadl_epi64(reinterpret_cast<const __m128i*>(group))
            :_mm_loadu_si128(reinterpret_cast<const __m128i*>(group));
        alignas(16) uint8_t first[16],last[16];alignas(64) int shifts[16];
        for(int lane=0;lane<16;lane++) {
            first[lane]=uint8_t(lane*bits/8);last[lane]=uint8_t((lane+16)*bits/8);
            shifts[lane]=(lane*bits)%8;
        }
        low=_mm512_cvtepu8_epi32(_mm_shuffle_epi8(packed,_mm_load_si128(reinterpret_cast<const __m128i*>(first))));
        high=_mm512_cvtepu8_epi32(_mm_shuffle_epi8(packed,_mm_load_si128(reinterpret_cast<const __m128i*>(last))));
        const __m512i shift=_mm512_load_si512(shifts);
        low=_mm512_srlv_epi32(low,shift);high=_mm512_srlv_epi32(high,shift);
    } else {
        // Width32 group3/6 contains12/24bytes. Zero padding permits the
        // four-byte gathers at the last lane without reading a later group.
        alignas(64)uint8_t padded[36]={};std::memcpy(padded,group,4*bits);
        alignas(64)int offsets0[16],offsets1[16],shifts0[16],shifts1[16];
        for(int lane=0;lane<16;lane++) {
            offsets0[lane]=lane*bits/8;offsets1[lane]=(lane+16)*bits/8;
            shifts0[lane]=(lane*bits)%8;shifts1[lane]=((lane+16)*bits)%8;
        }
        low=_mm512_srlv_epi32(_mm512_i32gather_epi32(_mm512_load_si512(offsets0),padded,1),_mm512_load_si512(shifts0));
        high=_mm512_srlv_epi32(_mm512_i32gather_epi32(_mm512_load_si512(offsets1),padded,1),_mm512_load_si512(shifts1));
    }
    const __m512i mask=_mm512_set1_epi32((1<<bits)-1);
    low=_mm512_and_si512(low,mask);high=_mm512_and_si512(high,mask);
}
template<int bits>
__attribute__((target("avx512f,avx512bw,ssse3,fma")))
float vector_row_fixed(const uint8_t* row,const float* x,int columns) {
    __m512 sum0=_mm512_setzero_ps(),sum1=_mm512_setzero_ps();
    const int code_bytes=columns*bits/8;
    const __m512i forbidden=_mm512_set1_epi32((1<<bits)-1);
    const __m512i zero=_mm512_set1_epi32((1<<(bits-1))-1);
    for(int first=0;first<columns;first+=32) {
        const float scale=scale_value(row,code_bytes,first/32);
        if(!std::isfinite(scale)||scale<0)return NAN;
        __m512i low,high;unpack32_simd<bits>(row+first*bits/8,low,high);
        if(_mm512_cmpeq_epi32_mask(low,forbidden)||_mm512_cmpeq_epi32_mask(high,forbidden))return NAN;
        const __m512 scalar=_mm512_set1_ps(scale);
        const __m512 w0=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(low,zero)),scalar);
        const __m512 w1=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(high,zero)),scalar);
        sum0=_mm512_fmadd_ps(w0,_mm512_loadu_ps(x+first),sum0);
        sum1=_mm512_fmadd_ps(w1,_mm512_loadu_ps(x+first+16),sum1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum0,sum1));
}
float vector_row(const uint8_t* row,const float* x,int columns,int bits) {
    switch(bits) {
        case 2:return vector_row_fixed<2>(row,x,columns);
        case 3:return vector_row_fixed<3>(row,x,columns);
        case 4:return vector_row_fixed<4>(row,x,columns);
        case 6:return vector_row_fixed<6>(row,x,columns);
        case 8:return vector_row_fixed<8>(row,x,columns);
        default:return NAN;
    }
}
float scalar_row(const uint8_t* row,const float* x,int columns,int bits) {
    float result=0;int codes[32];const int code_bytes=columns*bits/8;
    for(int first=0;first<columns;first+=32) {
        const float scale=scale_value(row,code_bytes,first/32);
        if(!std::isfinite(scale)||scale<0||!codes32(row,first,bits,codes))return NAN;
        for(int lane=0;lane<32;lane++) {
            const float weight=float(codes[lane])*scale;
            result=std::fma(weight,x[first+lane],result);
        }
    }
    return result;
}
struct Job {
    const uint8_t* raw=nullptr;const float* x=nullptr;float* output=nullptr;
    int rows=0,columns=0,bits=0,vectorized=0;
    uint64_t* done=nullptr;int* actual=nullptr;
};
class Pool {
public:
    std::vector<int> cpus;std::vector<std::thread> threads;
    std::mutex command_mutex,state_mutex;std::condition_variable ready_cv,job_cv,done_cv;
    std::atomic<int> error{0};bool stopping=false;int ready=0,pending=0;uint64_t generation=0;Job job;
    Pool(const int* values,int count):cpus(values,values+count){}
    int initialize() {
        try {for(size_t i=0;i<cpus.size();i++)threads.emplace_back([this,i]{worker(i);});}
        catch(...) {close();return 8;}
        std::unique_lock<std::mutex> lock(state_mutex);
        ready_cv.wait(lock,[this]{return ready==int(cpus.size());});return error.load();
    }
    void worker(size_t index) {
        cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpus[index],&set);
        if(sched_setaffinity(0,sizeof(set),&set)||sched_getcpu()!=cpus[index])error=6;
        uint64_t observed=0;
        {std::lock_guard<std::mutex> lock(state_mutex);ready++;ready_cv.notify_one();}
        for(;;) {
            Job current;
            {std::unique_lock<std::mutex> lock(state_mutex);
             job_cv.wait(lock,[this,observed]{return stopping||generation!=observed;});
             if(stopping)return;
             current=job;observed=generation;}
            const int first=int(uint64_t(current.rows)*index/cpus.size());
            const int last=int(uint64_t(current.rows)*(index+1)/cpus.size());
            const uint64_t row_bytes=uint64_t(current.columns)*current.bits/8+current.columns/32*2;
            uint64_t done=0;
            if(!error.load())for(int row=first;row<last;row++) {
                const uint8_t* data=current.raw+uint64_t(row)*row_bytes;
                const float value=current.vectorized?vector_row(data,current.x,current.columns,current.bits)
                    :scalar_row(data,current.x,current.columns,current.bits);
                if(!std::isfinite(value)){error=5;break;}
                current.output[row]=value;done++;
            }
            current.done[index]=done;current.actual[index]=sched_getcpu();
            if(current.actual[index]!=cpus[index])error=6;
            {std::lock_guard<std::mutex> lock(state_mutex);pending--;if(!pending)done_cv.notify_one();}
        }
    }
    int multiply(const uint8_t* raw,const float* x,float* output,int rows,int columns,int bits,int vectorized,
                 uint64_t* done,int* actual) {
        std::lock_guard<std::mutex> command(command_mutex);
        if(!raw||!x||!output||!done||!actual||!geometry(rows,columns,bits)||(vectorized!=0&&vectorized!=1))return 1;
        if(vectorized&&(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512bw")||!__builtin_cpu_supports("ssse3")||!__builtin_cpu_supports("fma")))return 3;
        for(int i=0;i<columns;i++)if(!std::isfinite(x[i]))return 4;
        std::unique_lock<std::mutex> lock(state_mutex);
        if(stopping)return 7;
        if(error.load())return error.load();
        job={raw,x,output,rows,columns,bits,vectorized,done,actual};pending=int(cpus.size());generation++;
        job_cv.notify_all();done_cv.wait(lock,[this]{return pending==0;});return error.load();
    }
    void close() {
        {std::lock_guard<std::mutex> lock(state_mutex);stopping=true;job_cv.notify_all();}
        for(auto&thread:threads)if(thread.joinable())thread.join();threads.clear();
    }
    ~Pool(){close();}
};
}
extern "C" {
void* ds41_own_pool_create(const int* cpus,int count,int* status) {
    if(!status)return nullptr;*status=1;
    if(!cpus||count<1||count>256)return nullptr;
    cpu_set_t allowed;if(sched_getaffinity(0,sizeof(allowed),&allowed))return nullptr;
    for(int i=0;i<count;i++) {
        if(cpus[i]<0||cpus[i]>=CPU_SETSIZE||!CPU_ISSET(cpus[i],&allowed))return nullptr;
        for(int j=0;j<i;j++)if(cpus[i]==cpus[j])return nullptr;
    }
    Pool* pool=nullptr;
    try {pool=new Pool(cpus,count);*status=pool->initialize();}
    catch(...) {*status=8;delete pool;return nullptr;}
    if(*status){delete pool;return nullptr;}return pool;
}
int ds41_own_pool_gemv(void* pool,const uint8_t* raw,const float* x,float* output,int rows,int columns,
                     int bits,int vectorized,uint64_t* done,int* actual) {
    if(!pool)return 1;
    try{return static_cast<Pool*>(pool)->multiply(raw,x,output,rows,columns,bits,vectorized,done,actual);}
    catch(...){return 8;}
}
void ds41_own_pool_destroy(void* pool){delete static_cast<Pool*>(pool);}
}
