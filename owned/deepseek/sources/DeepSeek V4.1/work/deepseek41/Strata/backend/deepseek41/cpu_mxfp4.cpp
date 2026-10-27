// Native MXFP4/E8M0 expert matrix-vector component, not a complete CPU backend.
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <sched.h>
#include <omp.h>

namespace {
constexpr float fp4[16] = {0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
float e8m0(uint8_t code) {
    if (!code) return std::ldexp(1.f,-127);
    uint32_t bits = uint32_t(code)<<23;
    float value;
    std::memcpy(&value,&bits,sizeof(value));
    return value;
}

float scalar_row(const uint8_t* w,const uint8_t* s,const float* x,int columns) {
    double sum = 0;
    for (int j=0;j<columns;j+=32) {
        double scale = e8m0(s[j/32]);
        for (int k=0;k<32;k++) {
            uint8_t packed = w[(j+k)/2];
            int code = k%2 ? packed>>4 : packed&15;
            sum += double(x[j+k])*fp4[code]*scale;
        }
    }
    return float(sum);
}

__attribute__((target("avx512f,avx512bw,fma")))
float vector_row(const uint8_t* w,const uint8_t* s,const float* transposed,int columns) {
    const __m512 table = _mm512_loadu_ps(fp4);
    const __m512i mask = _mm512_set1_epi32(15);
    __m512 even = _mm512_setzero_ps(),odd = even;
    for (int j=0;j<columns;j+=32) {
        __m512i packed = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+j/2)));
        __m512 low = _mm512_permutexvar_ps(_mm512_and_si512(packed,mask),table);
        __m512 high = _mm512_permutexvar_ps(_mm512_srli_epi32(packed,4),table);
        __m512 scale = _mm512_set1_ps(e8m0(s[j/32]));
        even = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(transposed+j),scale),low,even);
        odd = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(transposed+j+16),scale),high,odd);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(even,odd));
}
}

int gemv_impl(const uint8_t* weights,const uint8_t* scales,
    const float* x,float* output,int rows,int columns,int threads,int vectorized,
    const int* cpus,uint64_t* worker_rows,int* actual_cpus) {
    if (!weights || !scales || !x || !output || rows<1 || columns<32 || columns%32
        || rows>65536 || columns>65536 || threads<1 || threads>256
        || (vectorized!=0 && vectorized!=1)) return 1;
    if (uint64_t(rows)*columns/2 > 64*1024*1024) return 2;
    if (vectorized && (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw"))) return 3;
    for (int j=0;j<columns;j++) if (!std::isfinite(x[j])) return 4;
    for (int j=0;j<rows*(columns/32);j++) if (scales[j]==255) return 5;
    std::vector<float> transposed(columns);
    if (vectorized) for (int j=0;j<columns;j+=32) for (int k=0;k<16;k++) {
        transposed[j+k] = x[j+2*k];
        transposed[j+16+k] = x[j+2*k+1];
    }
    std::atomic<int> error{0};
    #pragma omp parallel num_threads(threads)
    {
        const int worker=omp_get_thread_num();
        cpu_set_t previous,selected;
        bool restore=false;
        uint64_t done=0;
        if (cpus) {
            CPU_ZERO(&selected);
            if (cpus[worker]<0 || cpus[worker]>=CPU_SETSIZE || sched_getaffinity(0,sizeof(previous),&previous)) error=6;
            else {
                CPU_SET(cpus[worker],&selected);
                if (sched_setaffinity(0,sizeof(selected),&selected)) error=6;
                else restore=true;
            }
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int row=0;row<rows;row++) {
            if (error.load()) continue;
            const uint8_t* w = weights+uint64_t(row)*(columns/2);
            const uint8_t* s = scales+uint64_t(row)*(columns/32);
            output[row] = vectorized ? vector_row(w,s,transposed.data(),columns) : scalar_row(w,s,x,columns);
            done++;
        }
        if (worker_rows) worker_rows[worker]=done;
        if (actual_cpus) actual_cpus[worker]=sched_getcpu();
        if (restore && sched_setaffinity(0,sizeof(previous),&previous)) error=7;
    }
    return error.load();
}

extern "C" int ds41_mxfp4_gemv(const uint8_t* weights,const uint8_t* scales,
    const float* x,float* output,int rows,int columns,int threads,int vectorized) {
    return gemv_impl(weights,scales,x,output,rows,columns,threads,vectorized,nullptr,nullptr,nullptr);
}

extern "C" int ds41_mxfp4_gemv_pinned(const uint8_t* weights,const uint8_t* scales,
    const float* x,float* output,int rows,int columns,int threads,int vectorized,
    const int* cpus,uint64_t* worker_rows,int* actual_cpus) {
    if (!cpus || !worker_rows || !actual_cpus) return 8;
    return gemv_impl(weights,scales,x,output,rows,columns,threads,vectorized,cpus,worker_rows,actual_cpus);
}
