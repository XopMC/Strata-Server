#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>

// CPU-only borrowed views, valid until the existing batch completion barrier.
// Every columns/32 group contains32 signed codes,1 FP32 scale and8 int32 sums
// (one sum per four codes), exactly as frozen v66 quantize32_vector produces.
struct Ds41A8ViewV118 {
    const int8_t* codes;
    const float* scales;
    const int32_t* sums;
    int columns;
};
static_assert(std::is_same_v<int,int32_t>);
static_assert(sizeof(Ds41A8ViewV118)==32);
static_assert(offsetof(Ds41A8ViewV118,columns)==24);
using Ds41ExactQuantV118=int(*)(const float*,int,int8_t*,float*,int32_t*);
using Ds41PrequantBatchV118=int(*)(void*,const uintptr_t*,const uintptr_t*,
    const uintptr_t*,const uintptr_t*,const int*,const int*,const int*,int,
    const Ds41A8ViewV118*,uint64_t*,int*);

extern "C" int ds41_wxa8_quantize_exact_v118(const float*,int,int8_t*,float*,int32_t*);
extern "C" int ds41_wxa8_batch_gemv_prequant_v118(void*,const uintptr_t*,const uintptr_t*,
    const uintptr_t*,const uintptr_t*,const int*,const int*,const int*,int,
    const Ds41A8ViewV118*,uint64_t*,int*);
// Successful explicit-view calls only; schema1/calls/projection entries/columns.
// count>=4, reset0/1; reset returns the old snapshot before clearing. Query idle.
extern "C" int ds41_wxa8_prequant_info_v118(uint64_t*,int,int);
