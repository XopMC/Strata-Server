#pragma once
// CPU-private output16 cache; checkpoint/GPU row encodings remain untouched.
#include <unordered_map>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <sys/mman.h>
struct PackedOutput16 {
 bool owns_memory=true;uint8_t*data=nullptr;size_t bytes=0;int rows=0,columns=0,bits=0;
 ~PackedOutput16(){if(owns_memory)std::free(data);}
};
inline int packed_code_bytes(int bits){return 64*(bits?bits:4);}
inline int packed_tile_bytes(int bits){return packed_code_bytes(bits)+32;}
inline std::shared_ptr<PackedOutput16> pack_output16(const uint8_t*w,const uint8_t*s,int rows,int columns,int bits,int threads=16){
 auto value=std::make_shared<PackedOutput16>();value->rows=rows;value->columns=columns;value->bits=bits;value->bytes=size_t(rows/16)*(columns/32)*packed_tile_bytes(bits);
 void*memory=nullptr;if(posix_memalign(&memory,2*1024*1024,value->bytes))throw std::bad_alloc();value->data=static_cast<uint8_t*>(memory);
 const uintptr_t page=(reinterpret_cast<uintptr_t>(memory)+4095)&~uintptr_t(4095),end=(reinterpret_cast<uintptr_t>(memory)+value->bytes)&~uintptr_t(4095);if(end>page)madvise(reinterpret_cast<void*>(page),end-page,MADV_HUGEPAGE);
 std::atomic<int>bad{0};const int groups=columns/32,stride=bits==0?columns/2:columns*bits/8+groups*2,code_bytes=packed_code_bytes(bits);
 auto fill=[&](int tile){for(int g=0;g<groups;g++){
  uint8_t*dst=value->data+(size_t(tile)*groups+g)*packed_tile_bytes(bits);
  for(int r=0;r<16;r++){
   const int row=tile*16+r;const uint8_t*src=w+size_t(row)*stride;uint16_t scale;
   if(bits==0){scale=s[size_t(row)*groups+g];if(scale==255)bad=1;}
   else{std::memcpy(&scale,src+columns*bits/8+g*2,2);if((scale&0x8000)||(scale&0x7f80)==0x7f80)bad=1;}
   std::memcpy(dst+code_bytes+r*2,&scale,2);
  }
  for(int j=0;j<8;j++)for(int r=0;r<16;r+=2){
   uint32_t combined=0;
   for(int pair=0;pair<2;pair++){
    const uint8_t*src=w+size_t(tile*16+r+pair)*stride;uint16_t code=0;
    if(bits==0||bits==4)std::memcpy(&code,src+g*16+j*2,2);
    else if(bits==3){std::memcpy(&code,src+g*12+(j*12)/8,2);code=(code>>((j*12)%8))&0xfff;}
    else code=src[g*8+j];
    const int width=bits?bits:4;
    if(bits)for(int k=0;k<4;k++)if(((code>>(k*bits))&((1<<bits)-1))==((1<<bits)-1))bad=1;
    combined|=uint32_t(code)<<(pair*4*width);
   }
   const int width=bits?bits:4;
   // Eight codes (two output rows) occupy exactly width bytes.
   std::memcpy(dst+j*8*width+(r/2)*width,&combined,width);
  }
 }};
 if(threads==1){for(int tile=0;tile<rows/16;tile++)fill(tile);}
 else{
 #pragma omp parallel for num_threads(threads) schedule(static)
 for(int tile=0;tile<rows/16;tile++)fill(tile);
 }
 if(bad.load())throw std::runtime_error("invalid compact CPU cache weight code/scale");return value;
}

// v125 experiment: bounded lookahead only. Cache layout/arithmetic unchanged.
#ifndef DS41_CPU_PREFETCH_GROUPS_V125
#define DS41_CPU_PREFETCH_GROUPS_V125 4
#endif
template<int Bits>
__attribute__((target("avx512f")))
inline void compact_prefetch_v125(const uint8_t*blocks,int group,int groups){
 constexpr int lookahead=DS41_CPU_PREFETCH_GROUPS_V125;
 if constexpr(lookahead>0){
  if(group+lookahead<groups){
   const uint8_t*future=blocks+size_t(group+lookahead)*packed_tile_bytes(Bits);
   for(int offset=0;offset<packed_tile_bytes(Bits);offset+=64)
    _mm_prefetch(reinterpret_cast<const char*>(future+offset),_MM_HINT_T0);
  }
 }
}
template<int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,fma")))
inline __m512i compact_code_vector(const uint8_t*p){
 if constexpr(Bits==0||Bits==4){
  const auto duplicated=_mm512_permutexvar_epi64(_mm512_set_epi64(3,3,2,2,1,1,0,0),_mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))));
  const auto shifts=_mm512_set_epi64(0x3c3834302c282420ll,0x1c1814100c080400ll,0x3c3834302c282420ll,0x1c1814100c080400ll,0x3c3834302c282420ll,0x1c1814100c080400ll,0x3c3834302c282420ll,0x1c1814100c080400ll);
  return _mm512_and_si512(_mm512_multishift_epi64_epi8(shifts,duplicated),_mm512_set1_epi8(15));
 }else{
  static constexpr auto indices=[](){std::array<uint8_t,64>v{};for(int lane=0;lane<8;lane++)for(int b=0;b<Bits;b++)v[lane*8+b]=lane*Bits+b;return v;}();
  constexpr uint64_t mask=Bits==3?0x0707070707070707ull:0x0303030303030303ull;
  constexpr uint64_t shifts=Bits==3?0x15120f0c09060300ull:0x0e0c0a0806040200ull;
  const auto source=_mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
  const auto expanded=_mm512_maskz_permutexvar_epi8(mask,_mm512_loadu_si512(indices.data()),source);
  return _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_set1_epi64(shifts),expanded),_mm512_set1_epi8((1<<Bits)-1));
 }
}
template<int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,fma")))
void packed_output16(const uint8_t*blocks,const int8_t*x,const float*xs,const int*xsum,int columns,float*out){
 const __m512i table=_mm512_broadcast_i32x4(_mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0));
 __m512 acc=_mm512_setzero_ps();
 for(int g=0;g<columns/32;g++){
  compact_prefetch_v125<Bits>(blocks,g,columns/32);
  const uint8_t*block=blocks+size_t(g)*packed_tile_bytes(Bits);auto scales=_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block+packed_code_bytes(Bits))));__m512 factor;
  if constexpr(Bits==0){auto raw=_mm512_slli_epi32(scales,23);raw=_mm512_mask_mov_epi32(raw,_mm512_cmpeq_epi32_mask(scales,_mm512_setzero_si512()),_mm512_set1_epi32(0x00400000));factor=_mm512_mul_ps(_mm512_castsi512_ps(raw),_mm512_set1_ps(.5f));}
  else factor=_mm512_castsi512_ps(_mm512_slli_epi32(scales,16));
  __m512i sum=_mm512_setzero_si512();
  for(int j=0;j<8;j++){
   auto codes=compact_code_vector<Bits>(block+j*8*(Bits?Bits:4));
   if constexpr(Bits==0)codes=_mm512_shuffle_epi8(table,codes);
   uint32_t activation;std::memcpy(&activation,x+g*32+j*4,4);sum=_mm512_dpbusd_epi32(sum,codes,_mm512_set1_epi32(activation));
  }
  int total=0;for(int j=0;j<8;j++)total+=xsum[g*8+j];constexpr int bias=Bits==0?12:(1<<(Bits-1))-1;
  sum=_mm512_sub_epi32(sum,_mm512_set1_epi32(bias*total));factor=_mm512_mul_ps(factor,_mm512_set1_ps(xs[g]));acc=_mm512_fmadd_ps(_mm512_cvtepi32_ps(sum),factor,acc);
 }
 _mm512_storeu_ps(out,acc);
}
inline void run_packed_output16(const PackedOutput16&p,int row,const int8_t*x,const float*xs,const int*sums,float*out){
 const uint8_t*block=p.data+size_t(row/16)*(p.columns/32)*packed_tile_bytes(p.bits);
 switch(p.bits){case 0:packed_output16<0>(block,x,xs,sums,p.columns,out);break;case 2:packed_output16<2>(block,x,xs,sums,p.columns,out);break;case 3:packed_output16<3>(block,x,xs,sums,p.columns,out);break;case 4:packed_output16<4>(block,x,xs,sums,p.columns,out);break;default:throw std::runtime_error("unhandled blocked precision");}
}
