#pragma once
#include "native_packed_output16_prefetch_v125.hpp"
template<int Bits,int Count>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,fma")))
void packed_output16_many(const uint8_t*blocks,const int8_t*const*x,const float*const*xs,const int*const*xsum,int columns,float*const*out,int row){
 const __m512i table=_mm512_broadcast_i32x4(_mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0));
 __m512 acc[Count];
#pragma GCC unroll 8
 for(int t=0;t<Count;t++)acc[t]=_mm512_setzero_ps();
 for(int g=0;g<columns/32;g++){
  compact_prefetch_v125<Bits>(blocks,g,columns/32);
  const uint8_t*block=blocks+size_t(g)*packed_tile_bytes(Bits);auto scales=_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block+packed_code_bytes(Bits))));__m512 factor;
  if constexpr(Bits==0){auto raw=_mm512_slli_epi32(scales,23);raw=_mm512_mask_mov_epi32(raw,_mm512_cmpeq_epi32_mask(scales,_mm512_setzero_si512()),_mm512_set1_epi32(0x00400000));factor=_mm512_mul_ps(_mm512_castsi512_ps(raw),_mm512_set1_ps(.5f));}
  else factor=_mm512_castsi512_ps(_mm512_slli_epi32(scales,16));
  __m512i sum[Count];
#pragma GCC unroll 8
  for(int t=0;t<Count;t++)sum[t]=_mm512_setzero_si512();
  for(int j=0;j<8;j++){
   auto codes=compact_code_vector<Bits>(block+j*8*(Bits?Bits:4));
   if constexpr(Bits==0)codes=_mm512_shuffle_epi8(table,codes);
   
#pragma GCC unroll 8
   for(int t=0;t<Count;t++){uint32_t activation;std::memcpy(&activation,x[t]+g*32+j*4,4);sum[t]=_mm512_dpbusd_epi32(sum[t],codes,_mm512_set1_epi32(activation));}
  }
  
#pragma GCC unroll 8
  for(int t=0;t<Count;t++){
   int total=0;for(int j=0;j<8;j++)total+=xsum[t][g*8+j];constexpr int bias=Bits==0?12:(1<<(Bits-1))-1;
   sum[t]=_mm512_sub_epi32(sum[t],_mm512_set1_epi32(bias*total));auto scaled=_mm512_mul_ps(factor,_mm512_set1_ps(xs[t][g]));acc[t]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(sum[t]),scaled,acc[t]);
  }
 }
 
#pragma GCC unroll 8
 for(int t=0;t<Count;t++)_mm512_storeu_ps(out[t]+row,acc[t]);
}

template<int Bits>
inline void dispatch_packed_many(const uint8_t*block,const int8_t*const*x,const float*const*xs,const int*const*sums,int columns,float*const*out,int row,int count){
#define CALL(N) case N:packed_output16_many<Bits,N>(block,x,xs,sums,columns,out,row);return
 switch(count){CALL(2);CALL(3);CALL(4);CALL(5);CALL(6);CALL(7);CALL(8);default:throw std::runtime_error("packed multirow count");}
#undef CALL
}
inline void run_packed_many(const PackedOutput16&p,int row,const int8_t*const*x,const float*const*xs,const int*const*sums,float*const*out,int count){
 const uint8_t*block=p.data+size_t(row/16)*(p.columns/32)*packed_tile_bytes(p.bits);
 switch(p.bits){case 0:dispatch_packed_many<0>(block,x,xs,sums,p.columns,out,row,count);break;case 2:dispatch_packed_many<2>(block,x,xs,sums,p.columns,out,row,count);break;case 3:dispatch_packed_many<3>(block,x,xs,sums,p.columns,out,row,count);break;case 4:dispatch_packed_many<4>(block,x,xs,sums,p.columns,out,row,count);break;default:throw std::runtime_error("packed multirow precision");}
}
