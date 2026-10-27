#pragma once
// Literal selected115 factor/kernel; namespace only, ABI view alias is layout-identical.
namespace ds143_pair_leaf {
using Ds1017Pair32ViewV115=Ds41A8ViewV118;
__attribute__((target("avx512f,avx512bw")))
inline __m512 factor(const uint8_t*p){
 auto scales=_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
 auto raw=_mm512_slli_epi32(scales,23);
 raw=_mm512_mask_mov_epi32(raw,_mm512_cmpeq_epi32_mask(scales,_mm512_setzero_si512()),_mm512_set1_epi32(0x00400000));
 return _mm512_mul_ps(_mm512_castsi512_ps(raw),_mm512_set1_ps(.5f));
}
template<int Count>
__attribute__((noinline,target("avx512f,avx512bw,avx512vnni,fma")))
void ds1017_pair32_kernel_v115(const uint8_t*blocks,int columns,
 const Ds1017Pair32ViewV115*view,float*const*out,int row,bool first,bool second){
 const __m512i table=_mm512_broadcast_i32x4(_mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0));
 const __m512i mask=_mm512_set1_epi8(15);
 __m512 acc0[Count],acc1[Count];
 #pragma clang loop unroll(full)
 for(int t=0;t<Count;t++){acc0[t]=_mm512_setzero_ps();acc1[t]=_mm512_setzero_ps();}
 for(int g=0;g<columns/32;g++){
  const uint8_t*b=blocks+size_t(g)*576;
  // Preserve original prefetch distance16/K32; address only in this pair image.
  if(g+16<columns/32)for(int at=0;at<576;at+=64)
   _mm_prefetch(reinterpret_cast<const char*>(blocks+size_t(g+16)*576+at),_MM_HINT_T0);
  if constexpr(Count<=3){
   __m512i sum0[Count],sum1[Count];
   #pragma clang loop unroll(full)
   for(int t=0;t<Count;t++){sum0[t]=_mm512_setzero_si512();sum1[t]=_mm512_setzero_si512();}
   for(int j=0;j<8;j++){
    const auto packed=_mm512_load_si512(b+j*64);
    const auto c0=_mm512_shuffle_epi8(table,_mm512_and_si512(packed,mask));
    const auto c1=_mm512_shuffle_epi8(table,_mm512_and_si512(_mm512_srli_epi16(packed,4),mask));
    #pragma clang loop unroll(full)
    for(int t=0;t<Count;t++){
     uint32_t a;std::memcpy(&a,view[t].codes+g*32+j*4,4);const auto av=_mm512_set1_epi32(a);
     if(first)sum0[t]=_mm512_dpbusd_epi32(sum0[t],c0,av);
     if(second)sum1[t]=_mm512_dpbusd_epi32(sum1[t],c1,av);
    }
   }
   const auto f0=factor(b+512),f1=factor(b+544);
   #pragma clang loop unroll(full)
   for(int t=0;t<Count;t++){
    int total=0;for(int j=0;j<8;j++)total+=view[t].sums[g*8+j];
    const auto bias=_mm512_set1_epi32(12*total);const auto scale=_mm512_set1_ps(view[t].scales[g]);
    if(first)acc0[t]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(sum0[t],bias)),_mm512_mul_ps(f0,scale),acc0[t]);
    if(second)acc1[t]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(sum1[t],bias)),_mm512_mul_ps(f1,scale),acc1[t]);
   }
  }else{
   // Count4..8: serialize the two integer bodies inside the SAME K32 group,
   // retaining two ordered float accumulator banks but only one INT32 bank.
   // This trades a repeated L1 load of the512B codes for lower vector liveness;
   // no assurance against spills until Root inspects actual generated code.
   #pragma clang loop unroll(full)
   for(int half=0;half<2;half++){
    if((half==0&&!first)||(half==1&&!second))continue;
    __m512i sum[Count];
    #pragma clang loop unroll(full)
    for(int t=0;t<Count;t++)sum[t]=_mm512_setzero_si512();
    for(int j=0;j<8;j++){
     const auto packed=_mm512_load_si512(b+j*64);
     const auto n=half?_mm512_and_si512(_mm512_srli_epi16(packed,4),mask):_mm512_and_si512(packed,mask);
     const auto codes=_mm512_shuffle_epi8(table,n);
     #pragma clang loop unroll(full)
     for(int t=0;t<Count;t++){
      uint32_t a;std::memcpy(&a,view[t].codes+g*32+j*4,4);
      sum[t]=_mm512_dpbusd_epi32(sum[t],codes,_mm512_set1_epi32(a));
     }
    }
    const auto f=factor(b+512+half*32);
    #pragma clang loop unroll(full)
    for(int t=0;t<Count;t++){
     int total=0;for(int j=0;j<8;j++)total+=view[t].sums[g*8+j];
     const auto scaled=_mm512_mul_ps(f,_mm512_set1_ps(view[t].scales[g]));
     const auto exact=_mm512_sub_epi32(sum[t],_mm512_set1_epi32(12*total));
     if(half==0)acc0[t]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(exact),scaled,acc0[t]);
     else acc1[t]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(exact),scaled,acc1[t]);
    }
   }
  }
 }
 #pragma clang loop unroll(full)
 for(int t=0;t<Count;t++){
  if(first)_mm512_storeu_ps(out[t]+row,acc0[t]);
  if(second)_mm512_storeu_ps(out[t]+row+16,acc1[t]);
 }
}

}
