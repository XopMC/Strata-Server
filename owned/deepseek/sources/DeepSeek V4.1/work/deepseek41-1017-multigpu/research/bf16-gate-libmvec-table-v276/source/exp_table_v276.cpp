#include "exp_table_v276.h"
#include <immintrin.h>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <algorithm>
extern "C" __m512 _ZGVeN16v_expf(__m512);
namespace {
constexpr uint32_t Count=65536,Valid=49313,LimitBits=0x41200000,Control=0x1f80;
bool legal(uint16_t c){return(c&0x7f80)!=0x7f80&&((c&0x8000)||c<=0x4120);}
bool controls(){return(_mm_getcsr()&~uint32_t(0x3f))==Control&&std::fegetround()==FE_TONEAREST;}
float value(uint16_t c){uint32_t u=uint32_t(c)<<16;float f;std::memcpy(&f,&u,4);return f;}
uint64_t hash(const uint32_t*v,const uint8_t*d){uint64_t h=1469598103934665603ull;for(size_t i=0;i<Count;i++){for(int j=0;j<4;j++){h^=uint8_t(v[i]>>(8*j));h*=1099511628211ull;}h^=d[i];h*=1099511628211ull;}return h;}
bool header(const Ds276ExpTable*t){return t&&t->struct_bytes==sizeof(*t)&&t->version==276&&t->entries==Count&&t->valid_entries==Valid&&t->limit_f32_bits==LimitBits&&t->mxcsr_controls==Control&&!t->reserved0&&!t->reserved1&&t->values&&t->domain&&t->resolved_libmvec==reinterpret_cast<uintptr_t>(&_ZGVeN16v_expf);}
bool span(const void*p,size_t n,uintptr_t&a,uintptr_t&z){a=reinterpret_cast<uintptr_t>(p);if(!a||n>UINTPTR_MAX-a)return false;z=a+n;return true;}
bool overlap(const void*a,size_t n,const void*b,size_t m){uintptr_t x,y,z,w;if(!span(a,n,x,y)||!span(b,m,z,w))return true;return x<w&&z<y;}
}
extern "C" int ds276_exp_table_build(Ds276ExpTable*out,uint32_t*v,size_t entries,uint8_t*d,size_t dsize,uint64_t*stats,int words){
 if(!out||!v||!d||!stats||entries!=Count||dsize!=Count||words!=16||reinterpret_cast<uintptr_t>(v)%4)return 1;
 if(overlap(v,Count*4,d,Count)||overlap(out,sizeof(*out),v,Count*4)||overlap(out,sizeof(*out),d,Count)||overlap(stats,128,v,Count*4)||overlap(stats,128,d,Count)||overlap(out,sizeof(*out),stats,128))return 1;
 if(!controls())return 4;
 std::array<uint64_t,16>s{};s[0]=276;s[4]=_mm_getcsr();s[6]=s[4]&~uint64_t(0x3f);
 uint32_t valid=0;
 for(uint32_t c=0;c<Count;c++){
  d[c]=legal(uint16_t(c));v[c]=0x7fc00276u;if(!d[c])continue;
  // Literal selected call: +zero minus rounded/clamped BF16 gate, not scalar expf.
  const __m512 g=_mm512_set1_ps(value(uint16_t(c)));
  const __m512 z=_ZGVeN16v_expf(_mm512_sub_ps(_mm512_setzero_ps(),g));
  alignas(64)uint32_t lanes[16];_mm512_store_si512(lanes,_mm512_castps_si512(z));
  for(int i=1;i<16;i++)if(lanes[i]!=lanes[0])return 5;
  if((lanes[0]&0x7fffffff)>0x7f800000||lanes[0]&0x80000000)return 5;
  v[c]=lanes[0];valid++;s[2]++;s[3]+=16;if(lanes[0]==0x7f800000)s[8]++;
 }
 s[1]=valid;s[5]=_mm_getcsr();s[7]=s[5]&~uint64_t(0x3f);
 if(valid!=Valid||s[6]!=Control||s[7]!=Control||!controls())return 4;
 Ds276ExpTable result{sizeof(Ds276ExpTable),276,Count,Valid,LimitBits,Control,0,0,v,d,hash(v,d),reinterpret_cast<uintptr_t>(&_ZGVeN16v_expf)};
 *out=result;std::copy(s.begin(),s.end(),stats);return 0;
}
extern "C" int ds276_exp_table_validate(const Ds276ExpTable*t){
 if(!header(t))return 1;if(!controls())return 4;
 uint32_t n=0;for(uint32_t c=0;c<Count;c++){const bool x=legal(uint16_t(c));if(t->domain[c]!=uint8_t(x))return 2;if(x){n++;if((t->values[c]&0x7fffffff)>0x7f800000||t->values[c]&0x80000000)return 2;}else if(t->values[c]!=0x7fc00276u)return 2;}
 return n==Valid&&hash(t->values,t->domain)==t->content_fnv64?0:2;
}
extern "C" int ds276_exp_lookup16(const Ds276ExpTable*t,const uint16_t*g,uint32_t*out){
 if(!header(t)||!g||!out)return 1;if(!controls())return 4;
 if(overlap(out,64,t->values,Count*4)||overlap(out,64,t->domain,Count)||overlap(out,64,g,32)||overlap(out,64,t,sizeof(*t)))return 1;
 for(int i=0;i<16;i++)if(!legal(g[i])||t->domain[g[i]]!=1)return 2;
 for(int i=0;i<16;i++)out[i]=t->values[g[i]];return 0;
}
