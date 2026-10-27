#include "strata/ds41/borrowed_rows.hpp"
#include <cmath>
#include <algorithm>
#include <atomic>
#include <map>
#include <array>
#include <immintrin.h>
#include <cstring>
#include <stdexcept>
namespace strata::ds41 {
namespace {
std::atomic<uint64_t>protected_cache_loads_r1{0},protected_cache_hits_r1{0},protected_cache_bytes_r1{0};
[[noreturn]]void bad(const char*why){throw std::invalid_argument(std::string("borrowed rows: ")+why);}
template<class T>T read(const uint8_t*p){T result;std::memcpy(&result,p,sizeof result);return result;}
float bf16(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
uint16_t rounded(float f){
    if(!std::isfinite(f))bad("nonfinite decoded row");
    uint32_t u;std::memcpy(&u,&f,4);auto result=uint16_t((u+0x7fff+((u>>16)&1))>>16);
    if((result&0x7f80)==0x7f80)bad("BF16 overflow");return result;
}
float fp8(uint8_t b){
    int exponent=(b>>3)&15,mantissa=b&7;
    if(exponent==15&&mantissa==7)bad("nonfinite FP8 code");
    float value=exponent?std::ldexp(1.f+float(mantissa)/8.f,exponent-7):std::ldexp(float(mantissa),-9);
    return b&128?-value:value;
}
__attribute__((target("avx512f,avx512bw,avx512vl")))
void store_rounded(__m512 value,uint16_t*out){
 auto u=_mm512_castps_si512(value);
 if(_mm512_cmp_ps_mask(_mm512_castsi512_ps(_mm512_and_si512(u,_mm512_set1_epi32(0x7fffffff))),_mm512_set1_ps(3.402823466e38f),_CMP_LE_OQ)!=0xffff)bad("nonfinite decoded row");
 u=_mm512_add_epi32(u,_mm512_add_epi32(_mm512_set1_epi32(0x7fff),_mm512_and_si512(_mm512_srli_epi32(u,16),_mm512_set1_epi32(1))));
 auto b=_mm512_srli_epi32(u,16);
 if(_mm512_cmpeq_epi32_mask(_mm512_and_si512(b,_mm512_set1_epi32(0x7f80)),_mm512_set1_epi32(0x7f80)))bad("BF16 overflow");
 _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),_mm512_cvtepi32_epi16(b));
}
__attribute__((target("avx512f,avx512bw,avx512vl")))
void decode_native_simd(const uint8_t*p,const uint8_t*s,int n,uint16_t*out){
 static const std::array<float,256>table=[](){std::array<float,256>v{};for(int i=0;i<256;i++)v[i]=(i&127)==127?NAN:fp8(i);return v;}();
 for(int j=0;j<n;j+=32){
  if(s[j/32]==255)bad("nonfinite E8M0 scale");uint32_t bits=s[j/32]?uint32_t(s[j/32])<<23:0x00400000;float scale;std::memcpy(&scale,&bits,4);
  for(int half=0;half<2;half++){
   auto codes=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+j+half*16)));
   store_rounded(_mm512_mul_ps(_mm512_i32gather_ps(codes,table.data(),4),_mm512_set1_ps(scale)),out+j+half*16);
  }
 }
}
__attribute__((target("avx512f,avx512bw,avx512vl")))
void decode_uniform_simd(const uint8_t*p,int n,int bits,uint16_t*out){
 const int zero=(1<<(bits-1))-1,mask=(1<<bits)-1;
 for(int j=0;j<n;j+=32){
  alignas(64)uint8_t packed[64]{};std::memcpy(packed,p+j*bits/8,4*bits);
  float scale=bf16(read<uint16_t>(p+n*bits/8+(j/32)*2));if(!std::isfinite(scale)||scale<0)bad("invalid Uniform scale");
  for(int half=0;half<2;half++){
   auto positions=_mm512_mullo_epi32(_mm512_add_epi32(_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),_mm512_set1_epi32(16*half)),_mm512_set1_epi32(bits));
   auto words=_mm512_i32gather_epi32(_mm512_srli_epi32(positions,3),packed,1);
   auto codes=_mm512_and_si512(_mm512_srlv_epi32(words,_mm512_and_si512(positions,_mm512_set1_epi32(7))),_mm512_set1_epi32(mask));
   if(_mm512_cmpeq_epi32_mask(codes,_mm512_set1_epi32(mask)))bad("reserved Uniform code");
   store_rounded(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(codes,_mm512_set1_epi32(zero))),_mm512_set1_ps(scale)),out+j+half*16);
  }
 }
}
int width(const BorrowedTensor&t){
    if(t.shape.size()!=2||t.shape[1]<1||t.shape[1]>65536)bad("expected bounded row-major matrix");return int(t.shape[1]);
}
void scale_shape(const BorrowedTensor&w,const BorrowedTensor&s){
    int n=width(w);
    if(w.encoding!=TensorEncoding::F8e4m3||n%32||s.encoding!=TensorEncoding::E8m0||s.shape!=std::vector<int64_t>{w.shape[0],n/32})bad("native Engram needs row-wise FP8/E8M0 groups32");
}
void decode(const BorrowedTensor&t,const BorrowedTensor*scale,const uint8_t*p,const uint8_t*s,uint16_t*out){
    int n=width(t);
    if(t.encoding==TensorEncoding::F8e4m3){
        if(!scale)bad("missing native scales");scale_shape(t,*scale);
        decode_native_simd(p,s,n,out);
    }else if(t.encoding==TensorEncoding::Uniform){
        decode_uniform_simd(p,n,t.bits,out);
    }else if(t.encoding==TensorEncoding::Bf16){
        for(int j=0;j<n;j++){uint16_t b=read<uint16_t>(p+j*2);if((b&0x7f80)==0x7f80)bad("nonfinite BF16 row");out[j]=b;}
    }else if(t.encoding==TensorEncoding::F32){for(int j=0;j<n;j++)out[j]=rounded(read<float>(p+j*4));}
    else bad("unsupported row encoding");
}
}
BorrowedRows::BorrowedRows(std::shared_ptr<const BorrowedWeights> weights):weights_(std::move(weights)){
    if(!weights_)bad("missing allocation owner");
}
void BorrowedRows::protect(const std::string&tensor,const std::string&ids,const std::string&w,const std::string&s){
    const auto&t=weights_->at(tensor);const auto&i=weights_->at(ids);const auto&nw=weights_->at(w);const auto&ns=weights_->at(s);
    if(t.encoding!=TensorEncoding::Uniform||!t.protected_native_rows||overlays_.count(tensor))bad("invalid/duplicate protected-row binding");
    if(i.encoding!=TensorEncoding::I64||i.shape.size()!=1||i.shape[0]>2000000)bad("invalid protected row IDs");
    scale_shape(nw,ns);
    if(width(nw)!=width(t)||nw.shape[0]!=i.shape[0])bad("protected native payload shape differs");
    int64_t last=-1;std::vector<uint8_t>scratch;
    for(int64_t j=0;j<i.shape[0];j++){int64_t value=read<int64_t>(weights_->read_range(ids,size_t(j)*8,8,scratch));if(value<=last||value>=t.shape[0])bad("protected IDs must be unique sorted source positions");last=value;}
    overlays_.emplace(tensor,Overlay{ids,w,s});
}
std::vector<uint16_t> BorrowedRows::gather_bf16(const std::string&tensor,const std::vector<int64_t>&ids,size_t maximum_bytes)const{
    const auto&t=weights_->at(tensor);const int n=width(t);
    if(maximum_bytes>64ull<<20||maximum_bytes<2||ids.size()>maximum_bytes/(size_t(n)*2))bad("row gather exceeds bounded output budget");
    std::string scale_name;
    if(t.encoding==TensorEncoding::F8e4m3){
        const auto dot=tensor.rfind('.');if(dot==std::string::npos)bad("native tensor lacks scale prefix");
        scale_name=tensor.substr(0,dot)+".scale";scale_shape(t,weights_->at(scale_name));
    }
    auto overlay=overlays_.find(tensor);
    if(t.protected_native_rows&&overlay==overlays_.end())bad("protected native rows are not bound");
    // Immutable source owner identity controls thread-local metadata lifetime.
    // A complete index is published only after the bounded copy succeeds.
    struct ProtectedIdsCacheR1 {
        std::weak_ptr<const BorrowedWeights> owner;
        std::map<std::string,std::vector<int64_t>> indices;
    };
    static thread_local ProtectedIdsCacheR1 protected_cache;
    const std::vector<int64_t>*protected_ids=nullptr;
    if(overlay!=overlays_.end()){
        auto owner=protected_cache.owner.lock();
        if(owner.get()!=weights_.get()){
            protected_cache.indices.clear();protected_cache.owner=weights_;
        }
        const auto&name=overlay->second.ids;
        auto found=protected_cache.indices.find(name);
        if(found==protected_cache.indices.end()){
            if(protected_cache.indices.size()>=8)protected_cache.indices.clear();
            const auto&source=weights_->at(name);
            if(source.encoding!=TensorEncoding::I64||source.shape.size()!=1||source.shape[0]<0||source.shape[0]>2000000)bad("protected ID cache geometry");
            std::vector<int64_t>values(size_t(source.shape[0]));std::vector<uint8_t>scratch;
            const size_t bytes=values.size()*sizeof(int64_t);
            for(size_t offset=0;offset<bytes;offset+=1024*1024){
                const size_t count=std::min(size_t(1024*1024),bytes-offset);
                const auto*p=weights_->read_range(name,offset,count,scratch);
                std::memcpy(reinterpret_cast<uint8_t*>(values.data())+offset,p,count);
            }
            int64_t last=-1;
            for(int64_t id:values){if(id<=last||id>=t.shape[0])bad("protected ID cache order/range");last=id;}
            found=protected_cache.indices.emplace(name,std::move(values)).first;
            protected_cache_loads_r1.fetch_add(1,std::memory_order_relaxed);
            protected_cache_bytes_r1.fetch_add(bytes,std::memory_order_relaxed);
        }else protected_cache_hits_r1.fetch_add(1,std::memory_order_relaxed);
        protected_ids=&found->second;
    }
    std::vector<uint16_t>result(ids.size()*size_t(n));
    std::vector<uint8_t>weight_scratch,scale_scratch,id_scratch;
    auto decode_row=[&](const std::string&name,const std::string&scales,int64_t row,uint16_t*out){
        const auto&matrix=weights_->at(name);size_t stride=matrix.bytes/size_t(matrix.shape[0]);
        const auto*p=weights_->read_range(name,size_t(row)*stride,stride,weight_scratch);
        const BorrowedTensor*scale=nullptr;const uint8_t*s=nullptr;
        if(!scales.empty()){
            scale=&weights_->at(scales);size_t stride_scale=scale->bytes/size_t(scale->shape[0]);
            s=weights_->read_range(scales,size_t(row)*stride_scale,stride_scale,scale_scratch);
        }
        decode(matrix,scale,p,s,out);
    };
    for(size_t j=0;j<ids.size();j++){
        int64_t row=ids[j];if(row<0||row>=t.shape[0])bad("row ID outside tensor");
        auto*out=result.data()+j*n;
        if(overlay!=overlays_.end()){
            if(!protected_ids)bad("missing protected ID cache");
            const auto at=std::lower_bound(protected_ids->begin(),protected_ids->end(),row);
            const int64_t lo=at-protected_ids->begin();
            if(at!=protected_ids->end()&&*at==row){
                decode_row(overlay->second.weight,overlay->second.scale,lo,out);continue;
            }
        }
        decode_row(tensor,scale_name,row,out);
    }
    return result;
}
}

extern "C" uint64_t ds41_protected_id_cache_info_r1(int field){
 using namespace strata::ds41;
 if(field==0)return protected_cache_loads_r1.load(std::memory_order_relaxed);
 if(field==1)return protected_cache_hits_r1.load(std::memory_order_relaxed);
 if(field==2)return protected_cache_bytes_r1.load(std::memory_order_relaxed);
 return UINT64_MAX;
}
