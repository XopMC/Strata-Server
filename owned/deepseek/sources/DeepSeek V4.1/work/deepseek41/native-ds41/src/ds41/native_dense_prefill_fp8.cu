#include "strata/ds41/native_dense_prefill.hpp"
#include "strata/ds41/prefill_ops.hpp"
#include "strata/ds41/ops.hpp"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <map>
#include <cstdlib>
#include <cstring>
extern "C" {void*ds41_fp8lt_create();void ds41_fp8lt_destroy(void*);const char*ds41_fp8lt_error();int ds41_fp8lt_encode(const uint16_t*,int,int,uint8_t*,float*);int ds41_fp8lt_act_encode(const uint16_t*,int,int,int,uint8_t*,float*);int ds41_fp8lt_run(void*,const uint8_t*,const uint8_t*,const float*,const float*,int,int,int,float*);int ds41_fp8lt_run_bf16(void*,const uint8_t*,const uint8_t*,const float*,const float*,int,int,int,float*,uint16_t*,int);}
#include <stdexcept>
namespace strata::ds41 {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void check(int e){if(e)throw std::runtime_error(ds41_fp8lt_error());}
__global__ void round_output(const float*x,__nv_bfloat16*y,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=__float2bfloat16_rn(x[i]);}
__global__ void decode_dense(const uint8_t*w,const uint8_t*s,int64_t n,int64_t k,__nv_bfloat16*out){
 const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*k)return;
 __nv_fp8_e4m3 value;value.__x=w[i];const int64_t row=i/k,col=i%k;
 out[i]=__float2bfloat16_rn(float(value)*ldexpf(1.f,int(s[(row/32)*(k/32)+col/32])-127));
}
}
struct NativeDensePrefill::Impl {
 struct Entry{const uint8_t*scale;int64_t k,n;__nv_bfloat16*decoded;uint8_t*q=nullptr;float*row_scales=nullptr;};
 uint8_t*arena=nullptr;__nv_bfloat16*x=nullptr;float*tmp=nullptr;size_t budget=0,used=0;
 std::map<const uint8_t*,Entry>entries;int max_rows=256;bool fp8=false;void*lt=nullptr;uint8_t*qx=nullptr;float*xs=nullptr;__nv_bfloat16*scratch=nullptr;
 ~Impl(){if(lt)ds41_fp8lt_destroy(lt);if(qx)cudaFree(qx);if(xs)cudaFree(xs);if(arena)cudaFree(arena);if(x)cudaFree(x);if(tmp)cudaFree(tmp);}
};
NativeDensePrefill::NativeDensePrefill(size_t bytes,int max_rows):impl_(new Impl){
 if(max_rows<1||max_rows>4096)throw std::invalid_argument("dense batch must fit4096");impl_->max_rows=(max_rows+15)&~15;
 if(bytes<64*1024*1024||bytes>512*1024*1024)throw std::invalid_argument("bounded dense cache required");
 impl_->budget=bytes;check(cudaMalloc(&impl_->arena,bytes));
 const char*mode=std::getenv("DS41_NATIVE_DENSE_FP8");impl_->fp8=mode&&std::strcmp(mode,"1")==0;
 if(impl_->fp8){if(bytes<256*1024*1024)throw std::invalid_argument("dense FP8 cache too small");impl_->budget-=128*1024*1024;impl_->scratch=reinterpret_cast<__nv_bfloat16*>(impl_->arena+impl_->budget);impl_->lt=ds41_fp8lt_create();if(!impl_->lt)throw std::runtime_error(ds41_fp8lt_error());check(cudaMalloc(&impl_->qx,size_t(impl_->max_rows)*8192));check(cudaMalloc(&impl_->xs,size_t(impl_->max_rows)*4));}check(cudaMalloc(&impl_->x,size_t(impl_->max_rows)*8192*2));check(cudaMalloc(&impl_->tmp,size_t(impl_->max_rows)*32768*4));
}
NativeDensePrefill::~NativeDensePrefill()=default;
void NativeDensePrefill::next_layer(){impl_->entries.clear();impl_->used=0;}
bool NativeDensePrefill::run(const uint16_t*x,int rows,int64_t k,const uint8_t*w,const uint8_t*s,int64_t n,uint16_t*y){
 if(rows<1||rows>impl_->max_rows||k<32||k>8192||k%32||n<1||n>32768)return false;
 auto&p=*impl_;auto entry=p.entries.find(w);
 if(entry==p.entries.end()){
  if(p.fp8&&(n%16||size_t(n)*k*2>size_t(128)*1024*1024))return false;
  size_t bytes=p.fp8?size_t(n)*k+size_t(n)*4:size_t(n)*k*2,at=(p.used+255)&~size_t(255);if(at>p.budget||bytes>p.budget-at)return false;
  auto*decoded=p.fp8?p.scratch:reinterpret_cast<__nv_bfloat16*>(p.arena+at);decode_dense<<<(size_t(n)*k+255)/256,256>>>(w,s,n,k,decoded);check(cudaGetLastError());
  Impl::Entry value{s,k,n,decoded};
  if(p.fp8){value.q=p.arena+at;value.row_scales=reinterpret_cast<float*>(value.q+size_t(n)*k);check(ds41_fp8lt_encode(reinterpret_cast<uint16_t*>(decoded),n,k,value.q,value.row_scales));value.decoded=nullptr;}
  entry=p.entries.emplace(w,value).first;p.used=at+bytes;
 }else if(entry->second.scale!=s||entry->second.k!=k||entry->second.n!=n)throw std::runtime_error("dense cache pointer alias mismatch");
 if(p.fp8){int padded=(rows+15)&~15;check(ds41_fp8lt_act_encode(x,padded,rows,k,p.qx,p.xs));check(ds41_fp8lt_run_bf16(p.lt,p.qx,entry->second.q,p.xs,entry->second.row_scales,padded,k,n,p.tmp,y,rows));return true;}
 check(cudaMemcpyAsync(p.x,x,size_t(rows)*k*2,cudaMemcpyDeviceToDevice,0));ops::act_quant_inplace(p.x,rows*k);
 prefill::bf16_gemm(p.x,entry->second.decoded,rows,k,n,reinterpret_cast<__nv_bfloat16*>(y),nullptr,p.tmp);return true;
}
}
