#include "strata/ds41/native_fp8_experts_compact_v252.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/native_fp8_output16_v244.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <vector>
#include "pair_scope_v531.hpp"
extern "C" {
void*ds41_fp8lt_create();void ds41_fp8lt_destroy(void*);const char*ds41_fp8lt_error();
int ds41_fp8lt_encode(const uint16_t*,int,int,uint8_t*,float*);int ds41_fp8lt_act_encode(const uint16_t*,int,int,int,uint8_t*,float*);
int ds41_fp8lt_packed_encode(const uint8_t*,const uint8_t*,int,int,int,uint8_t*,float*);
int ds41_fp8lt_run(void*,const uint8_t*,const uint8_t*,const float*,const float*,int,int,int,float*);
int ds41_fp8lt_run_bf16(void*,const uint8_t*,const uint8_t*,const float*,const float*,int,int,int,float*,uint16_t*,int);
}
namespace strata::ds41 {
namespace {
constexpr int D=5120,H=2304;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void check(int e){if(e)throw std::runtime_error(ds41_fp8lt_error());}
__global__ void rounded(const float*x,__nv_bfloat16*y,size_t count){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)y[i]=__float2bfloat16_rn(x[i]);}
__global__ void hidden(const __nv_bfloat16*g,const __nv_bfloat16*u,const float*alpha,int rows,int padded,__nv_bfloat16*out){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(padded)*H)return;if(i>=size_t(rows)*H){out[i]=__float2bfloat16_rn(0);return;}
 float a=fminf(__bfloat162float(g[i]),10.f),b=fmaxf(-10.f,fminf(__bfloat162float(u[i]),10.f));out[i]=__float2bfloat16_rn((a/(1.f+expf(-a))*b)*alpha[i/H]);
}
}
struct NativeFp8Experts::Impl {
 int batch;void*lt=nullptr;std::vector<void*>allocations;uint8_t*w[3]{},*q=nullptr;float*ws[3]{},*scale=nullptr,*temporary=nullptr;__nv_bfloat16*x=nullptr,*g=nullptr,*u=nullptr,*h=nullptr;
 explicit Impl(int n):batch(n){}
 template<class T>void allocate(T*&p,size_t count){void*v=nullptr;check(cudaMalloc(&v,count*sizeof(T)));allocations.push_back(v);p=static_cast<T*>(v);}
 ~Impl(){if(lt)ds41_fp8lt_destroy(lt);for(void*p:allocations)cudaFree(p);}
 void multiply(int which,int rows,int k,int n,__nv_bfloat16*out){check(ds41_fp8lt_run_bf16(lt,q,w[which],scale,ws[which],rows,k,n,temporary,reinterpret_cast<uint16_t*>(out),rows));}
};
NativeFp8Experts::NativeFp8Experts(int batch):impl_(new Impl(batch)){
 if(batch<1||batch>4096)throw std::invalid_argument("FP8 experts need a batch up to4096");batch=(batch+15)&~15;auto&p=*impl_;p.batch=batch;p.lt=ds41_fp8lt_create();if(!p.lt)throw std::runtime_error(ds41_fp8lt_error());
 for(int i=0;i<3;i++){p.allocate(p.w[i],size_t(D)*H);p.allocate(p.ws[i],i==2?D:H);}
 p.allocate(p.q,size_t(batch)*D);p.allocate(p.scale,batch);p.allocate(p.temporary,size_t(batch)*D);p.allocate(p.x,size_t(batch)*D);p.allocate(p.g,size_t(batch)*H);p.allocate(p.u,size_t(batch)*H);p.allocate(p.h,size_t(batch)*H);
}
NativeFp8Experts::~NativeFp8Experts()=default;
void NativeFp8Experts::weights(const uint16_t*decoded){auto&p=*impl_;for(int i=0;i<3;i++)check(ds41_fp8lt_encode(decoded+size_t(i)*D*H,i==2?D:H,i==2?H:D,p.w[i],p.ws[i]));}
void NativeFp8Experts::packed_weights(NativeDeviceProjection gate,NativeDeviceProjection up,NativeDeviceProjection down){
 auto&p=*impl_;const NativeDeviceProjection values[3]={gate,up,down};
 for(int i=0;i<3;i++)check(ds41_fp8lt_packed_encode(values[i].weights,values[i].scales,values[i].bits,i==2?D:H,i==2?H:D,p.w[i],p.ws[i]));
}
void NativeFp8Experts::output16_weights_v252(NativeDeviceCompactV252 gate,NativeDeviceCompactV252 up,NativeDeviceCompactV252 down){
 auto&p=*impl_;const NativeDeviceCompactV252 values[3]={gate,up,down};
 std::array<pair531::Source,3>source{};std::array<pair531::Destination,3>dest{};
 for(int i=0;i<3;i++){source[i]={values[i].data,values[i].bytes,values[i].bits};dest[i]={p.w[i],p.ws[i],i==2?D:H,i==2?H:D};}
 if(pair531::encode_if_scoped(this,source,dest))return;
 for(int i=0;i<3;i++)if(ds41_fp8_row_encode_output16_v244(values[i].data,values[i].bytes,values[i].bits,
     i==2?D:H,i==2?H:D,p.w[i],p.ws[i],0))throw std::runtime_error(ds41_fp8_output16_v244_error());
}
void NativeFp8Experts::run(const uint16_t*x,const float*alpha,int rows,uint16_t*out){
 auto&p=*impl_;int padded=(rows+15)&~15;if(rows<1||padded>p.batch)throw std::invalid_argument("FP8 expert rows exceed batch");
 check(ds41_fp8lt_act_encode(x,padded,rows,D,p.q,p.scale));p.multiply(0,padded,D,H,p.g);p.multiply(1,padded,D,H,p.u);
 hidden<<<(size_t(padded)*H+255)/256,256>>>(p.g,p.u,alpha,rows,padded,p.h);check(cudaGetLastError());
 check(ds41_fp8lt_act_encode(reinterpret_cast<uint16_t*>(p.h),padded,rows,H,p.q,p.scale));check(ds41_fp8lt_run_bf16(p.lt,p.q,p.w[2],p.scale,p.ws[2],padded,H,D,p.temporary,out,rows));
}
}
