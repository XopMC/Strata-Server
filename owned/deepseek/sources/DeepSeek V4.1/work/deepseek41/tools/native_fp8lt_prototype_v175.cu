#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <stdexcept>
#include <string>
#include <cmath>
#include "strata/ds41/ops.hpp"
// v175 isolated opt-in dispatch prototype, not integrated or executed.
// Original FP8LT path is retained verbatim except ds41_fp8lt_run_bf16 dispatch.
// Link this TU IN PLACE OF native_fp8lt_prototype.cu, plus the separate closed
// native_fp8_scaled_epilogue_v170.cu. Do not compile both FP8LT TUs together.
// Only rows>=128, n==32768, k==1280 use CUTLASS when DS41_FP8_SCALED_V175=1.
// All other shapes retain the original cuBLASLt+rescale implementation.
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <atomic>
extern "C" {
size_t ds41_fp8_scaled_v170_workspace_bytes(int,int,int,int);
int ds41_fp8_scaled_v170_run(const uint8_t*,const uint8_t*,const float*,const float*,
                           int,int,int,uint16_t*,int,void*,size_t,uintptr_t);
const char* ds41_fp8_scaled_v170_error();
}
namespace {
thread_local std::string error;
std::atomic<uint64_t> query_calls_v175{0}, query_rows_v175{0}, query_values_v175{0};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void check(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLASLt status "+std::to_string(e));}
__global__ void encode_rows(const __nv_bfloat16*x,int rows,int width,uint8_t*q,float*scales){
 int row=blockIdx.x;float maximum=1e-4f;__shared__ float shared[256];
 for(int c=threadIdx.x;c<width;c+=blockDim.x)maximum=fmaxf(maximum,fabsf(__bfloat162float(x[row*width+c])));
 shared[threadIdx.x]=maximum;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)shared[threadIdx.x]=fmaxf(shared[threadIdx.x],shared[threadIdx.x+d]);__syncthreads();}
 float raw=shared[0]*(1.f/448.f);uint32_t bits=__float_as_uint(raw);int exponent=int((bits>>23)&255)-127+((bits&0x7fffff)!=0);float scale=ldexpf(1.f,exponent);if(threadIdx.x==0)scales[row]=scale;
 for(int c=threadIdx.x;c<width;c+=blockDim.x){__nv_fp8_e4m3 value(__bfloat162float(x[row*width+c])/scale);q[row*width+c]=value.__x;}
}
__global__ void rescale(float*y,const float*xs,const float*ws,int rows,int n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(rows)*n)y[i]*=xs[i/n]*ws[i%n];}
__global__ void rescale_bf16(const float*y,const float*xs,const float*ws,int rows,int n,uint16_t*out){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<size_t(rows)*n)reinterpret_cast<__nv_bfloat16*>(out)[i]=__float2bfloat16_rn(y[i]*(xs[i/n]*ws[i%n]));
}

// v67: source per32 FP8/BF16 rounding and row FP8 encoding in one CTA.
// Intermediate values stay in registers. Each source group and rounding step
// is unchanged; padded rows are positive zero as in the original memset path.
template<int Capacity>
__global__ void encode_activation_rows(const __nv_bfloat16*x,int rows,int valid_rows,int width,uint8_t*q,float*scales){
 constexpr int Iterations=(Capacity+255)/256;const int row=blockIdx.x;
 float values[Iterations];float maximum=1e-4f;__shared__ float shared[256];
#pragma unroll
 for(int iteration=0;iteration<Iterations;iteration++){
  const int c=threadIdx.x+iteration*256;
  const float original=row<valid_rows&&c<width?__bfloat162float(x[size_t(row)*width+c]):0.f;
  float amax=fabsf(original);
#pragma unroll
  for(int offset=16;offset;offset>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,offset));
  const float raw=fmaxf(amax,1e-4f)*(1.f/448.f);int exponent;
  const float mantissa=frexpf(raw,&exponent);const float group_scale=ldexpf(1.f,mantissa==.5f?exponent-1:exponent);
  const __nv_fp8_e4m3 rounded(fminf(fmaxf(original/group_scale,-448.f),448.f));
  values[iteration]=__bfloat162float(__float2bfloat16_rn(float(rounded)*group_scale));
  if(c<width)maximum=fmaxf(maximum,fabsf(values[iteration]));
 }
 shared[threadIdx.x]=maximum;__syncthreads();
 for(int offset=128;offset;offset>>=1){if(threadIdx.x<offset)shared[threadIdx.x]=fmaxf(shared[threadIdx.x],shared[threadIdx.x+offset]);__syncthreads();}
 const float raw=shared[0]*(1.f/448.f);const uint32_t bits=__float_as_uint(raw);
 const int exponent=int((bits>>23)&255)-127+((bits&0x7fffff)!=0);const float scale=ldexpf(1.f,exponent);
 if(threadIdx.x==0)scales[row]=scale;
#pragma unroll
 for(int iteration=0;iteration<Iterations;iteration++){
  const int c=threadIdx.x+iteration*256;
  if(c<width){const __nv_fp8_e4m3 encoded(values[iteration]/scale);q[size_t(row)*width+c]=encoded.__x;}
 }
}

// v78: preserve decode BF16 rounding, then row FP8 encoding, in registers.
template<int Bits,int Width>
__device__ __forceinline__ float packed_bf16_value_v78(const uint8_t*w,const uint8_t*s,int row,int col){
 float value;
 if constexpr(Bits==0){
  const int code=(w[size_t(row)*(Width/2)+col/2]>>(4*(col%2)))&15,magnitude=code&7;
  const uint32_t raw=magnitude<2?(magnitude?0x3f000000u:0u):(uint32_t((magnitude>>1)+126)<<23)|(uint32_t(magnitude&1)<<22);
  const float decoded=__uint_as_float(raw|((code&8)?0x80000000u:0u));
  const uint8_t scale=s[size_t(row)*(Width/32)+col/32];
  const float factor=scale==255?nanf(""):__uint_as_float(scale?uint32_t(scale)<<23:0x00400000u);
  value=decoded*factor;
 }else{
  constexpr size_t stride=size_t(Width)*Bits/8+size_t(Width/32)*2;
  const uint8_t*base=w+size_t(row)*stride;const int at=col*Bits,byte=at/8;
  const int code=((int(base[byte])|(int(base[byte+1])<<8))>>(at%8))&((1<<Bits)-1);
  const uint8_t*ss=base+size_t(Width)*Bits/8+(col/32)*2;
  const float scale=__uint_as_float(uint32_t(uint16_t(ss[0])|(uint16_t(ss[1])<<8))<<16);
  value=(code==(1<<Bits)-1||!isfinite(scale)||scale<0)?nanf(""):float(code-((1<<(Bits-1))-1))*scale;
 }
 return __bfloat162float(__float2bfloat16_rn(value));
}
template<int Bits,int Width>
__global__ void encode_packed_rows_v78(const uint8_t*w,const uint8_t*s,uint8_t*q,float*scales){
 constexpr int Iterations=(Width+255)/256;const int row=blockIdx.x;
 float values[Iterations],maximum=1e-4f;__shared__ float shared[256];
 #pragma unroll
 for(int it=0;it<Iterations;it++){const int c=threadIdx.x+it*256;values[it]=c<Width?packed_bf16_value_v78<Bits,Width>(w,s,row,c):0.f;if(c<Width)maximum=fmaxf(maximum,fabsf(values[it]));}
 shared[threadIdx.x]=maximum;__syncthreads();
 for(int d=128;d;d>>=1){if(threadIdx.x<d)shared[threadIdx.x]=fmaxf(shared[threadIdx.x],shared[threadIdx.x+d]);__syncthreads();}
 const float raw=shared[0]*(1.f/448.f);const uint32_t bits=__float_as_uint(raw);const int exponent=int((bits>>23)&255)-127+((bits&0x7fffff)!=0);const float scale=ldexpf(1.f,exponent);
 if(threadIdx.x==0)scales[row]=scale;
 #pragma unroll
 for(int it=0;it<Iterations;it++){const int c=threadIdx.x+it*256;if(c<Width){const __nv_fp8_e4m3 value(values[it]/scale);q[size_t(row)*Width+c]=value.__x;}}
}
struct Plan{cublasLtMatmulDesc_t op{};cublasLtMatrixLayout_t a{},b{},c{};cublasLtMatmulAlgo_t algo{};~Plan(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}};
struct Engine{
 cublasLtHandle_t handle{};void*workspace=nullptr;float*one=nullptr;size_t bytes=32*1024*1024;std::map<std::tuple<int,int,int>,Plan*>plans;
 Engine(){try{check(cublasLtCreate(&handle));check(cudaMalloc(&workspace,bytes));check(cudaMalloc(&one,4));float value=1;check(cudaMemcpy(one,&value,4,cudaMemcpyHostToDevice));}catch(...){if(workspace)cudaFree(workspace);if(one)cudaFree(one);if(handle)cublasLtDestroy(handle);throw;}}
 ~Engine(){for(auto&pair:plans)delete pair.second;if(workspace)cudaFree(workspace);if(one)cudaFree(one);if(handle)cublasLtDestroy(handle);}
 Plan&plan(int rows,int k,int n){
  auto key=std::make_tuple(rows,k,n);auto it=plans.find(key);if(it!=plans.end())return *it->second;
  Plan*p=new Plan;try{
   check(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof trans));
   check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&one,sizeof one));check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&one,sizeof one));
   check(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_8F_E4M3,k,n,k));check(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_8F_E4M3,k,rows,k));check(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_32F,n,rows,n));
   cublasLtMatmulPreference_t preference;check(cublasLtMatmulPreferenceCreate(&preference));check(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof bytes));
   cublasLtMatmulHeuristicResult_t result[8];int found=0;auto status=cublasLtMatmulAlgoGetHeuristic(handle,p->op,p->a,p->b,p->c,p->c,preference,8,result,&found);cublasLtMatmulPreferenceDestroy(preference);check(status);
   int selected=-1;for(int i=0;i<found;i++)if(result[i].state==CUBLAS_STATUS_SUCCESS){selected=i;break;}if(selected<0)throw std::runtime_error("No SM89 FP8/F32 tensor-core algorithm for shape");p->algo=result[selected].algo;
   plans.emplace(key,p);return *p;
  }catch(...){delete p;throw;}
 }
};
}
extern "C" {
const char* ds41_fp8lt_error(){return error.c_str();}
int ds41_fp8lt_query_info_v175(uint64_t*out,size_t count,int reset){
 if(!out||count!=6||(reset!=0&&reset!=1)){error="v175 query info needs six values and an explicit reset flag";return 1;}
 const char*mode=std::getenv("DS41_FP8_SCALED_V175");
 out[0]=175;out[1]=query_calls_v175.load(std::memory_order_relaxed);
 out[2]=query_rows_v175.load(std::memory_order_relaxed);out[3]=query_values_v175.load(std::memory_order_relaxed);
 out[4]=mode&&mode[0]=='1'&&mode[1]=='\0';out[5]=128;
 if(reset){query_calls_v175.store(0,std::memory_order_relaxed);query_rows_v175.store(0,std::memory_order_relaxed);query_values_v175.store(0,std::memory_order_relaxed);}
 return 0;
}
void*ds41_fp8lt_create(){try{return new Engine;}catch(const std::exception&e){error=e.what();return nullptr;}}
void ds41_fp8lt_destroy(void*p){delete static_cast<Engine*>(p);}
int ds41_fp8lt_encode(const uint16_t*x,int rows,int k,uint8_t*out,float*scale){try{if(!x||!out||!scale||rows<1||k<32||k%32)throw std::invalid_argument("Invalid row quant input");encode_rows<<<rows,256>>>(reinterpret_cast<const __nv_bfloat16*>(x),rows,k,out,scale);check(cudaGetLastError());return 0;}catch(const std::exception&e){error=e.what();return 1;}}

int ds41_fp8lt_act_encode(const uint16_t*x,int rows,int valid_rows,int k,uint8_t*out,float*scale){try{
 if(!x||!out||!scale||rows<1||valid_rows<1||valid_rows>rows||k<32||k>8192||k%32)throw std::invalid_argument("Invalid fused activation row input");
#define WIDTH(K) case K:encode_activation_rows<K><<<rows,256>>>(reinterpret_cast<const __nv_bfloat16*>(x),rows,valid_rows,k,out,scale);break
 switch(k){WIDTH(128);WIDTH(256);WIDTH(512);WIDTH(768);WIDTH(1024);WIDTH(1280);WIDTH(1536);WIDTH(2048);WIDTH(2304);WIDTH(4096);WIDTH(5120);WIDTH(6144);WIDTH(8192);default:encode_activation_rows<8192><<<rows,256>>>(reinterpret_cast<const __nv_bfloat16*>(x),rows,valid_rows,k,out,scale);break;}
#undef WIDTH
 check(cudaGetLastError());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
int ds41_fp8lt_source_act_quant_v67(uint16_t*x,int rows,int k){try{if(!x||rows<1||k<32||k%32)throw std::invalid_argument("Invalid source activation quant input");strata::ds41::ops::act_quant_inplace(reinterpret_cast<__nv_bfloat16*>(x),rows*k);return 0;}catch(const std::exception&e){error=e.what();return 1;}}

int ds41_fp8lt_packed_encode(const uint8_t*w,const uint8_t*s,int bits,int rows,int k,uint8_t*q,float*scales){try{
 if(!w||!q||!scales||rows<1||rows>32768||(bits==0&&!s)||(k!=5120&&k!=2304))throw std::invalid_argument("Bounded native packed FP8 matrix required");
 #define CALL(B,K) encode_packed_rows_v78<B,K><<<rows,256>>>(w,s,q,scales)
 #define CASE(B) case B:if(k==5120){CALL(B,5120);}else{CALL(B,2304);}break
 switch(bits){CASE(0);CASE(2);CASE(3);CASE(4);CASE(6);CASE(8);default:throw std::invalid_argument("Packed FP8 precision unsupported");}
 #undef CASE
 #undef CALL
 check(cudaGetLastError());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
int ds41_fp8lt_run(void*opaque,const uint8_t*x,const uint8_t*w,const float*xs,const float*ws,int rows,int k,int n,float*out){try{
 if(!opaque||!x||!w||!xs||!ws||!out||rows<16||rows%16||k<32||k%32||n<16||n%16)throw std::invalid_argument("Invalid aligned FP8 matmul");auto&e=*static_cast<Engine*>(opaque);auto&p=e.plan(rows,k,n);float one=1,zero=0;
 check(cublasLtMatmul(e.handle,p.op,&one,w,p.a,x,p.b,&zero,out,p.c,out,p.c,&p.algo,e.workspace,e.bytes,0));rescale<<<(size_t(rows)*n+255)/256,256>>>(out,xs,ws,rows,n);check(cudaGetLastError());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
int ds41_fp8lt_run_raw(void*opaque,const uint8_t*x,const uint8_t*w,const float*xs,const float*ws,int rows,int k,int n,float*out){try{
 if(!opaque||!x||!w||!xs||!ws||!out||rows<16||rows%16||k<32||k%32||n<16||n%16)throw std::invalid_argument("Invalid aligned FP8 matmul");auto&e=*static_cast<Engine*>(opaque);auto&p=e.plan(rows,k,n);float one=1,zero=0;
 check(cublasLtMatmul(e.handle,p.op,&one,w,p.a,x,p.b,&zero,out,p.c,out,p.c,&p.algo,e.workspace,e.bytes,0));return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
int ds41_fp8lt_run_bf16(void*opaque,const uint8_t*x,const uint8_t*w,const float*xs,const float*ws,int rows,int k,int n,float*scratch,uint16_t*out,int valid_rows){
 try{
  if(!out||valid_rows<1||valid_rows>rows)throw std::invalid_argument("Invalid fused BF16 row count");
  if(rows>=128&&n==32768&&k==1280){
   const char*mode=std::getenv("DS41_FP8_SCALED_V175");
   if(mode&&mode[0]=='1'&&mode[1]=='\0'){
    if(!opaque||!x||!w||!xs||!ws||!scratch||rows%16)throw std::invalid_argument("Invalid v175 selected FP8 matmul");
    auto&e=*static_cast<Engine*>(opaque);
    const size_t required=ds41_fp8_scaled_v170_workspace_bytes(rows,k,n,valid_rows);
    if(required==SIZE_MAX)throw std::runtime_error(ds41_fp8_scaled_v170_error());
    if(required>e.bytes)throw std::runtime_error("v175 CUTLASS workspace exceeds existing FP8LT workspace");
    // Current regular/no-split-K kernel requires zero workspace. Future edits
    // must preserve the check and must be warmed before any graph capture.
    if(ds41_fp8_scaled_v170_run(x,w,xs,ws,rows,k,n,out,valid_rows,e.workspace,e.bytes,0))
     throw std::runtime_error(ds41_fp8_scaled_v170_error());
    query_calls_v175.fetch_add(1,std::memory_order_relaxed);
    query_rows_v175.fetch_add(uint64_t(valid_rows),std::memory_order_relaxed);
    query_values_v175.fetch_add(uint64_t(valid_rows)*uint64_t(n),std::memory_order_relaxed);
    return 0;
   }
  }
  if(ds41_fp8lt_run_raw(opaque,x,w,xs,ws,rows,k,n,scratch))return 1;
  rescale_bf16<<<(size_t(valid_rows)*n+255)/256,256>>>(scratch,xs,ws,valid_rows,n,out);check(cudaGetLastError());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}

}
