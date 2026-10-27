// Private source-only CPU-semantics NV4 component. No production dispatcher.
// expf table/polynomial derived from Arm optimized-routines, commit
// f5205d212ac67727689176b7fcfaef7864d898a5, math/{expf,exp2f_data}.c.
// Copyright (c) 2017-2025, Arm Limited.
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
// Modified for explicit installed __expf_fma instruction order; see LICENSES.md.
// CPU NV4 and software-half operation order adapted from the MIT-licensed
// Strata/ggml sources pinned in SOURCE_PINS.json; original notices retained.
#include "cpu_semantics_v86.hpp"
#include <cuda_fp16.h>
#include <climits>
namespace swift_cpu_semantics_v86 { namespace {
__device__ __constant__ unsigned long long exp_table[32]={
 0x3ff0000000000000ull,0x3fefd9b0d3158574ull,0x3fefb5586cf9890full,0x3fef9301d0125b51ull,
 0x3fef72b83c7d517bull,0x3fef54873168b9aaull,0x3fef387a6e756238ull,0x3fef1e9df51fdee1ull,
 0x3fef06fe0a31b715ull,0x3feef1a7373aa9cbull,0x3feedea64c123422ull,0x3feece086061892dull,
 0x3feebfdad5362a27ull,0x3feeb42b569d4f82ull,0x3feeab07dd485429ull,0x3feea47eb03a5585ull,
 0x3feea09e667f3bcdull,0x3fee9f75e8ec5f74ull,0x3feea11473eb0187ull,0x3feea589994cce13ull,
 0x3feeace5422aa0dbull,0x3feeb737b0cdc5e5ull,0x3feec49182a3f090ull,0x3feed503b23e255dull,
 0x3feee89f995ad3adull,0x3feeff76f2fb5e47ull,0x3fef199bdd85529cull,0x3fef3720dcef9069ull,
 0x3fef5818dcfba487ull,0x3fef7c97337b9b5full,0x3fefa4afa2a490daull,0x3fefd0765b6e4540ull};
__device__ __forceinline__ bool finite(float x){return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u;}
__device__ __forceinline__ float exp_cpu(float x){
 const unsigned u=__float_as_uint(x),abstop=(u>>20)&0x7ff;
 if(abstop>=0x42b){
  if(u==0xff800000u)return 0.f;
  if(abstop>=0x7f8){if((u&0x007fffffu)!=0)return __uint_as_float(u|0x00400000u);return x;}
  if(x>0x1.62e42ep6f)return __fmul_rn(0x1p97f,0x1p97f);
  if(x< -0x1.9fe368p6f)return __fmul_rn(0x1p-95f,0x1p-95f);
  if(x< -0x1.9d1d9ep6f)return __fmul_rn(0x1.4p-75f,0x1.4p-75f);
 }
 const double xd=double(x),A=0x1.71547652b82fep5,shift=0x1.8p52;
 const double image=__fma_rn(A,xd,shift);
 const unsigned long long ki=static_cast<unsigned long long>(__double_as_longlong(image));
 const double kd=__dsub_rn(image,shift);
 const double r=__fma_rn(A,xd,-kd);
 const unsigned long long sb=exp_table[ki&31ull]+(ki<<47);
 const double s=__longlong_as_double(static_cast<long long>(sb));
 const double z=__fma_rn(0x1.c6af84b912394p-20,r,0x1.ebfce50fac4f3p-13);
 const double r2=__dmul_rn(r,r);
 const double y0=__fma_rn(0x1.62e42ff0c52d6p-6,r,1.0);
 const double y=__fma_rn(z,r2,y0);
 return __double2float_rn(__dmul_rn(y,s));
}
// x86 MAXPS returns its second operand on unordered or equal operands.
__device__ __forceinline__ float max_cpu(float a,float b){return a>b?a:b;}
__device__ __forceinline__ unsigned short half_cpu(float f){
 const unsigned w=__float_as_uint(f),twice=w+w,sign=w&0x80000000u;
 float base=__fmul_rn(__fmul_rn(__uint_as_float(w&0x7fffffffu),0x1p112f),0x1p-110f);
 unsigned bias=twice&0xff000000u;if(bias<0x71000000u)bias=0x71000000u;
 base=__fadd_rn(__uint_as_float((bias>>1)+0x07800000u),base);
 const unsigned v=__float_as_uint(base),ns=((v>>13)&0x7c00u)+(v&0xfffu);
 return static_cast<unsigned short>((sign>>16)|(twice>0xff000000u?0x7e00u:ns));
}
__device__ __forceinline__ int quant_cpu(float x,float reciprocal){
 const float v=__fmul_rn(x,reciprocal);
 // x86 indefinite int32 on NaN/overflow; two signed saturation packs -> -128.
 if(!finite(v)||v>=2147483648.f||v< -2147483648.f)return -128;
 int q=__float2int_rn(v);q=q>32767?32767:q< -32768?-32768:q;
 return q>127?127:q< -128?-128:q;
}
__device__ __forceinline__ void quant_block(const float*x,unsigned char*y,unsigned*error){
 float a[8];
 #pragma unroll
 for(int j=0;j<8;++j){a[j]=__uint_as_float(__float_as_uint(x[j])&0x7fffffffu);
  #pragma unroll
  for(int g=1;g<4;++g)a[j]=max_cpu(a[j],__uint_as_float(__float_as_uint(x[8*g+j])&0x7fffffffu));
 }
 float b[4];
 #pragma unroll
 for(int j=0;j<4;++j)b[j]=max_cpu(a[j+4],a[j]);
 const float c0=max_cpu(b[0],b[2]),c1=max_cpu(b[1],b[3]);
 const float maximum=max_cpu(c0,c1);
 const float d=__fdiv_rn(maximum,127.f),id=maximum!=0.f?__fdiv_rn(127.f,maximum):0.f;
 const unsigned short h=half_cpu(d);y[0]=static_cast<unsigned char>(h);y[1]=static_cast<unsigned char>(h>>8);
 #pragma unroll
 for(int j=0;j<32;++j){y[2+j]=static_cast<unsigned char>(quant_cpu(x[j],id));if(error&&!finite(x[j]))atomicOr(error,1u);}
 if(error&&(h&0x7c00u)==0x7c00u)atomicOr(error,2u);
}
__global__ void exp_kernel(const float*x,float*y,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=exp_cpu(x[i]);}
__global__ void quant_kernel(const float*x,unsigned char*y,int blocks,unsigned*error){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<blocks)quant_block(x+size_t(i)*32,y+size_t(i)*34,error);}
__device__ __forceinline__ float scale_cpu(unsigned char x){
 if(x==0||x==0x7f)return 0.f;
 unsigned e=(x>>3)&15,m=x&7;
 return e?__uint_as_float(((e+119u)<<23)|(m<<20)):__fmul_rn(float(m),0x1p-10f);
}
__device__ __forceinline__ float half_expand(const unsigned char*p){unsigned short h=static_cast<unsigned short>(p[0]|unsigned(p[1])<<8);return __half2float(__ushort_as_half(h));}
__device__ __forceinline__ float row_dot(const unsigned char*w,const unsigned char*q,int blocks,int lane){
 constexpr unsigned char lut[8]={0,1,2,3,4,6,8,12};float acc=0.f;
 for(int ib=0;ib<blocks;++ib){
  const unsigned char*wb=w+size_t(ib)*36;const unsigned char*yb=q+size_t(2*ib+lane/8)*34;
  unsigned pw=0,py=0;
  #pragma unroll
  for(int k=0;k<4;++k){int el=(lane%4)*4+k;unsigned byte=wb[4+(lane/4)*8+(el&7)];unsigned code=(byte>>(el>=8?4:0))&15;
   unsigned yy=yb[2+(lane%8)*4+k];if(code&8)yy=(0u-yy)&255u;
   pw|=unsigned(lut[code&7])<<(8*k);py|=yy<<(8*k);
  }
  const int p=__dp4a(static_cast<int>(pw),static_cast<int>(py),0);
  const float sc=__fmul_rn(scale_cpu(wb[lane/4]),half_expand(yb));
  acc=__fmaf_rn(sc,__int2float_rn(p),acc);
 }
 #pragma unroll
 for(int d=8;d;d>>=1)acc=__fadd_rn(acc,__shfl_xor_sync(0xffffffffu,acc,d,16));
 return acc;
}
__global__ void dot_kernel(const unsigned char*w,size_t row_bytes,const unsigned char*q,int n,int rows,int lo,int hi,float*out){
 int lane=threadIdx.x&15,row=int(blockIdx.x)*(blockDim.x/16)+int(threadIdx.x)/16+lo;
 // Every active16-lane group uses a width16 shuffle. Entire warp executes the
 // shuffle, including inactive padded row groups, to honor the full mask.
 const bool valid=row<hi;const unsigned char*wr=w+size_t(valid?row:lo)*row_bytes;
 const unsigned char*qr=q+size_t(blockIdx.y)*(n/32*34);
 float dot=row_dot(wr,qr,n/64,lane);
 if(valid&&lane==0)out[size_t(blockIdx.y)*rows+row]=dot;
}
__global__ void gu_kernel(const unsigned char*w,const unsigned char*q,int T,float*g,float*u){
 int lane=threadIdx.x&15,r=int(blockIdx.x)*(blockDim.x/16)+int(threadIdx.x)/16;
 // F640 is a multiple of16 rows/CTA, so no divergent warp tail.
 if(r>=2*F)return;int row=r%F;const unsigned char*wr=w+(r>=F?UpOffset:0)+size_t(row)*GuRow;
 float d=row_dot(wr,q+size_t(blockIdx.y)*InputQ,H/64,lane);
 if(lane==0)(r>=F?u:g)[size_t(blockIdx.y)*F+row]=d;
}
__global__ void hidden_kernel(const unsigned char*w,const float*g,const float*u,float*gs,float*us,float*h,int count,unsigned*error){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const float*sc=reinterpret_cast<const float*>(w+TailOffset);
 float a=__fmul_rn(g[i],sc[0]),b=__fmul_rn(u[i],sc[1]);gs[i]=a;us[i]=b;
 const float neg=__uint_as_float(__float_as_uint(a)^0x80000000u);
 const float d=__fadd_rn(1.f,exp_cpu(neg));const float v=__fmul_rn(__fdiv_rn(a,d),b);h[i]=v;
 if(!finite(a)||!finite(b)||!finite(v)||!finite(sc[0])||!finite(sc[1]))atomicOr(error,4u);
}
__global__ void down_kernel(const unsigned char*w,const unsigned char*q,float*raw,float*out,unsigned*error){
 int lane=threadIdx.x&15,r=int(blockIdx.x)*(blockDim.x/16)+int(threadIdx.x)/16;if(r>=H)return;
 const float d=row_dot(w+DownOffset+size_t(r)*DownRow,q+size_t(blockIdx.y)*HiddenQ,F/64,lane);
 if(lane==0){const float sd=reinterpret_cast<const float*>(w+TailOffset)[2];const float z=__fmul_rn(d,sd);size_t at=size_t(blockIdx.y)*H+r;raw[at]=d;out[at]=z;if(!finite(d)||!finite(sd)||!finite(z))atomicOr(error,8u);}
}
bool aligned(const void*p,size_t n){return p&&(reinterpret_cast<uintptr_t>(p)&(n-1))==0;}
struct Span{const void*p;size_t bytes;};
bool disjoint(const Span*v,int n){for(int i=0;i<n;++i){uintptr_t a=reinterpret_cast<uintptr_t>(v[i].p);if(!a||v[i].bytes>UINTPTR_MAX-a)return false;for(int j=0;j<i;++j){uintptr_t b=reinterpret_cast<uintptr_t>(v[j].p);if(a<b+v[j].bytes&&b<a+v[i].bytes)return false;}}return true;}
} // anonymous
cudaError_t exp_probe(const float*x,float*y,int n,cudaStream_t s){if(!s||!aligned(x,4)||!aligned(y,4)||n<1||n>(1<<20))return cudaErrorInvalidValue;Span v[]={{x,size_t(n)*4},{y,size_t(n)*4}};if(!disjoint(v,2))return cudaErrorInvalidValue;exp_kernel<<<(n+255)/256,256,0,s>>>(x,y,n);return cudaGetLastError();}
cudaError_t quant_probe(const float*x,uint8_t*y,int blocks,cudaStream_t s){if(!s||!aligned(x,4)||!y||blocks<1||blocks>(1<<16))return cudaErrorInvalidValue;Span v[]={{x,size_t(blocks)*128},{y,size_t(blocks)*34}};if(!disjoint(v,2))return cudaErrorInvalidValue;quant_kernel<<<(blocks+127)/128,128,0,s>>>(x,y,blocks,nullptr);return cudaGetLastError();}
cudaError_t dot_probe(const uint8_t*w,size_t rb,const uint8_t*q,int n,int rows,int T,int lo,int hi,float*out,cudaStream_t s){
 if(!s||!w||!q||!aligned(out,4)||(n!=H&&n!=F)||rows<1||rows>H||rb!=size_t(n/64*36)||T<1||T>MaxT||lo<0||hi<lo||hi>rows)return cudaErrorInvalidValue;
 Span v[]={{w,size_t(rows)*rb},{q,size_t(T)*(n/32)*34},{out,size_t(T)*rows*4}};if(!disjoint(v,3))return cudaErrorInvalidValue;
 if(lo==hi)return cudaSuccess;
 dot_kernel<<<dim3((hi-lo+15)/16,T),256,0,s>>>(w,rb,q,n,rows,lo,hi,out);return cudaGetLastError();
}
cudaError_t chain(const Images&i,int T,cudaStream_t s){
 if(!s||T<1||T>MaxT||!aligned(i.weights,4)||!aligned(i.input,4)||!i.input_q||!aligned(i.gate_raw,4)||!aligned(i.up_raw,4)||!aligned(i.gate_scaled,4)||!aligned(i.up_scaled,4)||!aligned(i.hidden,4)||!i.hidden_q||!aligned(i.down_raw,4)||!aligned(i.output,4)||!aligned(i.error,4))return cudaErrorInvalidValue;
 Span spans[]={{i.weights,Blob},{i.input,size_t(T)*H*4},{i.input_q,size_t(T)*InputQ},{i.gate_raw,size_t(T)*F*4},{i.up_raw,size_t(T)*F*4},{i.gate_scaled,size_t(T)*F*4},{i.up_scaled,size_t(T)*F*4},{i.hidden,size_t(T)*F*4},{i.hidden_q,size_t(T)*HiddenQ},{i.down_raw,size_t(T)*H*4},{i.output,size_t(T)*H*4},{i.error,4}};if(!disjoint(spans,12))return cudaErrorInvalidValue;
 quant_kernel<<<(T*H/32+127)/128,128,0,s>>>(i.input,i.input_q,T*H/32,i.error);cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
 gu_kernel<<<dim3(2*F/16,T),256,0,s>>>(i.weights,i.input_q,T,i.gate_raw,i.up_raw);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 hidden_kernel<<<(T*F+255)/256,256,0,s>>>(i.weights,i.gate_raw,i.up_raw,i.gate_scaled,i.up_scaled,i.hidden,T*F,i.error);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 quant_kernel<<<(T*F/32+127)/128,128,0,s>>>(i.hidden,i.hidden_q,T*F/32,i.error);e=cudaGetLastError();if(e!=cudaSuccess)return e;
 down_kernel<<<dim3(H/16,T),256,0,s>>>(i.weights,i.hidden_q,i.down_raw,i.output,i.error);return cudaGetLastError();
}
} // namespace
