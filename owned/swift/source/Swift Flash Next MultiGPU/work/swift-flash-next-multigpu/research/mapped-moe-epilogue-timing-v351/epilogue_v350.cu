// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// PRIVATE STANDALONE COMPONENT. No public ABI/production source change.
#include "epilogue_v350.hpp"
#include <cstdlib>
namespace mapped_moe_v350 {
namespace {
bool span(const void*p,size_t n){const auto u=reinterpret_cast<uintptr_t>(p);return p&&n<=UINTPTR_MAX-u;}
bool overlap(const void*a,size_t an,const void*b,size_t bn){const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+bn&&y<x+an;}
// Original add_hits_kernel TU has NO --use_fast_math. The combine TU DOES.
// Force the original separately rounded, non-FTZ add here; don't fold +0+hit.
__device__ __forceinline__ float zero_add(float hit){
 float p;asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(p) : "f"(0.0f),"f"(hit));return p;
}
__device__ __forceinline__ float4 part(const float4* cpu,const float4* hits,int mask,
 int expert,int64_t at,float4* receipt){
 float4 p;
 if((mask>>expert)&1){const float4 h=hits[at];p=make_float4(zero_add(h.x),zero_add(h.y),zero_add(h.z),zero_add(h.w));}
 else p=cpu[at]; // raw load after unchanged original final-ready wait
 if(receipt)receipt[at]=p;
 return p;
}
__global__ void mapped_epilogue_v350(const float4* __restrict__ cpu,
 const float4* __restrict__ hits,const int32_t* __restrict__ dst,const int32_t* __restrict__ count,
 const float* __restrict__ weights,const float4* __restrict__ shared,
 float4* __restrict__ output,float4* __restrict__ receipt,int64_t n4){
 const int tk=blockIdx.y;
 __shared__ int hit_mask;
 if(threadIdx.x==0){int mask=0;const int n=*count;
  for(int i=0;i<n;++i){const int r=dst[i];if(r/10==tk)mask|=1<<(r%10);}
  hit_mask=mask;
 }
 __syncthreads();
 const int mask=hit_mask;const int64_t base=int64_t(tk)*10*n4;
 weights+=tk*10;shared+=tk*n4;output+=tk*n4;
 const int64_t c4=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(c4>=n4)return;
 // Verbatim combine_k10_vec4 operation sequence; operand selection only changed.
 const float w0=__ldg(weights+0);
 const float4 p0=part(cpu,hits,mask,0,base+c4,receipt);
 float4 sum=make_float4(p0.x*w0,p0.y*w0,p0.z*w0,p0.w*w0);
#pragma unroll
 for(int expert=1;expert<10;++expert){
  const float w=__ldg(weights+expert);
  const float4 p=part(cpu,hits,mask,expert,base+int64_t(expert)*n4+c4,receipt);
  sum.x=fmaf(p.x,w,sum.x);sum.y=fmaf(p.y,w,sum.y);
  sum.z=fmaf(p.z,w,sum.z);sum.w=fmaf(p.w,w,sum.w);
 }
 const float4 sh=shared[c4];
 sum.x+=sh.x;sum.y+=sh.y;sum.z+=sh.z;sum.w+=sh.w;output[c4]=sum;
}
}
bool enabled(){static const bool value=[](){const char*p=std::getenv("STRATA_MAPPED_MOE_EPILOGUE_V350");return p&&p[0]=='1'&&p[1]=='\0';}();return value;}
cudaError_t launch(const Args&a,bool&admitted){
 admitted=false;
 const uintptr_t bits=reinterpret_cast<uintptr_t>(a.cpu)|reinterpret_cast<uintptr_t>(a.hits)|reinterpret_cast<uintptr_t>(a.shared)|reinterpret_cast<uintptr_t>(a.output)|reinterpret_cast<uintptr_t>(a.parts_receipt);
 if(!enabled()||a.width!=2560||a.k!=10||a.tokens<1||a.tokens>8||!a.stream||
  !a.cpu||!a.hits||!a.dst||!a.count||!a.weights||!a.shared||!a.output||(bits&15u))return cudaSuccess;
 const size_t rows=size_t(a.tokens)*10,parts=rows*2560*4,out=size_t(a.tokens)*2560*4;
 const void*inputs[]={a.cpu,a.hits,a.dst,a.count,a.weights,a.shared};
 const size_t sizes[]={parts,parts,rows*4,4,rows*4,out};
 if(!span(a.output,out)||(a.parts_receipt&&!span(a.parts_receipt,parts)))return cudaSuccess;
 for(int i=0;i<6;++i)if(!span(inputs[i],sizes[i])||overlap(a.output,out,inputs[i],sizes[i])||
  (a.parts_receipt&&overlap(a.parts_receipt,parts,inputs[i],sizes[i])))return cudaSuccess;
 if(a.parts_receipt&&overlap(a.output,out,a.parts_receipt,parts))return cudaSuccess;
 // Allocation ownership/extents/disjointness and valid unique dst/count come
 // from the bounded component's immutable host plan, not arbitrary GPU data.
 admitted=true;
 mapped_epilogue_v350<<<dim3(5,unsigned(a.tokens)),128,0,a.stream>>>(
  reinterpret_cast<const float4*>(a.cpu),reinterpret_cast<const float4*>(a.hits),a.dst,a.count,
  a.weights,reinterpret_cast<const float4*>(a.shared),reinterpret_cast<float4*>(a.output),
  reinterpret_cast<float4*>(a.parts_receipt),640);
 return cudaGetLastError();
}
void*function(){return reinterpret_cast<void*>(mapped_epilogue_v350);}
}
