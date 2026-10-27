#include "strata/ds41/native_prefill.hpp"
#include "strata/ds41/prefill_ops.hpp"
#include "strata/ds41/ops.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>
namespace strata::ds41 {
namespace {
using bf16=__nv_bfloat16;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("native prefill: ")+cudaGetErrorString(e));}
__global__ void gather(const uint16_t*source,const int32_t*ids,int rows,int source_rows,int dim,uint16_t*out){
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=size_t(rows)*dim)return;
    int row=ids[at/dim];assert(row>=0&&row<source_rows);out[at]=source[size_t(row)*dim+at%dim];
}
__global__ void scatter(const bf16*source,const int32_t*ids,int rows,int output_rows,int dim,float*out){
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=size_t(rows)*dim)return;
    int row=ids[at/dim];assert(row>=0&&row<output_rows);out[size_t(row)*dim+at%dim]+=__bfloat162float(source[at]);
}
void rows_valid(const void*source,const int32_t*ids,int rows,int capacity,int dim,const void*out){
    if(!source||!ids||!out||rows<1||rows>16384||capacity<rows||capacity>1048576||dim<32||dim>8192||dim%32)
        throw std::invalid_argument("native gathered expert: invalid row geometry");
}
__global__ void unpack(NativeDeviceProjection p,int rows,int columns,bf16*out){
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(rows)*columns)return;
    int row=int(index/columns),col=int(index%columns);float value;
    if(p.bits==0){
        const uint8_t packed=p.weights[size_t(row)*(columns/2)+col/2];int code=(packed>>(4*(col%2)))&15;
        const int magnitude=code&7;
        const uint32_t raw=magnitude<2?(magnitude?0x3f000000u:0u):(uint32_t((magnitude>>1)+126)<<23)|(uint32_t(magnitude&1)<<22);
        const float decoded=__uint_as_float(raw|((code&8)?0x80000000u:0u));
        uint8_t scale=p.scales[size_t(row)*(columns/32)+col/32];
        const float factor=scale==255?nanf(""):__uint_as_float(scale?uint32_t(scale)<<23:0x00400000u);
        value=decoded*factor;
    }else{
        size_t stride=size_t(columns)*p.bits/8+size_t(columns/32)*2;
        const uint8_t*w=p.weights+size_t(row)*stride;int code=0;
        int at=col*p.bits,byte=at/8;code=((int(w[byte])|(int(w[byte+1])<<8))>>(at%8))&((1<<p.bits)-1);
        const auto*s=w+size_t(columns)*p.bits/8+(col/32)*2;
        uint16_t bits=uint16_t(s[0])|(uint16_t(s[1])<<8);float scale=__uint_as_float(uint32_t(bits)<<16);
        value=(code==(1<<p.bits)-1||!isfinite(scale)||scale<0)?nanf(""):float(code-((1<<(p.bits-1))-1))*scale;
    }
    out[index]=__float2bfloat16_rn(value);
}
__global__ void routed_hidden(const bf16*gate,const bf16*up,const float*alpha,int rows,int inter,float limit,bf16*out){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(rows)*inter)return;
    float g=__bfloat162float(gate[i]),u=__bfloat162float(up[i]);
    if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
    float weight=alpha[i/inter];
    out[i]=__float2bfloat16_rn((g/(1.f+expf(-g))*u)*weight);
}
void validate(NativeDeviceProjection p){
    if(!p.weights||(p.bits==0&&!p.scales)||(p.bits!=0&&p.bits!=2&&p.bits!=3&&p.bits!=4&&p.bits!=6&&p.bits!=8))
        throw std::invalid_argument("native prefill: unsupported packed projection");
}
}
void native_prefill_gather(const uint16_t*source,const int32_t*ids,int rows,int capacity,int dim,uint16_t*out){
    rows_valid(source,ids,rows,capacity,dim,out);gather<<<(size_t(rows)*dim+255)/256,256>>>(source,ids,rows,capacity,dim,out);check(cudaGetLastError());
}
void native_prefill_scatter(const uint16_t*source,const int32_t*ids,int rows,int capacity,int dim,float*out){
    rows_valid(source,ids,rows,capacity,dim,out);scatter<<<(size_t(rows)*dim+255)/256,256>>>(reinterpret_cast<const bf16*>(source),ids,rows,capacity,dim,out);check(cudaGetLastError());
}
void native_expert_prefill(const uint16_t*x,const float*alpha,int rows,int dim,int inter,float limit,
                          NativeDeviceProjection gate,NativeDeviceProjection up,NativeDeviceProjection down,
                          uint16_t*out,void*workspace,size_t bytes){
    const auto l=native_prefill_layout(rows,dim,inter);
    if(!x||!alpha||!out||!workspace||bytes<l.bytes||uintptr_t(workspace)%256||!std::isfinite(limit)||limit<0)
        throw std::invalid_argument("native prefill: invalid pointers/limit/workspace");
    validate(gate);validate(up);validate(down);
    auto*base=static_cast<uint8_t*>(workspace);
    auto*w=reinterpret_cast<bf16*>(base+l.weight);auto*xq=reinterpret_cast<bf16*>(base+l.x);
    auto*g=reinterpret_cast<bf16*>(base+l.gate);auto*u=reinterpret_cast<bf16*>(base+l.up);auto*h=reinterpret_cast<bf16*>(base+l.hidden);
    auto*tmp=reinterpret_cast<float*>(base+l.tmp);
    check(cudaMemcpyAsync(xq,x,size_t(rows)*dim*2,cudaMemcpyDeviceToDevice,0));
    ops::act_quant_inplace(xq,rows*dim);
    const int blocks=(dim*inter+255)/256;
    unpack<<<blocks,256>>>(gate,inter,dim,w);check(cudaGetLastError());
    prefill::bf16_gemm(xq,w,rows,dim,inter,g,nullptr,tmp);
    unpack<<<blocks,256>>>(up,inter,dim,w);check(cudaGetLastError());
    prefill::bf16_gemm(xq,w,rows,dim,inter,u,nullptr,tmp);
    routed_hidden<<<(rows*inter+255)/256,256>>>(g,u,alpha,rows,inter,limit,h);check(cudaGetLastError());
    ops::act_quant_inplace(h,rows*inter);
    unpack<<<blocks,256>>>(down,dim,inter,w);check(cudaGetLastError());
    prefill::bf16_gemm(h,w,rows,inter,dim,reinterpret_cast<bf16*>(out),nullptr,tmp);
}
void native_expert_decode_weights(int dim,int inter,NativeDeviceProjection gate,NativeDeviceProjection up,
                                 NativeDeviceProjection down,uint16_t*out,size_t elements){
    native_prefill_layout(1,dim,inter); // Reuse bounded geometry validation.
    const size_t matrix=size_t(dim)*inter;
    if(!out||elements!=3*matrix)throw std::invalid_argument("invalid decoded expert matrix storage");
    validate(gate);validate(up);validate(down);
    const int blocks=int((matrix+255)/256);auto*w=reinterpret_cast<bf16*>(out);
    unpack<<<blocks,256>>>(gate,inter,dim,w);check(cudaGetLastError());
    unpack<<<blocks,256>>>(up,inter,dim,w+matrix);check(cudaGetLastError());
    unpack<<<blocks,256>>>(down,dim,inter,w+2*matrix);check(cudaGetLastError());
}
void native_expert_prefill_decoded(const uint16_t*x,const float*alpha,int rows,int dim,int inter,float limit,
                                 const uint16_t*decoded,size_t elements,uint16_t*out,void*workspace,size_t bytes){
    const auto l=native_prefill_layout(rows,dim,inter);const size_t matrix=size_t(dim)*inter;
    if(!x||!alpha||!decoded||elements!=3*matrix||!out||!workspace||bytes<l.bytes||uintptr_t(workspace)%256||!std::isfinite(limit)||limit<0)
        throw std::invalid_argument("invalid predecoded expert prefill buffers");
    auto*base=static_cast<uint8_t*>(workspace);const auto*w=reinterpret_cast<const bf16*>(decoded);
    auto*xq=reinterpret_cast<bf16*>(base+l.x);auto*g=reinterpret_cast<bf16*>(base+l.gate);
    auto*u=reinterpret_cast<bf16*>(base+l.up);auto*h=reinterpret_cast<bf16*>(base+l.hidden);auto*tmp=reinterpret_cast<float*>(base+l.tmp);
    check(cudaMemcpyAsync(xq,x,size_t(rows)*dim*2,cudaMemcpyDeviceToDevice,0));ops::act_quant_inplace(xq,rows*dim);
    prefill::bf16_gemm(xq,w,rows,dim,inter,g,nullptr,tmp);
    prefill::bf16_gemm(xq,w+matrix,rows,dim,inter,u,nullptr,tmp);
    routed_hidden<<<(rows*inter+255)/256,256>>>(g,u,alpha,rows,inter,limit,h);check(cudaGetLastError());
    ops::act_quant_inplace(h,rows*inter);
    prefill::bf16_gemm(h,w+2*matrix,rows,inter,dim,reinterpret_cast<bf16*>(out),nullptr,tmp);
}
}
