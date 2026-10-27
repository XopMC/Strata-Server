#include "strata/ds41/native_main.hpp"
#include <cuda_bf16.h>
#include <stdexcept>
namespace strata::ds41 {
namespace {
constexpr int D=5120,H=4,L=3,W=128;
__global__ void capture(const __nv_bfloat16*hc,int rows,int position,int target,__nv_bfloat16*out){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*D)return;
    const int row=i/D,col=i%D;
    // Only the final128 positions of a large tile may write this ring.
    if(row<rows-W)return;
    const auto*p=hc+size_t(row)*H*D+col;
    const float a=__bfloat162float(p[0]),b=__bfloat162float(p[D]);
    const float c=__bfloat162float(p[2*D]),d=__bfloat162float(p[3*D]);
    out[(size_t((position+row)%W)*L+target)*D+col]=__float2bfloat16_rn(((a+b)+(c+d))*.25f);
}
__global__ void commit(const uint16_t*src,int keep,int position,uint16_t*dst){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=keep*L*D)return;
    const int row=i/(L*D),col=i%(L*D);
    dst[size_t((position+row)%W)*L*D+col]=src[size_t((position+row)%W)*L*D+col];
}
void check(){const auto code=cudaGetLastError();if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
}
void native_main_capture(const uint16_t*hc,int rows,int position,int target,uint16_t*ring,cudaStream_t stream){
    if(!hc||!ring||rows<1||rows>16384||position<0||position>1048576-rows||target<0||target>=L)throw std::invalid_argument("invalid DSpark main capture");
    capture<<<(rows*D+255)/256,256,0,stream>>>((const __nv_bfloat16*)hc,rows,position,target,(__nv_bfloat16*)ring);check();
}
void native_main_commit(const uint16_t*src,int keep,int position,uint16_t*ring,cudaStream_t stream){
    if(!src||!ring||keep<1||keep>4||position<0||position>1048576-keep)throw std::invalid_argument("invalid DSpark main commit");
    commit<<<(keep*L*D+255)/256,256,0,stream>>>(src,keep,position,ring);check();
}
}
