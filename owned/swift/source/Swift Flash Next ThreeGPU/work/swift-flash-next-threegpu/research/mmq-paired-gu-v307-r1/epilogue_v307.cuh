#pragma once
// Original R56 FP32 boundary and scalar operation order; no BF16 conversion is introduced.
__device__ __forceinline__ float boundary_v307(float v){float r;asm volatile("mov.b32 %0, %1;":"=f"(r):"f"(v));return r;}
__device__ __forceinline__ float swiglu_v307(float gate,float up,float sg,float su){
 const float g=gate*sg,u=up*su;return boundary_v307((g/(1.f+__expf(-g)))*u);
}
// Exact R56 4 values/thread, scale32 reduction XOR4/2/1, roundf and D4 stores.
// band is64 live values or384 zero-padding values, both multiples32.
template<int J,int BAND>
__device__ __forceinline__ void quant_band_v307(const float* h,const int32_t* ids,int jmax,
 const paired_v307::Epilogue& p,int base_k){
 constexpr int threads=ggml_cuda_mmq_get_nthreads(GGML_TYPE_NVFP4,J,false);
 const int thread=threadIdx.y*32+threadIdx.x;
 auto* y=(block_q8_1_mmq*)p.hq;
 for(int q=thread;q<J*(BAND/4);q+=threads){
  const int j=q/(BAND/4),k=(q%(BAND/4))*4;
  const float4 xi=BAND==64?make_float4(h[j*64+k],h[j*64+k+1],h[j*64+k+2],h[j*64+k+3]):make_float4(0.f,0.f,0.f,0.f);
  float amax=fabsf(xi.x);amax=fmaxf(amax,fabsf(xi.y));amax=fmaxf(amax,fabsf(xi.z));amax=fmaxf(amax,fabsf(xi.w));
#pragma unroll
  for(int offset=4;offset>0;offset>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,offset,32));
  const float d_inv=127.0f/amax;
  char4 quant;quant.x=roundf(xi.x*d_inv);quant.y=roundf(xi.y*d_inv);quant.z=roundf(xi.z*d_inv);quant.w=roundf(xi.w*d_inv);
  const float d=1.0f/d_inv;
  if(j<=jmax){
   const int64_t row=int64_t(ids[j])-p.row0;const int global_k=base_k+k;
   const int64_t ib=(global_k/128)*p.nr+row;const int iqs=global_k%128;
   ((char4*)y[ib].qs)[iqs/4]=quant;
   if(iqs%32==0)y[ib].d4[iqs/32]=d;
  }
 }
}
template<int J>
__device__ __forceinline__ void write_paired_v307(const float*sum,const int32_t*ids,float*dst,
 int stride,int jmax,const paired_v307::Epilogue&p,int it,int group){
 typedef tile<16,8,int> tile_C;
 constexpr int rows_per_warp=ggml_cuda_mmq_get_rows_per_warp(GGML_TYPE_NVFP4,J,false);
 constexpr int ntx=rows_per_warp/tile_C::I;
 const int i0=(threadIdx.y/ntx)*(ntx*tile_C::I);
 extern __shared__ int data_mul_mat_q[];
 float* exchange=(float*)(data_mul_mat_q+J); // ids[0..J) retained; original A/B banks dead after final K join.
 const float sg=p.tails[4*group],su=p.tails[4*group+1];
 // Store up band first; each original accumulator owner writes exactly one shared cell.
#pragma unroll
 for(int j0=0;j0<J;j0+=ntx*tile_C::J){
#pragma unroll
  for(int n=0;n<ntx;++n){
#pragma unroll
   for(int l=0;l<tile_C::ne;++l){
    const int j=j0+(threadIdx.y%ntx)*tile_C::J+tile_C::get_j(l);
    const int i=i0+n*tile_C::I+tile_C::get_i(l);
    const float value=sum[(j0/tile_C::J+n)*tile_C::ne+l];
    if(i>=64)exchange[j*64+i-64]=value;
    if(p.materialize_gu&&j<=jmax)dst[int64_t(ids[j])*stride+it*64+(i%64)+(i>=64?640:0)]=value;
   }
  }
 }
 __syncthreads();
 // Gate owners read corresponding up exactly once, then replace that same cell by H.
#pragma unroll
 for(int j0=0;j0<J;j0+=ntx*tile_C::J){
#pragma unroll
  for(int n=0;n<ntx;++n){
#pragma unroll
   for(int l=0;l<tile_C::ne;++l){
    const int j=j0+(threadIdx.y%ntx)*tile_C::J+tile_C::get_j(l);
    const int i=i0+n*tile_C::I+tile_C::get_i(l);
    if(i<64)exchange[j*64+i]=swiglu_v307(sum[(j0/tile_C::J+n)*tile_C::ne+l],exchange[j*64+i],sg,su);
   }
  }
 }
 __syncthreads();
 quant_band_v307<J,64>(exchange,ids,jmax,p,it*64);
 if(it==9)quant_band_v307<J,384>(exchange,ids,jmax,p,640); // original zero arithmetic, never memset.
 if(it==0&&p.row_sd){
  const int thread=threadIdx.y*32+threadIdx.x;
  for(int j=thread;j<J;j+=ggml_cuda_mmq_get_nthreads(GGML_TYPE_NVFP4,J,false))if(j<=jmax)p.row_sd[int64_t(ids[j])-p.row0]=p.tails[4*group+2];
 }
 __syncthreads(); // stream-K may reuse the same shared banks for its next whole tile.
}
