#include "common.cuh"
#include "paired_v307.hpp"
#include <stdexcept>
#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#include "mmq_paired_v307.cuh"
DECL_MMQ_CASE(GGML_TYPE_NVFP4);
namespace paired_v307 {
bool dispatch_pair(ggml_backend_cuda_context&ctx,const strata::prefill::mmq::Product&p,const Epilogue&e,cudaStream_t stream){
 const int id=ggml_cuda_get_device();const auto&device=ggml_cuda_info().devices[id];
 int J_best=0,ntiles_J_best=INT_MAX;
 // Literal original J policy; no tuning or override.
 for(int J=8;J<=128&&ntiles_J_best>1;J+=8){
  const auto c=ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4,J,false,device.cc);
  if(c.type==GGML_TYPE_COUNT||mmq_get_nbytes_shared(c,device.cc)>device.smpbo)continue;
  const int ntiles_x=(p.max_rows+c.J-1)/c.J;
  if(ntiles_x<ntiles_J_best){J_best=J;ntiles_J_best=ntiles_x;}
 }
 if(!J_best)return false;
 const auto c=ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4,J_best,false,device.cc);
 if(c.I!=128||mmq_get_nbytes_shared(c,device.cc)<size_t((c.J+64*c.J)*sizeof(float)))return false;
 const int ntiles=((p.max_rows+c.J-1)/c.J)*10*p.n;
 const int waves=(ntiles+device.nsm-1)/device.nsm;
 const int efficiency=100*ntiles/(device.nsm*waves);
 const int original_grid=GGML_CUDA_CC_IS_NVIDIA(device.cc)&&efficiency>=90?ntiles:device.nsm;
 if(ggml_cuda_mmq_get_stream_k(GGML_TYPE_NVFP4,J_best,false,device.cc)&&ntiles%original_grid!=0)return false;
 const paired_v307_mmq_args a={(const char*)p.w,GGML_TYPE_NVFP4,(const int*)p.xq,p.ids,p.bounds,p.dst,p.y_scale,
  p.w_cols,p.w_rows,p.total_rows,p.w_cols/64,p.total_rows,p.ld_dst,
  p.n,p.n,(int64_t)(p.expert_bytes/36),0,0,1,1,0,0,0,p.max_rows,p.max_rows,e};
 paired_v307_mul_mat_q_case<GGML_TYPE_NVFP4>(ctx,a,stream);
 return true;
}
}
