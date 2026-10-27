#include <cuda_runtime.h>
#include <cstdint>
namespace {
__global__ void check(const int32_t* a,const int32_t* b,const int32_t* steps,int nq,int64_t cap,uint64_t* counters){
 __shared__ unsigned errors[256];unsigned e=0;
 for(int i=threadIdx.x;i<nq*cap;i+=blockDim.x){int q=i/cap,j=i%cap;const int width=steps[q*4+3];if(j<width&&width>=0&&width<=cap)e+=a[i]!=b[i];}
 errors[threadIdx.x]=e;__syncthreads();
 if(threadIdx.x==0){uint64_t total=0,invalid=0,mismatch=0;for(int q=0;q<nq;++q){int w=steps[q*4+3];if(w<0||w>cap)++invalid;else total+=w;}for(int t=0;t<256;++t)mismatch+=errors[t];
  volatile uint64_t* out=counters;out[0]=out[0]+1;out[1]=out[1]+total;out[2]=out[2]+mismatch;out[3]=out[3]+invalid;__threadfence_system();
 }
}
}
extern "C" cudaError_t swift168_check_ids(const int32_t* a,const int32_t* b,const int32_t* st,int nq,int64_t cap,uint64_t* count,cudaStream_t cs){check<<<1,256,0,cs>>>(a,b,st,nq,cap,count);return cudaGetLastError();}
