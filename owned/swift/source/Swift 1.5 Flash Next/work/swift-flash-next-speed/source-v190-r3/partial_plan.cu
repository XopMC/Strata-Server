#include "partial_plan.hpp"
namespace swift190 { namespace {
__global__ void wait_kernel(const volatile uint64_t* ready,uint32_t ring,uint32_t* error){
 if(threadIdx.x)return;
 const auto start=clock64();const uint64_t expected=uint64_t(ring)<<1;
 while(*ready<expected){if(clock64()-start>1000000000ull){atomicOr(error,1u);return;}__nanosleep(64);}
 __threadfence_system();if((*ready>>1)!=ring)atomicOr(error,2u);
}
__global__ __launch_bounds__(MaxEntries) void plan_kernel(const int32_t* ids,int n,int k,
 const int32_t* res,int experts,const uint8_t* cache,const uint64_t* offsets,
 int slots,int64_t blob,int32_t* pl,int cap,const volatile uint64_t* ready,
 uint32_t ring,uint32_t* error){
 __shared__ int eid[MaxEntries],hit[MaxEntries],first[MaxEntries],count[MaxEntries],starts[MaxEntries],bad;
 __shared__ unsigned long long pointers[MaxEntries];
 const int t=threadIdx.x;if(!t)bad=0;__syncthreads();
 const uint64_t code=*ready;
 const bool enabled=(code>>1)==ring&&(code&1)&&!*error;
 int id=-1,slot=-1;
 if(t<n&&enabled){id=ids[t];if(id>=0&&id<experts)slot=res[id];else atomicOr(&bad,1);if(slot>=slots)atomicOr(&bad,2);}
 eid[t]=id;hit[t]=t<n&&enabled&&slot>=0&&slot<slots;first[t]=0;count[t]=0;starts[t]=0;
 if(hit[t])pointers[t]=reinterpret_cast<unsigned long long>(cache)+(offsets?offsets[slot]:uint64_t(slot)*uint64_t(blob));
 __syncthreads();
 int leader=t,rank=0,number=0;
 if(hit[t]&&!bad){for(int j=0;j<n;++j)if(hit[j]&&eid[j]==id){leader=min(leader,j);rank+=j<t;++number;}if(leader==t){first[t]=1;count[t]=number;}}
 __syncthreads();
 int32_t* bounds=pl+4;int32_t* dst=bounds+cap+1;int32_t* tok=dst+cap;
 const int po=(4+(cap+1)+2*cap+1)&~1;
 auto* ptr=reinterpret_cast<unsigned long long*>(pl+po);int32_t* bounds2=pl+po+4*cap;
 if(first[t]){int group=0,start=0;for(int j=0;j<t;++j){group+=first[j];start+=count[j];}ptr[group]=pointers[t];bounds[group]=start;starts[t]=start;}
 __syncthreads();
 if(hit[t]&&!bad){const int pos=starts[leader]+rank;dst[pos]=t;tok[pos]=t/k;}
 if(!t){int groups=0,entries=0;for(int j=0;j<n;++j){groups+=first[j];entries+=count[j];}bounds[groups]=entries;bounds2[0]=entries;pl[0]=groups;pl[1]=entries;pl[2]=0;if(bad)atomicOr(error,4u*unsigned(bad));}
}
}
cudaError_t prepare(){
 cudaFuncAttributes attributes{};
 cudaError_t e=cudaFuncGetAttributes(&attributes,wait_kernel);
 if(e!=cudaSuccess)return e;
 return cudaFuncGetAttributes(&attributes,plan_kernel);
}
cudaError_t wait_ready(const uint64_t* ready,uint32_t ring,uint32_t* error,cudaStream_t stream){
 if(!ready||!error||!ring)return cudaErrorInvalidValue;
 wait_kernel<<<1,32,0,stream>>>(ready,ring,error);return cudaGetLastError();
}
cudaError_t plan(const int32_t* ids,int n,int k,const int32_t* res,int experts,
 const uint8_t* cache,const uint64_t* offsets,int slots,int64_t blob,int32_t* out,
 int cap,const uint64_t* ready,uint32_t ring,uint32_t* error,cudaStream_t stream){
 if(n<0||n>MaxEntries||cap<n||cap>MaxEntries||k<1||experts<1||slots<1||blob<1||!ready||!error||!ring)return cudaErrorInvalidValue;
 plan_kernel<<<1,MaxEntries,0,stream>>>(ids,n,k,res,experts,cache,offsets,slots,blob,out,cap,ready,ring,error);
 return cudaGetLastError();
}
}
