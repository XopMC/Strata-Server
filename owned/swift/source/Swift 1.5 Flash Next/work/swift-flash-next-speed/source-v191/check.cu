#include <cuda_runtime.h>
#include <cstdint>
namespace swift191 {namespace {
__global__ void check_kernel(const int32_t* plan,const volatile int32_t* host,int cap,int entries,
 const volatile uint64_t* ready,uint32_t ring,uint32_t* status){
 const int t=threadIdx.x;const int po=(4+(cap+1)+2*cap+1)&~1;
 const int32_t* dst=plan+4+cap+1;const int32_t* tok=dst+cap;
 const volatile int32_t* hd=host+4+cap+1;const volatile int32_t* ht=hd+cap;
 const auto* ptr=reinterpret_cast<const unsigned long long*>(plan+po);
 const auto* hp=reinterpret_cast<const volatile unsigned long long*>(host+po);
 if(t==0){if((*ready>>1)!=ring||!(*ready&1))atomicOr(status,16u);if(host[2]!=0||plan[1]!=host[1]||plan[0]!=host[0])atomicOr(status,32u);atomicAdd(status+1,1u);}
 if(t>=entries)return;
 int dg=-1,hg=-1;for(int i=0;i<plan[1];++i)if(dst[i]==t){if(dg!=-1)atomicOr(status,64u);for(int g=0;g<plan[0];++g)if(i>=plan[4+g]&&i<plan[4+g+1]){dg=g;break;}if(tok[i]!=t/10)atomicOr(status,128u);}
 for(int i=0;i<host[1];++i)if(hd[i]==t){for(int g=0;g<host[0];++g)if(i>=host[4+g]&&i<host[4+g+1]){hg=g;break;}if(ht[i]!=t/10)atomicOr(status,128u);}
 if((dg<0)!=(hg<0)||(dg>=0&&hg>=0&&ptr[dg]!=hp[hg]))atomicOr(status,256u);
}
}
cudaError_t prepare_check(){cudaFuncAttributes a{};return cudaFuncGetAttributes(&a,check_kernel);}
cudaError_t check_plan(const int32_t* a,const int32_t* b,int cap,int n,const uint64_t* ready,uint32_t ring,uint32_t* s,cudaStream_t cs){check_kernel<<<1,128,0,cs>>>(a,b,cap,n,ready,ring,s);return cudaGetLastError();}
}
