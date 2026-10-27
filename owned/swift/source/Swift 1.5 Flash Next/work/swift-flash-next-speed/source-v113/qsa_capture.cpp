// Private owned executable only: exact QSA query-register kernel on captured T3..6.
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" void* swift_score_kernel112(int);
extern "C" cudaError_t __real_cudaGraphInstantiate(cudaGraphExec_t*,cudaGraph_t,unsigned long long);
namespace {
[[noreturn]] void fail(const char* why,cudaError_t e=cudaSuccess){std::fprintf(stderr,"Swift QSA113 FATAL: %s (%s)\n",why,cudaGetErrorString(e));std::abort();}
void check(cudaError_t e,const char* msg){if(e!=cudaSuccess)fail(msg,e);}
bool enabled(){static const bool flag=[](){const char* p=std::getenv("STRATA_SWIFT_QSA_REGISTER113");if(!p||!p[0]||!std::strcmp(p,"0"))return false;if(std::strcmp(p,"1"))fail("flag must be0/1");return true;}();return flag;}
void verify_device(){static const bool checked=[](){int n=0;check(cudaGetDeviceCount(&n),"count");if(n!=1)fail("one visible GPU required");cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,0),"properties");const unsigned char u[]={0x7c,0x89,0xb7,0x99,0x91,0x49,0x74,0x94,0x39,0x90,0x02,0x81,0x3f,0x9a,0x57,0x75};if(std::memcmp(u,p.uuid.bytes,16)||p.major!=8||p.minor!=9)fail("exact GPU0 SM89 required");return true;}();(void)checked;}
std::atomic<unsigned> printed{0};
bool nested_score(cudaGraph_t graph){size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n),"nested nodes count");std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n),"nested nodes");for(auto node:nodes){cudaGraphNodeType t;check(cudaGraphNodeGetType(node,&t),"nested type");if(t==cudaGraphNodeTypeKernel){cudaKernelNodeParams p{};check(cudaGraphKernelNodeGetParams(node,&p),"nested params");const char* name=nullptr;check(cudaFuncGetName(&name,p.func),"nested name");if(name&&std::strstr(name,"block_scores_multi_kernel"))return true;}else if(t==cudaGraphNodeTypeGraph){cudaGraph_t g;check(cudaGraphChildGraphNodeGetGraph(node,&g),"nested graph");if(nested_score(g))return true;}}return false;}
}
extern "C" cudaError_t __wrap_cudaGraphInstantiate(cudaGraphExec_t* exec,cudaGraph_t graph,unsigned long long flags){
 const cudaError_t result=__real_cudaGraphInstantiate(exec,graph,flags);if(result!=cudaSuccess||!enabled())return result;
 verify_device();size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count),"nodes count");std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count),"nodes");
 for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type),"type");if(type==cudaGraphNodeTypeGraph){cudaGraph_t g;check(cudaGraphChildGraphNodeGetGraph(node,&g),"child graph");if(nested_score(g))fail("nested score graph not yet supported");continue;}if(type!=cudaGraphNodeTypeKernel)continue;
  cudaKernelNodeParams p{};check(cudaGraphKernelNodeGetParams(node,&p),"params");const char* name=nullptr;check(cudaFuncGetName(&name,p.func),"kernel name");if(!name||!std::strstr(name,"block_scores_multi_kernel"))continue;
  if(p.gridDim.x!=256||p.gridDim.y!=1||p.gridDim.z!=1||p.blockDim.x!=256||p.blockDim.y!=1||p.blockDim.z!=1||p.sharedMemBytes||!p.kernelParams||p.extra)fail("unexpected frozen score launch ABI");
  const int nq=*static_cast<const int*>(p.kernelParams[4]);const int64_t capacity=*static_cast<const int64_t*>(p.kernelParams[5]);if(nq<1||nq>8||capacity<1||capacity>262146)fail("unsupported score geometry");if(nq<3||nq>6)continue;
  p.func=swift_score_kernel112(nq);if(!p.func)fail("specialization missing");check(cudaGraphExecKernelNodeSetParams(*exec,node,&p),"replace score function only");const unsigned bit=1u<<nq;if(!(printed.fetch_or(bit)&bit))std::fprintf(stderr,"Swift QSA113: exact query-register kernel T=%d grid=256 capacity=%lld (weights/attention unchanged)\n",nq,(long long)capacity);
 }
 return result;
}
