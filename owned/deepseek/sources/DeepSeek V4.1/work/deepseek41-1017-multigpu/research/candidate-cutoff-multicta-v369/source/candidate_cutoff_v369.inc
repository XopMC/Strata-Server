#include "candidate_cutoff_v369.h"
#include <cstdint>
#include <limits>
namespace ds369::selected {
namespace {
constexpr int Part369=1024,Threads369=256,Small369=4096;
constexpr int64_t Capacity369=524288;
constexpr size_t Header369=256;
struct Workspace369{Ds369State*state;uint32_t*hist;uint32_t*ties;uint32_t*equal_before;int parts;};
size_t bytes369(int64_t capacity,int block){
 if(capacity<0||capacity>Capacity369||block<1||block>64)return 0;
 if(!capacity)return 0;
 const uint64_t blocks=(uint64_t(capacity)+uint64_t(block)-1)/uint64_t(block),parts=(blocks+Part369-1)/Part369;
 return Header369+size_t(parts)*(256+2)*sizeof(uint32_t);
}
Workspace369 split369(void*p,int64_t capacity,int block){
 const int parts=int(((capacity+block-1)/block+Part369-1)/Part369);
 auto*hist=reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(p)+Header369);
 return{static_cast<Ds369State*>(p),hist,hist+size_t(parts)*256,hist+size_t(parts)*257,parts};
}
__global__ void initialize369(Ds369State*s,const int*pos,int ratio,int64_t cap,int requested,int block){
 if(threadIdx.x)return;
 const int64_t positions=pos?live_length(pos,ratio,cap):cap;
 const int64_t blocks=positions?(positions-1)/block+1:0;
 const int k=int(blocks<requested?blocks:requested);
 // Tiny/full/zero paths execute literal conditional old kernel below, never an
 // alternative floating operation. Large kernels simply return for mode!=3.
 const int mode=!positions?0:((k<=0||k>=blocks||blocks<Small369)?1:3);
 *s={uint64_t(positions),uint64_t(blocks),uint64_t(k>0?k:0),0,0,k,mode,0,0,{0,0}};
}
#include "literal_fallback_v369.inc"
__global__ void histogram369(const float*scores,int block,Workspace369 w,int shift){
 if(w.state->mode!=3||w.state->status)return;
 __shared__ uint32_t bins[256];bins[threadIdx.x]=0;__syncthreads();
 const int tid=threadIdx.x,lane=tid&31;
 const int64_t begin=int64_t(blockIdx.x)*Part369;
 const Values<true>values{scores,int64_t(w.state->positions),block};
 const uint32_t prefix=w.state->prefix,mask=w.state->prefix_mask;
 #pragma unroll
 for(int part=0;part<4;part++){
  const int64_t j=begin+part*Threads369+tid;
  const uint32_t key=j<int64_t(w.state->blocks)?key_of<false>(values.get(j)):0;
  const bool valid=j<int64_t(w.state->blocks)&&(key&mask)==prefix;
  const unsigned active=__ballot_sync(kFullWarp,valid);
  if(valid){const uint32_t bin=(key>>shift)&255u;const unsigned peers=__match_any_sync(active,bin);if(lane==__ffs(peers)-1)atomicAdd(bins+bin,uint32_t(__popc(peers)));}
 }
 __syncthreads();w.hist[size_t(blockIdx.x)*256+tid]=bins[tid];
}
__global__ void choose369(Workspace369 w,int shift){
 if(w.state->mode!=3||w.state->status)return;
 __shared__ Count bins[256];Count sum=0;
 for(int part=0;part<w.parts;part++)sum+=w.hist[size_t(part)*256+threadIdx.x];
 bins[threadIdx.x]=sum;__syncthreads();
 if(!threadIdx.x){Count remaining=w.state->remaining;bool found=false;
  for(int bin=255;bin>=0;bin--){if(bins[bin]>=remaining){w.state->prefix|=uint32_t(bin)<<shift;w.state->remaining=remaining;w.state->prefix_mask|=255u<<shift;found=true;break;}remaining-=bins[bin];}
  if(!found)w.state->status=1;
 }
}
__global__ void prefix_ties369(Workspace369 w){
 if(w.state->mode!=3||w.state->status||threadIdx.x)return;
 // Final shift0 histograms were filtered by the exact firstthree keybytes.
 // Their chosenlow-byte bin IS the full-key equality count; no extra score or
 // blockMAX pass is needed. Same-stream ordering follows the final choose.
 uint32_t prefix=0;for(int part=0;part<w.parts;part++){uint32_t count=w.hist[size_t(part)*256+(w.state->prefix&255u)];w.ties[part]=count;w.equal_before[part]=prefix;prefix+=count;}
 if(uint64_t(prefix)<w.state->remaining)w.state->status=2;
}
__global__ void emit369(const float*scores,int block,uint8_t*cand,Workspace369 w){
 if(w.state->mode!=3||w.state->status)return;
 __shared__ uint32_t equals[32];const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const uint32_t lane_mask=(uint32_t(1)<<lane)-1;uint32_t keys[4];unsigned equal_lanes[4];
 const int64_t base=int64_t(blockIdx.x)*Part369;
 const Values<true>values{scores,int64_t(w.state->positions),block};
 // Literal original four consecutive256parts and32warp ordered integer scan.
 #pragma unroll
 for(int part=0;part<4;part++){
  const int64_t j=base+part*Threads369+tid;
  keys[part]=j<int64_t(w.state->blocks)?key_of<false>(values.get(j)):0;
  equal_lanes[part]=__ballot_sync(kFullWarp,j<int64_t(w.state->blocks)&&keys[part]==w.state->prefix);
  if(!lane)equals[part*8+warp]=__popc(equal_lanes[part]);
 }
 __syncthreads();
 if(!warp){uint32_t own=equals[lane],sum=own;for(int delta=1;delta<32;delta<<=1){uint32_t previous=__shfl_up_sync(kFullWarp,sum,delta);if(lane>=delta)sum+=previous;}equals[lane]=sum-own;}
 __syncthreads();
 #pragma unroll
 for(int part=0;part<4;part++){
  const int64_t j=base+part*Threads369+tid;
  if(j<int64_t(w.state->blocks)){
   const uint64_t prior=uint64_t(w.equal_before[blockIdx.x])+equals[part*8+warp]+__popc(equal_lanes[part]&lane_mask);
   const bool selected=keys[part]>w.state->prefix||(keys[part]==w.state->prefix&&prior<w.state->remaining);
   const uint8_t keep=selected&&keys[part]!=key_of<false>(-CUDART_INF_F);
   const int64_t begin=j*block;for(int d=0;d<block&&begin+d<int64_t(w.state->positions);d++)cand[begin+d]=keep;
  }
 }
}
bool span369(const void*p,size_t n,uintptr_t&lo,uintptr_t&hi){lo=reinterpret_cast<uintptr_t>(p);if(!p||n>std::numeric_limits<uintptr_t>::max()-lo)return false;hi=lo+n;return true;}
bool disjoint369(const void*a,size_t an,const void*b,size_t bn){uintptr_t al,ah,bl,bh;return span369(a,an,al,ah)&&span369(b,bn,bl,bh)&&!(al<bh&&bl<ah);}
int submit369(const float*scores,const int*pos,int ratio,int64_t cap,int k,int block,uint8_t*mask,void*scratch,size_t bytes,cudaStream_t stream,int enable,bool dynamic){
 if(!scores||!mask||!stream||cap<0||cap>Capacity369||ratio<1||ratio>64||block<1||block>64||k<0||k>2048||(enable!=0&&enable!=1)||(dynamic&&!pos)||reinterpret_cast<uintptr_t>(scores)%4||(pos&&reinterpret_cast<uintptr_t>(pos)%4))return 1;
 if(!cap)return 0;
 const size_t need=bytes369(cap,block);
 if(!disjoint369(scores,size_t(cap)*4,mask,size_t(cap)))return 1;
 if(pos&&(!disjoint369(pos,4,scores,size_t(cap)*4)||!disjoint369(pos,4,mask,size_t(cap))))return 1;
 if(enable&&(!scratch||reinterpret_cast<uintptr_t>(scratch)%256||bytes<need||!disjoint369(scratch,need,scores,size_t(cap)*4)||!disjoint369(scratch,need,mask,size_t(cap))||(pos&&!disjoint369(scratch,need,pos,4))))return 1;
 if(!enable){if(dynamic)candidate_blocks_device(scores,pos,ratio,cap,k,block,mask,stream);else candidate_blocks(scores,cap,k,block,mask,stream);return 0;}
 const int64_t nblocks=(cap-1)/block+1;
 // Host-known tiny/full/zero static paths are EXACT original publicbody.
 if(!dynamic&&(nblocks<Small369||k==0||k>=nblocks)){candidate_blocks(scores,cap,k,block,mask,stream);return 0;}
 auto w=split369(scratch,cap,block);
 initialize369<<<1,32,0,stream>>>(w.state,pos,ratio,cap,k,block);
 if(dynamic)fallback369<true><<<1,kThreads,0,stream>>>({scores,cap,block},0,k,0,nullptr,mask,pos,ratio,false,w.state);
 for(int shift=24;shift>=0;shift-=8){histogram369<<<w.parts,Threads369,0,stream>>>(scores,block,w,shift);choose369<<<1,Threads369,0,stream>>>(w,shift);}
 prefix_ties369<<<1,32,0,stream>>>(w);emit369<<<w.parts,Threads369,0,stream>>>(scores,block,mask,w);
 return cudaGetLastError()==cudaSuccess?0:2;
}
}
}
#define DS369_EXPORT __attribute__((visibility("default")))
extern "C" DS369_EXPORT size_t ds369_candidate_workspace_bytes(int64_t cap,int block){return ds369::selected::bytes369(cap,block);}
extern "C" DS369_EXPORT int ds369_candidate_static(const float*s,int64_t n,int k,int block,uint8_t*mask,void*w,size_t bytes,void*stream,int enable){try{return ds369::selected::submit369(s,nullptr,1,n,k,block,mask,w,bytes,static_cast<cudaStream_t>(stream),enable,false);}catch(...){return 2;}}
extern "C" DS369_EXPORT int ds369_candidate_dynamic(const float*s,const int32_t*pos,int ratio,int64_t cap,int k,int block,uint8_t*mask,void*w,size_t bytes,void*stream,int enable){try{return ds369::selected::submit369(s,pos,ratio,cap,k,block,mask,w,bytes,static_cast<cudaStream_t>(stream),enable,true);}catch(...){return 2;}}
extern "C" DS369_EXPORT int ds369_candidate_resource_info(uint64_t*out,int words){
 if(!out||words!=32)return 1;
 try{
  using namespace ds369::selected;cudaFuncAttributes h{},c{},e{},f{};int hb=0,eb=0;cudaDeviceProp prop{};int device=-1;
  if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&prop,device)!=cudaSuccess||cudaFuncGetAttributes(&h,histogram369)!=cudaSuccess||cudaFuncGetAttributes(&c,choose369)!=cudaSuccess||cudaFuncGetAttributes(&e,emit369)!=cudaSuccess||cudaFuncGetAttributes(&f,fallback369<true>)!=cudaSuccess||cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hb,histogram369,Threads369,0)!=cudaSuccess||cudaOccupancyMaxActiveBlocksPerMultiprocessor(&eb,emit369,Threads369,0)!=cudaSuccess)return 2;
  const uint64_t v[32]={369,Threads369,Part369,Capacity369,512,bytes369(Capacity369,8),bytes369(Capacity369,1),uint64_t(device),uint64_t(prop.major),uint64_t(prop.minor),uint64_t(h.numRegs),h.sharedSizeBytes,h.localSizeBytes,uint64_t(hb),uint64_t(c.numRegs),c.sharedSizeBytes,c.localSizeBytes,uint64_t(e.numRegs),e.sharedSizeBytes,e.localSizeBytes,uint64_t(eb),uint64_t(f.numRegs),f.sharedSizeBytes,f.localSizeBytes,uint64_t(prop.sharedMemPerMultiprocessor),uint64_t(prop.sharedMemPerBlock),uint64_t(prop.maxThreadsPerMultiProcessor),0,0,0,0,0};std::copy(v,v+32,out);return 0;
 }catch(...){return 2;}
}
