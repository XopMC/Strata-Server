#include "qsa_vmm.hpp"
#include "peer_parts.hpp"
#include "selector_v7_cuda.hpp"
#include "selector_v7.hpp"
#include "qsa_model.hpp"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <cstdio>
#include <stdexcept>
namespace qsa_vmm453 {namespace {
namespace L=swift_selector_v7_leaf;namespace P=qsa_peer_v441;
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void cu(CUresult e){if(e!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(e,&s);throw std::runtime_error(s?s:"VMM failure");}}
struct Device{int previous;explicit Device(int d){ck(cudaGetDevice(&previous));ck(cudaSetDevice(d));}~Device(){if(cudaSetDevice(previous)!=cudaSuccess)std::abort();}};
struct Pool{
 const void* state;int device;int64_t rows;size_t logical,gran,physical,lower;CUdeviceptr reservation{},va{};std::array<CUmemGenericAllocationHandle,2>h{};
 void access(CUdeviceptr ptr,size_t bytes){std::array<CUmemAccessDesc,2>a{};for(int d=0;d<2;++d){a[d].location.type=CU_MEM_LOCATION_TYPE_DEVICE;a[d].location.id=d;a[d].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;}cu(cuMemSetAccess(ptr,bytes,a.data(),2));}
 Pool(const void*s,int d,int64_t r):state(s),device(d),rows(r),logical(size_t(r)*512){Device dev(d);CUdevice id;cu(cuDeviceGet(&id,d));int support=0;cu(cuDeviceGetAttribute(&support,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,id));need(support,"V453 requires native VMM");CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=d;cu(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));physical=(logical+gran-1)/gran*gran;lower=(8*((r+15)/16)*512)/gran*gran;need(lower&&lower<physical,"V453 pooled physical split invalid");cu(cuMemAddressReserve(&reservation,physical+2*gran,gran,0,0));va=reservation+gran;for(int k=0;k<2;++k){size_t off=k?lower:0,n=k?physical-lower:lower;cu(cuMemCreate(&h[k],n,&prop,0));cu(cuMemMap(va+off,n,0,h[k],0));access(va+off,n);}}
 ~Pool(){Device d(device);for(int k=0;k<2;++k){cu(cuMemUnmap(va+(k?lower:0),k?physical-lower:lower));cu(cuMemRelease(h[k]));}cu(cuMemAddressFree(reservation,physical+2*gran));}
};
std::array<std::vector<std::unique_ptr<Pool>>,2>& pools(){static auto*p=new std::array<std::vector<std::unique_ptr<Pool>>,2>;return*p;}
bool foreign=false,ready=false;CUdeviceptr temporary=0;cudaStream_t migration{};uint64_t epoch=0;
void drain(){for(int d=0;d<2;++d){Device scope(d);ck(cudaDeviceSynchronize());}}
void exchange(){
 drain();need(pools()[0].size()==6&&pools()[1].size()==6,"V453 exact six QSA pools per owner required");Device home(0);
 size_t maxgran=0;for(int i=0;i<6;++i){auto&a=*pools()[0][i];auto&b=*pools()[1][i];need(a.gran==b.gran&&a.lower==b.lower&&a.logical==b.logical,"V453 unpaired physical spans");maxgran=std::max(maxgran,a.gran);}
 if(!temporary){ck(cudaStreamCreateWithFlags(&migration,cudaStreamNonBlocking));ck(cudaMalloc(reinterpret_cast<void**>(&temporary),maxgran));}
 for(int i=0;i<6;++i){auto&a=*pools()[0][i];auto&b=*pools()[1][i];for(size_t off=0;off<a.lower;off+=a.gran){cu(cuMemcpyDtoDAsync(temporary,a.va+off,a.gran,migration));cu(cuMemcpyDtoDAsync(a.va+off,b.va+off,a.gran,migration));cu(cuMemcpyDtoDAsync(b.va+off,temporary,a.gran,migration));}ck(cudaStreamSynchronize(migration));cu(cuMemUnmap(a.va,a.lower));cu(cuMemUnmap(b.va,b.lower));std::swap(a.h[0],b.h[0]);cu(cuMemMap(a.va,a.lower,0,a.h[0],0));cu(cuMemMap(b.va,b.lower,0,b.h[0],0));a.access(a.va,a.lower);b.access(b.va,b.lower);}
 foreign=!foreign;std::fprintf(stderr,"QSA453_PHYSICAL_EXCHANGE remote_lower=%d pools=12 duplicate_state_bytes=0 scratch_bytes=%zu charged_prompt_TTFT=1\n",int(foreign),maxgran);
}
struct Memory{int device;std::vector<void*>p;explicit Memory(int d):device(d){}
 template<class T>T*make(size_t n){Device d(device);void*q=nullptr;ck(cudaMalloc(&q,n*sizeof(T)));p.push_back(q);return static_cast<T*>(q);}
 ~Memory(){Device d(device);for(auto q:p)if(cudaFree(q)!=cudaSuccess)std::abort();}};
L::Workspace workspace(Memory&m){return{m.make<int32_t>(8*L::Parts*L::Keep),m.make<int32_t>(8*L::Parts),m.make<int32_t>(8*L::Packed),m.make<int32_t>(32),m.make<int32_t>(8*2051),m.make<float>(8*L::Packed)};}
struct Owner{
 int device,peer,maxT;int64_t blocks,half;cudaStream_t stream,helper{};cudaEvent_t fork{},done{};Memory own,other;L::Workspace local{},remote{};float*q{},*dead{},*dummy{},*scores{};int32_t*steps{};uint32_t*packet{},*landed{};std::vector<Layer>layers;bool peer_bank=false;std::array<cudaGraphExec_t,9>inactive{},inactive_nr{};std::array<bool,9>inactive_launched{},inactive_launched_nr{};uint64_t peer_captures=0,local_captures=0;
 Owner(int d,int t,int64_t b,cudaStream_t s):device(d),peer(1-d),maxT(t),blocks(b),half(std::min(b,8*((b+15)/16))),stream(s),own(d),other(1-d){}
 ~Owner(){Device d(device);ck(cudaStreamSynchronize(stream));{Device p(peer);ck(cudaStreamSynchronize(helper));}for(auto x:inactive)if(x)ck(cudaGraphExecDestroy(x));for(auto x:inactive_nr)if(x)ck(cudaGraphExecDestroy(x));ck(cudaEventDestroy(fork));{Device p(peer);ck(cudaEventDestroy(done));ck(cudaStreamDestroy(helper));}}
};
std::map<const void*,std::unique_ptr<Owner>>& owners(){static auto*p=new std::map<const void*,std::unique_ptr<Owner>>;return*p;}
Owner&owner(const void*k){auto i=owners().find(k);need(i!=owners().end(),"V453 unregistered owner");return*i->second;}
void score(int n,const float*p,const float*d,const float*q,const int32_t*st,int64_t b,float*out,cudaStream_t s){auto fn=swift_score_kernel112(n);void*a[]={&p,&d,&q,&st,&n,&b,&out};ck(cudaLaunchKernel(fn,dim3(256),dim3(256),a,0,s));}
void finish(float*sc,const int32_t*st,int n,int64_t b,L::Workspace w,int32_t*ids,cudaStream_t s){ck(L::pack(sc,st,n,b,w,s));ck(L::final_select(w.packed_scores,w.packed_steps,n,L::Packed,2051,w.packed_ids,s));ck(L::translate(st,n,2051,w,ids,s));}
}
float* allocate_pool(const void* state,int64_t rows,int dim){int d=0,count=0;ck(cudaGetDevice(&d));ck(cudaGetDeviceCount(&count));need(count==2&&d>=0&&d<2&&dim==128&&!ready,"V453 requires fresh TwoGPU native main state");for(const auto&v:pools()[d])need(v->state!=state,"V453 duplicate state allocation");auto p=std::make_unique<Pool>(state,d,rows);float*data=reinterpret_cast<float*>(p->va);std::fprintf(stderr,"QSA453_POOL device=%d ordinal=%zu logical_bytes=%zu physical_bytes=%zu pad_bytes=%zu stable_VA=1\n",d,pools()[d].size(),p->logical,p->physical,p->physical-p->logical);pools()[d].push_back(std::move(p));return data;}
bool before_prompt(std::string&e)noexcept{try{drain();ready=false;if(foreign)exchange();return true;}catch(const std::exception&x){e=x.what();return false;}}
bool activate_request(int64_t prefix,std::string&e)noexcept{try{drain();need(!ready&&!foreign&&owners().size()==2,"V453 request physical lifecycle invalid");auto&o=*owners().begin()->second;const bool want=o.blocks>131074&&prefix>=o.half*4; if(want)exchange();++epoch;ready=true;std::fprintf(stderr,"QSA453_REQUEST epoch=%llu actual_prefix=%lld capacity_blocks=%lld mode=%s current_register_local_not_legacy=1\n",(unsigned long long)epoch,(long long)prefix,(long long)o.blocks,foreign?"native_peer_parts":"modern_local");return true;}catch(const std::exception&x){e=x.what();return false;}}
bool prepare(const void*k,int d,int maxT,int64_t b,cudaStream_t s,const std::vector<Layer>&layers,std::string&e)noexcept{try{Device device(d);need(!owners().count(k)&&d>=0&&d<2&&layers.size()==6&&maxT<=8,"V453 target owner geometry");auto o=std::make_unique<Owner>(d,maxT,b,s);o->layers=layers;for(auto l:layers){auto found=std::find_if(pools()[d].begin(),pools()[d].end(),[&](auto&p){return reinterpret_cast<float*>(p->va)==l.pooled;});need(found!=pools()[d].end()&&(*found)->rows==b&&l.cells/4+2==b,"V453 qsa state does not bind owned VMM allocation");}o->local=workspace(o->own);o->landed=o->own.make<uint32_t>(8*P::PacketWords);o->remote=workspace(o->other);o->q=o->other.make<float>(8*512);o->dead=o->other.make<float>(128);o->dummy=o->other.make<float>(128);o->steps=o->other.make<int32_t>(32);o->scores=o->other.make<float>(8*b);o->packet=o->other.make<uint32_t>(8*P::PacketWords);ck(cudaEventCreateWithFlags(&o->fork,cudaEventDisableTiming));{Device peer(o->peer);ck(cudaStreamCreateWithFlags(&o->helper,cudaStreamNonBlocking));ck(cudaEventCreateWithFlags(&o->done,cudaEventDisableTiming));}owners().emplace(k,std::move(o));return true;}catch(const std::exception&x){e=x.what();return false;}}
bool select_graph_bank(const void*k,cudaGraphExec_t*a,cudaGraphExec_t*b,bool*la,bool*lb,int count,std::string&e)noexcept{try{auto&o=owner(k);need(ready&&count==9,"V453 graph capture before physical request admission");if(o.peer_bank!=foreign){Device d(o.device);ck(cudaStreamSynchronize(o.stream));for(int i=0;i<count;++i){std::swap(a[i],o.inactive[i]);std::swap(b[i],o.inactive_nr[i]);std::swap(la[i],o.inactive_launched[i]);std::swap(lb[i],o.inactive_launched_nr[i]);}o.peer_bank=foreign;}return true;}catch(const std::exception&x){e=x.what();return false;}}
void enqueue(const void*k,int layer,int n,int tb,const float*p,const float*d,const float*q,const int32_t*st,float*sc,int32_t*ids,const strata::kernels::QsaShapes&s,cudaStream_t stream){auto&o=owner(k);need(ready&&tb==0&&n>0&&n<=o.maxT&&s.idx_dim==128&&s.idx_n_head==4&&s.idx_block==4&&s.idx_top_k==2048,"V453 original complete QSA shape");auto it=std::find_if(o.layers.begin(),o.layers.end(),[&](auto&l){return l.global_layer==layer&&l.pooled==p&&l.dead==d;});need(it!=o.layers.end(),"V453 changed layer binding");if(!foreign){score(n,p,d,q,st,o.blocks,sc,stream);swift_selector_v7::topk(sc,st,n,o.blocks,2051,s,ids,stream,nullptr,uint64_t(0));++o.local_captures;return;}ck(cudaEventRecord(o.fork,stream));{Device peer(o.peer);ck(cudaStreamWaitEvent(o.helper,o.fork));ck(P::inputs(q,d,st,d,o.q,o.dead,o.steps,o.dummy,n,o.helper));ck(P::score(n,p,o.dead,o.q,o.steps,o.blocks,o.scores,0,o.half,o.helper));ck(P::chunks(o.scores,o.steps,n,o.blocks,o.remote,0,8,o.helper));ck(P::gather(o.scores,o.blocks,o.remote,o.packet,n,o.helper));ck(qsa_model_v449::return_packet(o.packet,o.landed,n,P::PacketWords,o.helper));ck(cudaEventRecord(o.done,o.helper));}ck(P::score(n,p,d,q,st,o.blocks,sc,o.half,o.blocks,stream));ck(P::chunks(sc,st,n,o.blocks,o.local,8,8,stream));ck(cudaStreamWaitEvent(stream,o.done));ck(P::publish(sc,o.blocks,o.local,o.landed,n,stream));finish(sc,st,n,o.blocks,o.local,ids,stream);++o.peer_captures;}
void release(const void*k)noexcept{try{auto it=owners().find(k);if(it==owners().end())return;std::fprintf(stderr,"QSA453_RELEASE device=%d local_layer_captures=%llu peer_layer_captures=%llu inactive_graph_bank_retired=1\n",it->second->device,(unsigned long long)it->second->local_captures,(unsigned long long)it->second->peer_captures);owners().erase(it);}catch(...){std::abort();}}
Lifetime::~Lifetime(){try{if(pools()[0].empty()&&pools()[1].empty())return;need(owners().empty(),"V453 live captured parent during pool destruction");drain();pools()[0].clear();pools()[1].clear();if(temporary){Device d(0);ck(cudaFree(reinterpret_cast<void*>(temporary)));ck(cudaStreamDestroy(migration));}std::fprintf(stderr,"QSA453_POOLS_RELEASED captured_owners=0 physical_allocations=0\n");}catch(...){std::abort();}}
}
