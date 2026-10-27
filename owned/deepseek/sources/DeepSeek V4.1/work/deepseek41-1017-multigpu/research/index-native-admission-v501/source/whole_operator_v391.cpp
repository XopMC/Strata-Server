#include "index_pair_admission_v501.hpp"
#include "../../whole-down-metadata-transport-v391/source/whole_operator_v303.hpp"
#include "../../whole-down-metadata-transport-v391/source/plan_v303.hpp"
#include "../../whole-down-metadata-transport-v391/source/transport_v391.h"
#include "../../gpu-gu-byteperm-v13/batch_v5.hpp"
#include "../../exact-down-component-v53/down_v53.hpp"
#include "../../bf16-gate-libmvec-table-v276/source/hidden_gpu_v276.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <exception>
namespace ds303 {
namespace {
namespace G=ds1017_gu_batch_v5;namespace D=ds1017_exact_down_v53;
static_assert(sizeof(G::Request)==65920&&sizeof(D::Request)==176896);
constexpr size_t DownPrefix391=offsetof(D::Request,codes);
static_assert(DownPrefix391==640&&offsetof(D::Request,scales)==83584&&offsetof(D::Request,sums)==93952);
static_assert(D::MaxViews==MaxRoutes&&D::K==Hidden&&D::Groups==Hidden/32);
std::mutex transport_mutex391;
std::array<uint64_t,16>transport_stats391={391,16};
bool mode391(){const char*p=std::getenv("DS1017_WHOLE_DOWN_METADATA_V391");if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("391 mode must be absent/0/1 before Operator create");return p&&!std::strcmp(p,"1");}
void created391(bool on){std::lock_guard<std::mutex>l(transport_mutex391);transport_stats391[2]++;transport_stats391[3]++;transport_stats391[on?5:4]++;}
void closed391(){std::lock_guard<std::mutex>l(transport_mutex391);if(!transport_stats391[2])std::terminate();transport_stats391[2]--;}
void recorded391(bool on,unsigned chunks,bool failed){
 std::lock_guard<std::mutex>l(transport_mutex391);if(failed){transport_stats391[14]++;return;}
 const uint64_t full=uint64_t(chunks)*sizeof(D::Request),actual=uint64_t(chunks)*(on?DownPrefix391:sizeof(D::Request));
 transport_stats391[6]++;transport_stats391[7]+=on;transport_stats391[on?9:8]+=chunks;
 transport_stats391[10]+=actual;transport_stats391[11]+=full;transport_stats391[12]+=full-actual;transport_stats391[13]+=actual;
}
int read_transport391(uint64_t*out,int words)noexcept{if(!out||words!=16)return 1;try{std::lock_guard<std::mutex>l(transport_mutex391);std::copy(transport_stats391.begin(),transport_stats391.end(),out);return 0;}catch(...){return 9;}}
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
uint64_t tid(){return uint64_t(syscall(SYS_gettid));}
void current(int expected){int d=-1;ck(cudaGetDevice(&d));need(d==expected,"303 caller must select original admitted rank");}
void pointer_table(const void*p,size_t count,int rank){
 need(p&&count&&count<=15360,"303 immutable resident table metadata");cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p));need(a.type==cudaMemoryTypeDevice&&a.device==rank,"303 original device table only");
 CUdeviceptr base=0;size_t size=0;need(cuMemGetAddressRange(&base,&size,reinterpret_cast<CUdeviceptr>(p))==CUDA_SUCCESS,"303 table allocation range");uintptr_t at=uintptr_t(p);need(at>=base&&at-base<=size&&count*sizeof(void*)<=size-(at-base),"303 real table extent");
}
void legal_view(const Ds41A8ViewV118&v){
 need(v.columns==Dim&&v.codes&&v.scales&&v.sums,"303 original A8 view geometry");
 for(int g=0;g<Dim/32;g++){
  __m256i c=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(v.codes+g*32));
  need(!_mm256_movemask_epi8(_mm256_cmpeq_epi8(c,_mm256_set1_epi8(char(-128)))),"303 A8 code -128 refused");
  __m256i s=_mm256_dpbusd_epi32(_mm256_setzero_si256(),_mm256_set1_epi8(1),c);
  need(_mm256_movemask_epi8(_mm256_cmpeq_epi32(s,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(v.sums+g*8))))==-1,"303 original per4 sums differ");
  need(std::isfinite(v.scales[g])&&v.scales[g]>=0.f,"303 finite nonnegative A8 scale");
 }
}
}
struct Operator::Impl {
 Config c;const bool metadata391=mode391();bool registered391=false;mutable std::mutex mutex;std::array<uint64_t,StatCount>s{};
 std::vector<std::pair<void*,size_t>>device,pinned;
 G::Request *hg=nullptr,*dg=nullptr;D::Request *hd=nullptr,*dd=nullptr;Plan *hp=nullptr,*dp=nullptr;
 uint32_t*exp=nullptr;uint8_t*domain=nullptr;float*gu=nullptr,*down=nullptr,*gate=nullptr,*up=nullptr,*hf=nullptr,*hs=nullptr;
 uint16_t*hb=nullptr,*out=nullptr,*host_out=nullptr;int8_t*hc=nullptr;int32_t*hz=nullptr,*sentinel=nullptr;int*error=nullptr,*host_error=nullptr;
 cudaEvent_t done=nullptr;unsigned chunks=0;bool shutdown_failed=false;
 explicit Impl(Config config):c(std::move(config)){s[Version]=303;s[Device]=c.logical_device;s[Slots]=c.resident_slots;s[GuVersion]=13;s[HiddenVersion]=276;s[DownVersion]=53;}
 template<class T>T*alloc(size_t n){
  need(n<=Operator::workspace_device_bytes()/sizeof(T)&&s[LiveDeviceBytes]<=Operator::workspace_device_bytes()-n*sizeof(T),"303 exact device workspace budget");T*p=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));
  try{device.push_back({p,n*sizeof(T)});}catch(...){cudaFree(p);throw;}s[LiveDeviceBytes]+=n*sizeof(T);s[PeakDeviceBytes]=std::max(s[PeakDeviceBytes],s[LiveDeviceBytes]);return p;
 }
 template<class T>T*pin(size_t n){
  need(n<=Operator::workspace_pinned_bytes()/sizeof(T)&&s[LivePinnedBytes]<=Operator::workspace_pinned_bytes()-n*sizeof(T),"303 exact pinned workspace budget");T*p=nullptr;ck(cudaHostAlloc(reinterpret_cast<void**>(&p),n*sizeof(T),0));
  try{pinned.push_back({p,n*sizeof(T)});}catch(...){cudaFreeHost(p);throw;}s[LivePinnedBytes]+=n*sizeof(T);s[PeakPinnedBytes]=std::max(s[PeakPinnedBytes],s[LivePinnedBytes]);return p;
 }
 void init(){
  ds501::require_environment();
  need(c.logical_device==0||c.logical_device==1,"303 admitted rank0/1 only");need(c.stream&&c.device_owner&&c.cpu_table,"303 caller stream/real resources/table required");current(c.logical_device);
  ds501::validate_device(c.logical_device);
  unsigned flags=0;int rank=-1;ck(cudaStreamGetFlags(c.stream,&flags));ck(cudaStreamGetDevice(c.stream,&rank));need((flags&cudaStreamNonBlocking)&&rank==c.logical_device,"303 ONLY explicit NONBLOCKING rank stream");cudaStreamCaptureStatus capture{};ck(cudaStreamIsCapturing(c.stream,&capture));need(capture==cudaStreamCaptureStatusNone,"303 ordinary idle create");
  pointer_table(c.gu_table,c.resident_slots,c.logical_device);pointer_table(c.down_table,c.resident_slots,c.logical_device);s[TablesValidated]=1;need(!ds276_exp_table_validate(c.cpu_table),"303 actual same-process qualified table276");
  need(G::launch(nullptr,nullptr,0,0,nullptr,nullptr,nullptr)==cudaErrorInvalidValue,"303 literal V13 policy must already be enabled");
  size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total));need(free>Operator::workspace_device_bytes()+(512ull<<20),"303 full workspace plus512MiB reserve");
  device.reserve(24);pinned.reserve(8); // no host ledger growth after CUDA allocation
  exp=alloc<uint32_t>(65536);domain=alloc<uint8_t>(65536);hg=pin<G::Request>(Chunks);hd=pin<D::Request>(Chunks);hp=pin<Plan>(1);host_out=pin<uint16_t>(MaxRows*TopK*Dim);host_error=pin<int>(1);
  dg=alloc<G::Request>(Chunks);dd=alloc<D::Request>(Chunks);dp=alloc<Plan>(1);gu=alloc<float>(2*MaxRoutes*Hidden);down=alloc<float>(MaxRoutes*Dim);gate=alloc<float>(MaxRoutes*Hidden);up=alloc<float>(MaxRoutes*Hidden);hf=alloc<float>(MaxRoutes*Hidden);hb=alloc<uint16_t>(MaxRoutes*Hidden);hc=alloc<int8_t>(MaxRoutes*Hidden);hs=alloc<float>(MaxRoutes*(Hidden/32));hz=alloc<int32_t>(MaxRoutes*(Hidden/4));out=alloc<uint16_t>(MaxRows*TopK*Dim);sentinel=alloc<int32_t>(1);error=alloc<int>(1);
  need(s[LiveDeviceBytes]==Operator::workspace_device_bytes()&&s[LivePinnedBytes]==Operator::workspace_pinned_bytes(),"303 exact requested workspace ledger");
  ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));ck(cudaMemcpyAsync(exp,c.cpu_table->values,65536*4,cudaMemcpyHostToDevice,c.stream));ck(cudaMemcpyAsync(domain,c.cpu_table->domain,65536,cudaMemcpyHostToDevice,c.stream));ck(cudaStreamSynchronize(c.stream));c.cpu_table=nullptr; // caller CPU arrays no longer borrowed
  created391(metadata391);registered391=true;
 }
 void snapshot(const Command&cmd){
  need(cmd.epoch>s[LastEpoch]&&cmd.rows>=1&&cmd.rows<=MaxRows&&cmd.groups>=1&&cmd.groups<=MaxGroups,"303 epoch/rows/groups");
  std::array<bool,MaxRoutes>routes{};std::array<bool,6>queries{};unsigned pairs=0;
  for(unsigned g=0;g<cmd.groups;g++){
   const auto&v=cmd.group[g];need(!v.reserved&&v.slot<c.resident_slots&&v.expert<384&&v.count>=1&&v.count<=6&&(g==0||v.expert>cmd.group[g-1].expert),"303 group metadata/count/order");
   for(unsigned a=0;a<g;a++)need(v.slot!=cmd.group[a].slot,"303 duplicate physical slot");
   need(pairs<=MaxRoutes-v.count,"303 total routes<=36");pairs+=v.count;
   for(unsigned t=0;t<6;t++){
    if(t>=v.count){need(!v.query[t]&&!v.route_slot[t]&&v.alpha[t]==0.f,"303 inactive group metadata must zero");continue;}
    unsigned q=v.query[t],r=v.route_slot[t];need(q<cmd.rows&&r<cmd.rows*TopK&&r/TopK==q&&!routes[r]&&(t==0||q>v.query[t-1])&&std::isfinite(v.alpha[t])&&v.alpha[t]>0.f,"303 original query/route/positivealpha conservation");routes[r]=true;queries[q]=true;
   }
  }
  for(unsigned q=0;q<6;q++){
   const auto&v=cmd.view[q];if(queries[q])legal_view(v);else if(q>=cmd.rows)need(!v.codes&&!v.scales&&!v.sums&&!v.columns,"303 outside-row view must zero");
  }
  chunks=(cmd.groups+11)/12;std::memset(hg,0,chunks*sizeof(*hg));
  if(metadata391){for(unsigned chunk=0;chunk<chunks;chunk++)std::memset(hd+chunk,0,DownPrefix391);}
  else std::memset(hd,0,chunks*sizeof(*hd));
  *hp=Plan{};hp->rows=cmd.rows;hp->routes=pairs;
  unsigned packed=0;
  for(unsigned g=0;g<cmd.groups;g++){
   unsigned chunk=g/12,local=g%12;const auto&v=cmd.group[g];auto&a=hg[chunk].entry[local];auto&b=hd[chunk].entry[local];a.weight=b.weight=v.slot;a.count=b.count=v.count;a.output_words=hp->gu_words;b.output_words=hp->down_words;
   for(unsigned t=0;t<v.count;t++,packed++){a.query[t]=v.query[t];b.view[t]=packed;hp->gate[packed]=hp->gu_words+t*Hidden;hp->up[packed]=hp->gu_words+(v.count+t)*Hidden;hp->down[packed]=hp->down_words+t*Dim;hp->original_slot[packed]=v.route_slot[t];hp->alpha[packed]=v.alpha[t];}
   hp->gu_words+=2*v.count*Hidden;hp->down_words+=v.count*Dim;hg[chunk].experts++;hd[chunk].experts++;hg[chunk].pairs+=v.count;hd[chunk].pairs+=v.count;
  }
  need(packed==pairs&&hp->gu_words==pairs*2*Hidden&&hp->down_words==pairs*Dim,"303 request offsets conserved");
  for(unsigned chunk=0;chunk<chunks;chunk++){
   hg[chunk].generation=hd[chunk].generation=cmd.epoch;hg[chunk].output_words=hp->gu_words;hd[chunk].output_words=hp->down_words;
   for(unsigned q=0;q<cmd.rows;q++)if(queries[q]){const auto&v=cmd.view[q];std::copy_n(v.codes,Dim,hg[chunk].codes+q*Dim);std::copy_n(v.scales,Dim/32,hg[chunk].scales+q*(Dim/32));std::copy_n(v.sums,Dim/4,hg[chunk].sums+q*(Dim/4));}
  }
 }
 void enqueue(){
  ck(cudaMemcpyAsync(dp,hp,sizeof(*hp),cudaMemcpyHostToDevice,c.stream));ck(cudaMemcpyAsync(dg,hg,chunks*sizeof(*hg),cudaMemcpyHostToDevice,c.stream));
  if(metadata391){for(unsigned chunk=0;chunk<chunks;chunk++)ck(cudaMemcpyAsync(dd+chunk,hd+chunk,DownPrefix391,cudaMemcpyHostToDevice,c.stream));}
  else ck(cudaMemcpyAsync(dd,hd,chunks*sizeof(*hd),cudaMemcpyHostToDevice,c.stream));
  s[InputCopyBytes]+=sizeof(*hp)+chunks*(sizeof(*hg)+(metadata391?DownPrefix391:sizeof(*hd)));
  ck(cudaMemsetAsync(out,0,MaxRows*TopK*Dim*2,c.stream));ck(cudaMemsetAsync(error,0,4,c.stream));ck(cudaMemsetAsync(hc,0,hp->routes*Hidden,c.stream));ck(cudaMemsetAsync(hs,0,hp->routes*(Hidden/32)*4,c.stream));ck(cudaMemsetAsync(hz,0,hp->routes*(Hidden/4)*4,c.stream));
  for(unsigned chunk=0;chunk<chunks;chunk++){ck(G::launch(c.gu_table,dg+chunk,hg[chunk].experts,hp->gu_words,sentinel,gu,c.stream));s[GuChunks]++;}
  gather303(dp,gu,gate,up,c.stream);ck(cudaPeekAtLastError());ds276_launch_hidden(exp,domain,gate,up,reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(dp)+offsetof(Plan,alpha)),hp->routes,hb,hf,hc,hs,hz,error,c.stream);ck(cudaPeekAtLastError());
  for(unsigned chunk=0;chunk<chunks;chunk++){
   ck(cudaMemcpyAsync(dd[chunk].codes,hc,hp->routes*Hidden,cudaMemcpyDeviceToDevice,c.stream));ck(cudaMemcpyAsync(dd[chunk].scales,hs,hp->routes*(Hidden/32)*4,cudaMemcpyDeviceToDevice,c.stream));ck(cudaMemcpyAsync(dd[chunk].sums,hz,hp->routes*(Hidden/4)*4,cudaMemcpyDeviceToDevice,c.stream));
   ck(D::launch(c.down_table,dd+chunk,hd[chunk].experts,hp->down_words,down,c.stream));s[DownChunks]++;
  }
  scatter_bf16303(dp,down,out,error,c.stream);ck(cudaPeekAtLastError());
  size_t bytes=hp->rows*TopK*Dim*2;ck(cudaMemcpyAsync(host_out,out,bytes,cudaMemcpyDeviceToHost,c.stream));ck(cudaMemcpyAsync(host_error,error,4,cudaMemcpyDeviceToHost,c.stream));s[OutputCopyBytes]+=bytes+4;ck(cudaEventRecord(done,c.stream));
 }
 void owner(uint64_t epoch){need(s[Active]&&s[ActiveEpoch]==epoch&&s[IssuerTid]==tid(),"303 actual active ticket epoch/issuer");current(c.logical_device);}
 void begin(const Command&cmd){
  std::lock_guard<std::mutex>lock(mutex);need(!s[Closed]&&!s[Poisoned]&&!s[Active],"303 idle nonpoisoned owner");current(c.logical_device);
  // Snapshot may throw, but has no CUDA writes or retained borrowed pointers.
  snapshot(cmd);s[Active]=1;s[BorrowedView]=0;s[ActiveEpoch]=s[LastEpoch]=cmd.epoch;s[IssuerTid]=tid();s[Rows]=cmd.rows;
  try{enqueue();recorded391(metadata391,chunks,false);s[Begins]++;s[Groups]+=cmd.groups;s[Routes]+=hp->routes;}
  catch(...){recorded391(metadata391,chunks,true);s[Failures]++;s[Poisoned]=1;if(cudaStreamSynchronize(c.stream)!=cudaSuccess){s[Failures]++;shutdown_failed=true;}else s[Active]=s[BorrowedView]=s[ActiveEpoch]=s[IssuerTid]=0;throw;}
 }
 Borrowed finish(uint64_t epoch){
  std::lock_guard<std::mutex>lock(mutex);owner(epoch);need(!s[Poisoned]&&!s[BorrowedView],"303 finish exactly once");
  try{ck(cudaEventSynchronize(done));s[CompletedEvents]++;need(*host_error==0,"303 whole-chain numerical error, no outputs admitted");s[BorrowedView]=1;s[Finishes]++;return {host_out,uint64_t(hp->rows)*TopK*Dim,epoch,hp->rows};}
  catch(...){s[Failures]++;s[Poisoned]=1;throw;}
 }
 void release(uint64_t epoch){std::lock_guard<std::mutex>lock(mutex);owner(epoch);need(s[BorrowedView]&&!s[Poisoned],"303 release only successful immutable borrow");s[Active]=s[BorrowedView]=s[ActiveEpoch]=s[IssuerTid]=0;s[Releases]++;}
 void abort(uint64_t epoch){
  std::lock_guard<std::mutex>lock(mutex);owner(epoch);need(!shutdown_failed,"303 previous drain unproved, no repeated CUDA cleanup");cudaError_t e=cudaStreamSynchronize(c.stream);s[Aborts]++;if(e!=cudaSuccess){s[Poisoned]=1;s[Failures]++;shutdown_failed=true;ck(e);}s[Active]=s[BorrowedView]=s[ActiveEpoch]=s[IssuerTid]=0;
 }
 int close()noexcept{
  std::lock_guard<std::mutex>lock(mutex);if(shutdown_failed)return 1;if(s[Closed])return 0;
  // Never revoke a borrowed immutable view or destroy an outstanding ticket.
  // The owner must first release or abort under its global N1 ticket guard.
  if(s[Active]||s[BorrowedView]){s[Failures]++;return 2;}
  if(device.empty()&&pinned.empty()&&!done){s[Closed]=1;return 0;} // invalid pre-allocation config never selects a device
  int prior=-1;bool selected=false;int failed=0;
  if(cudaGetDevice(&prior)!=cudaSuccess||cudaSetDevice(c.logical_device)!=cudaSuccess){s[Failures]++;s[Poisoned]=1;shutdown_failed=true;return 1;}selected=true;
  if(c.stream&&cudaStreamSynchronize(c.stream)!=cudaSuccess){s[Failures]++;s[Poisoned]=1;shutdown_failed=true;cudaSetDevice(prior);return 1;} // preserve ALL buffers/token while drain unproved
  if(done&&cudaEventDestroy(done)!=cudaSuccess)failed++;done=nullptr;
  for(auto i=device.rbegin();i!=device.rend();++i){if(cudaFree(i->first)!=cudaSuccess)failed++;else s[LiveDeviceBytes]-=i->second;}device.clear();
  for(auto i=pinned.rbegin();i!=pinned.rend();++i){if(cudaFreeHost(i->first)!=cudaSuccess)failed++;else s[LivePinnedBytes]-=i->second;}pinned.clear();
  if(selected&&cudaSetDevice(prior)!=cudaSuccess)failed++;
  if(failed){s[Failures]+=failed;s[Poisoned]=1;shutdown_failed=true;return 1;}if(registered391){closed391();registered391=false;}s[Closed]=1;return 0;
 }
 ~Impl(){if(close()!=0)std::terminate();} // never silently drop bank token after failed/unproved close
};
Operator::Operator(std::unique_ptr<Impl>p):p_(std::move(p)){}
std::unique_ptr<Operator>Operator::create(Config c){auto p=std::make_unique<Impl>(std::move(c));p->init();return std::unique_ptr<Operator>(new Operator(std::move(p)));}
Operator::~Operator()=default;
void Operator::begin(const Command&c){p_->begin(c);}Borrowed Operator::finish(uint64_t e){return p_->finish(e);}void Operator::release(uint64_t e){p_->release(e);}void Operator::abort(uint64_t e){p_->abort(e);}int Operator::shutdown()noexcept{return p_->close();}
std::array<uint64_t,StatCount>Operator::stats()const noexcept{std::lock_guard<std::mutex>lock(p_->mutex);return p_->s;}
}

extern "C" int ds391_whole_down_transport_info(uint64_t*out,int words)noexcept{return ds303::read_transport391(out,words);}
