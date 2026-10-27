#include "index_pair_admission_v501.hpp"
#include "../../whole-fixed-plan-owner-v469/source/resources_v469.hpp"
#include "../../pair-packed-inverse-v141-r2/source/pair_inverse_v141.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <immintrin.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <numeric>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
namespace ds469 {
namespace {
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device {int old=-1;Device(){ck(cudaGetDevice(&old));ck(cudaSetDevice(1));}~Device(){if(cudaSetDevice(old)!=cudaSuccess)std::terminate();}};
uint64_t ns(){timespec t{};need(!clock_gettime(CLOCK_MONOTONIC_RAW,&t),"469 monotonic clock");return uint64_t(t.tv_sec)*1000000000+uint64_t(t.tv_nsec);}
std::string name(const Ds143PairDescriptor&d){return "layers."+std::to_string(d.layer)+".ffn.experts."+std::to_string(d.expert)+(d.role==0?".w1":d.role==1?".w3":".w2");}
void metadata(const Triple&t,uint64_t gen,uint64_t epoch,unsigned id,const strata::ds41::BorrowedWeights&w){
 for(unsigned r=0;r<3;r++){
  const auto&d=t[r];need(d.struct_bytes==sizeof(d)&&d.layout_tag==139&&!d.bits&&d.data&&d.bytes==Resources::Matrix&&d.role==int(r)&&d.rows==(r==2?5120:2304)&&d.columns==(r==2?2304:5120)&&d.owner_generation==gen&&d.bank_format_epoch==epoch&&!d.reserved&&d.layer==t[0].layer&&d.expert==int(id%384)&&d.layer==int(id/384),"469 typed139 triple binding");
  const auto actual=strata::ds41::pair145::projection(w,name(d),d.rows,d.columns);
  need(actual.data==d.data&&actual.bytes==d.bytes&&actual.layout_tag==139&&actual.owner_generation==gen&&actual.bank_format_epoch==epoch&&actual.layer==d.layer&&actual.expert==d.expert&&actual.role==d.role,"469 actual retained registry identity");
 }
}
void footer(const Triple&t){
 const __m512i maximum=_mm512_set1_epi16(254);
 for(const auto&d:t){
  const auto*p=static_cast<const uint8_t*>(d.data);
  for(size_t b=0;b<Resources::Matrix;b+=576){
   // Exactly the original32 unsigned16-bit E8 codes; no value is skipped.
   const __m512i values=_mm512_loadu_si512(p+b+512);
   need(_mm512_cmp_epu16_mask(values,maximum,_MM_CMPINT_GT)==0,
        "469 E8 footer out of range before writes");
  }
 }
}
}
Resources::Resources(std::shared_ptr<const strata::ds41::BorrowedWeights>w,std::vector<Triple>catalog,std::vector<unsigned>ranked,const Ds276ExpTable&table,bool mode):weights_(std::move(w)),catalog_(std::move(catalog)),prompt_rank_(std::move(ranked)),mode_(mode){
 need(weights_&&catalog_.size()==15360&&prompt_rank_.size()==15360,"469 complete typed catalog/ranking");
 map_.fill(-1);slot_id_.fill(-1);spare_.fill(true);generation_=catalog_[0][0].owner_generation;format_=catalog_[0][0].bank_format_epoch;need(generation_&&format_,"469 generation/format");
 const auto binding=strata::ds41::pair145::info(*weights_);need(binding[0]==145&&binding[1]==139&&binding[2]==46080&&binding[3]==305764761600ull&&binding[4]==generation_&&binding[5]==format_&&binding[6]==1&&!binding[7],"469 same complete real139 registry");
 gate_id_.reserve(15360);std::array<bool,15360>seen{};
 for(unsigned id=0;id<15360;id++){
  metadata(catalog_[id],generation_,format_,id,*weights_); // Metadata ONLY; no full-bank footer walk.
  need(gate_id_.emplace(uintptr_t(catalog_[id][0].data),id).second,"469 unique real gate key");
  auto key=prompt_rank_[id];need(key<15360&&!seen[key],"469 exact full prompt rank permutation");seen[key]=true;
 }
 for(unsigned s=0;s<ActiveSlots;s++){map_[prompt_rank_[s]]=int(s);slot_id_[s]=int(prompt_rank_[s]);spare_[s]=false;}
 moves_.reserve(ActiveSlots);info_[Version]=469;info_[Words]=InfoWords;info_[Mode]=mode_;info_[LiveOwner]=1;info_[Generation]=generation_;info_[FormatEpoch]=format_;info_[Catalog]=15360;info_[Physical]=PhysicalSlots;info_[Active]=ActiveSlots;info_[Spares]=8;info_[DeclaredPeakBytes]=DeviceBytes;
 ds501::require_environment();Device device;ds501::validate_device(1);
 size_t f=0,total=0;ck(cudaMemGetInfo(&f,&total));need(DeviceBytes<=(20ull<<30)&&f>=DeviceBytes+(512ull<<20),"469 exact20GiB requested payload and actual512MiB reserve");std::array<const uint8_t*,1078>g{},dn{};
 try{
  ck(cudaStreamCreateWithFlags(&compute_,cudaStreamNonBlocking));ck(cudaMalloc(reinterpret_cast<void**>(&bank_),PhysicalSlots*TripleBytes));ck(cudaMalloc(reinterpret_cast<void**>(&gu_),PhysicalSlots*sizeof(void*)));ck(cudaMalloc(reinterpret_cast<void**>(&down_),PhysicalSlots*sizeof(void*)));
  for(unsigned s=0;s<PhysicalSlots;s++){g[s]=bank_+s*TripleBytes;dn[s]=g[s]+2*Matrix;}
  ck(cudaMemcpyAsync(gu_,g.data(),sizeof(g),cudaMemcpyHostToDevice,compute_));ck(cudaMemcpyAsync(down_,dn.data(),sizeof(dn),cudaMemcpyHostToDevice,compute_));ck(cudaStreamSynchronize(compute_));
  worker_=std::thread(&Resources::worker_main,this);{std::unique_lock l(mutex_);cv_.wait(l,[&]{return initialized_;});}need(!poisoned_.load(),"469 uploader startup failure");
  ds303::Config c{};c.logical_device=1;c.resident_slots=PhysicalSlots;c.gu_table=gu_;c.down_table=down_;c.cpu_table=&table;c.stream=compute_;c.device_owner=weights_;op_=ds303::Operator::create(std::move(c));
  // Exact original370 startup prime, epoch1. Real Task epoch n maps to n+1.
  std::array<int8_t,5120>zc{};std::array<float,160>zs{};zs.fill(1.f);std::array<int32_t,1280>zz{};ds303::Command prime{};prime.epoch=1;prime.rows=1;prime.groups=1;prime.view[0]={zc.data(),zs.data(),zz.data(),5120};prime.group[0].slot=0;prime.group[0].expert=prompt_rank_[0]%384;prime.group[0].count=1;prime.group[0].alpha[0]=1.f;
  op_->begin(prime);auto result=op_->finish(1);need(result.words==6*5120&&result.data,"469 actual391 startup prime");for(unsigned i=0;i<result.words;i++)need(!result.data[i],"469 startup zero exact");op_->release(1);info_[StartupJobs]=1;last_op_=op_->stats();ck(cudaMemGetInfo(&f,&total));free_after_workspace_=f;need(f>=(512ull<<20),"469 actual final reserve");
 }catch(...){try{if(op_&&op_->stats()[ds303::Active])op_->abort(1);close_impl();}catch(...){std::terminate();}throw;}
}
void Resources::check_idle()const{need(!closed_.load()&&!poisoned_.load()&&op_&&!active_epoch_.load(),"469 idle open owner");auto s=op_->stats();need(!s[ds303::Active]&&!s[ds303::BorrowedView]&&!s[ds303::Poisoned]&&!s[ds303::Closed],"469 actual391 idle/no borrowed view");}
unsigned Resources::id_for(int e,uintptr_t g,uintptr_t u,uintptr_t d)const{auto it=gate_id_.find(g);need(it!=gate_id_.end()&&e>=0&&e<384&&it->second%384==unsigned(e),"469 actual layer/expert gate witness");unsigned id=it->second;need(u==uintptr_t(catalog_[id][1].data)&&d==uintptr_t(catalog_[id][2].data),"469 exact triple role keys");return id;}
void Resources::build_plan(){
 const auto start=ns();timespec cpu{};need(!clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu),"469 planner clock");uint64_t cpustart=uint64_t(cpu.tv_sec)*1000000000+cpu.tv_nsec;
 std::array<unsigned,15360>observed{};size_t n=0;for(unsigned id=0;id<15360;id++)if(score_[id])observed[n++]=id;
 std::sort(observed.begin(),observed.begin()+n,[&](unsigned a,unsigned b){if(score_[a]!=score_[b])return score_[a]>score_[b];if(groups_[a]!=groups_[b])return groups_[a]>groups_[b];if(routes_[a]!=routes_[b])return routes_[a]>routes_[b];return a<b;});
 std::array<unsigned,1070>target{},victims{};std::array<bool,15360>wanted{};unsigned selected=0;
 for(size_t i=0;i<n&&selected<ActiveSlots;i++){target[selected++]=observed[i];wanted[observed[i]]=true;}
 for(auto id:prompt_rank_)if(selected<ActiveSlots&&!wanted[id]){target[selected++]=id;wanted[id]=true;}
 need(selected==ActiveSlots,"469 fixed target capacity");unsigned nv=0;for(unsigned s=0;s<PhysicalSlots;s++)if(slot_id_[s]>=0&&!wanted[unsigned(slot_id_[s])])victims[nv++]=unsigned(slot_id_[s]);std::sort(victims.begin(),victims.begin()+nv);
 unsigned at=0;for(auto id:target)if(map_[id]<0){need(at<nv,"469 target/victim conservation");moves_.push_back({id,victims[at++]});}need(at==nv,"469 complete fixed plan");
 need(!clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu),"469 planner clock end");uint64_t end=uint64_t(cpu.tv_sec)*1000000000+cpu.tv_nsec;
 std::lock_guard l(mutex_);plan_built_=true;info_[PlanBuilt]=1;info_[PlanMoves]=moves_.size();info_[QueuedMoves]=mode_?moves_.size():0;info_[PlannerNs]=ns()-start;info_[PlannerThreadNs]=end-cpustart;cv_.notify_all();
}
void Resources::worker_upload(unsigned j,unsigned credit){
 auto&t=catalog_[jobs_[j].candidate];auto at=ns();timespec cpu{};need(!clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu),"469 stage clock");uint64_t cstart=uint64_t(cpu.tv_sec)*1000000000+cpu.tv_nsec;
 metadata(t,generation_,format_,jobs_[j].candidate,*weights_);footer(t);auto valid=ns()-at;auto*pin=pins_+size_t(credit)*TripleBytes;at=ns();for(unsigned r=0;r<3;r++)std::memcpy(pin+r*Matrix,t[r].data,Matrix);auto copied=ns()-at;
 {std::lock_guard l(mutex_);info_[ValidationNs]+=valid;info_[CopyNs]+=copied;}
 for(unsigned r=0;r<3;r++){
  ck(cudaMemcpyAsync(scratch_,pin+r*Matrix,Matrix,cudaMemcpyHostToDevice,upload_));
  need(!ds141_pair32_to_output16(scratch_,Matrix,t[r].rows,t[r].columns,139,bank_+(jobs_[j].slot*3+r)*Matrix,Matrix,1,reinterpret_cast<uint64_t>(upload_)),"469 literal141 spare inverse");
  std::lock_guard l(mutex_);info_[H2DBytes]+=Matrix;info_[InverseReadBytes]+=Matrix;info_[InverseWriteBytes]+=Matrix;
 }
 ck(cudaEventRecord(jobs_[j].done,upload_));need(!clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu),"469 stage clock end");uint64_t cend=uint64_t(cpu.tv_sec)*1000000000+cpu.tv_nsec;
 {std::lock_guard l(mutex_);jobs_[j].state=State::Submitted;info_[StageThreadNs]+=cend-cstart;}
}
void Resources::worker_drain()noexcept{
 Device device;if(upload_&&cudaStreamSynchronize(upload_)!=cudaSuccess)std::terminate();
 std::lock_guard l(mutex_);info_[UploadDrained]=1;
 for(auto&j:jobs_)if(j.state!=State::Unused){j.state=State::Unused;info_[Cancelled]++;if(!info_[Pending])std::terminate();info_[Pending]--;}
 if(mode_&&plan_built_){info_[Cancelled]+=moves_.size()-next_move_;info_[QueuedMoves]=0;next_move_=unsigned(moves_.size());}
}
void Resources::worker_main()noexcept{
 try{
  cpu_set_t a;CPU_ZERO(&a);CPU_SET(253,&a);need(!sched_setaffinity(0,sizeof(a),&a)&&sched_getcpu()==253,"469 reserved uploader253");Device device;int low=0,high=0;ck(cudaDeviceGetStreamPriorityRange(&low,&high));ck(cudaStreamCreateWithPriority(&upload_,cudaStreamNonBlocking,low));
  ck(cudaMalloc(reinterpret_cast<void**>(&scratch_),Matrix));ck(cudaHostAlloc(reinterpret_cast<void**>(&pins_),PinnedBytes,0));for(auto&j:jobs_)ck(cudaEventCreateWithFlags(&j.done,cudaEventDisableTiming));
  std::array<int,2>last{-1,-1};
  for(unsigned s=0;s<ActiveSlots;s++){
   unsigned credit=s%2;if(last[credit]>=0)ck(cudaEventSynchronize(jobs_[unsigned(last[credit])].done));
   const auto&t=catalog_[unsigned(slot_id_[s])];footer(t);auto*pin=pins_+size_t(credit)*TripleBytes;for(unsigned r=0;r<3;r++)std::memcpy(pin+r*Matrix,t[r].data,Matrix);
   for(unsigned r=0;r<3;r++){ck(cudaMemcpyAsync(scratch_,pin+r*Matrix,Matrix,cudaMemcpyHostToDevice,upload_));need(!ds141_pair32_to_output16(scratch_,Matrix,t[r].rows,t[r].columns,139,bank_+(s*3+r)*Matrix,Matrix,1,reinterpret_cast<uint64_t>(upload_)),"469 startup literal141");}
   ck(cudaEventRecord(jobs_[credit].done,upload_));last[credit]=int(credit);
  }ck(cudaStreamSynchronize(upload_));last={-1,-1};
  {std::lock_guard l(mutex_);info_[WorkerCPU]=sched_getcpu();info_[WorkerInitialized]=1;info_[LivePinnedBytes]=PinnedBytes;info_[LiveDeviceBytes]=DeviceBytes;info_[StartupInverseCalls]=ActiveSlots*3;info_[StartupInverseBytes]=ActiveSlots*TripleBytes;initialized_=true;cv_.notify_all();}
  for(;;){
   bool make_plan=false;{std::lock_guard l(mutex_);if(stop_)break;make_plan=sealed_&&!plan_built_;}
   if(make_plan){build_plan();continue;}
   bool progressed=false;
   for(unsigned credit=0;credit<2;credit++)if(last[credit]>=0){unsigned j=unsigned(last[credit]);auto rc=cudaEventQuery(jobs_[j].done);if(rc==cudaErrorNotReady)continue;ck(rc);{std::lock_guard l(mutex_);need(jobs_[j].state==State::Submitted,"469 credit serial state");jobs_[j].state=State::Ready;info_[Ready]++;}last[credit]=-1;progressed=true;}
   unsigned chosen=8,credit=2;
   {std::lock_guard l(mutex_);if(stop_)break;
    if(mode_&&plan_built_&&next_move_<moves_.size()&&info_[Pending]<8){
     for(unsigned c=0;c<2;c++)if(last[c]<0){credit=c;break;}
     unsigned free_job=8,free_slot=PhysicalSlots;for(unsigned j=0;j<8;j++)if(jobs_[j].state==State::Unused){free_job=j;break;}for(unsigned s=0;s<PhysicalSlots;s++)if(spare_[s]){free_slot=s;break;}
     if(credit<2&&free_job<8&&free_slot<PhysicalSlots){chosen=free_job;const auto move=moves_[next_move_++];auto&j=jobs_[chosen];j.candidate=move.candidate;j.victim=move.victim;j.slot=free_slot;j.credit=credit;j.nonce=next_nonce_++;j.generation=generation_;j.format=format_;j.state=State::Copying;spare_[free_slot]=false;info_[Started]++;info_[QueuedMoves]--;info_[Pending]++;info_[MaxPending]=std::max(info_[MaxPending],info_[Pending]);}
    }
   }
   if(chosen<8){worker_upload(chosen,credit);last[credit]=int(chosen);continue;}if(progressed)continue;
   int wait=-1;for(int j:last)if(j>=0){wait=j;break;}if(wait>=0){ck(cudaEventSynchronize(jobs_[unsigned(wait)].done));continue;}
   std::unique_lock l(mutex_);info_[WorkerParked]=1;if(mode_&&plan_built_&&next_move_<moves_.size())info_[BackpressureParks]++;
   cv_.wait(l,[&]{bool free=false;for(bool s:spare_)free|=s;return stop_||(sealed_&&!plan_built_)||(mode_&&plan_built_&&next_move_<moves_.size()&&info_[Pending]<8&&free);});info_[WorkerParked]=0;
  }worker_drain();
 }catch(...){try{throw;}catch(const std::exception&e){std::fprintf(stderr,"469 worker failure: %s\n",e.what());}catch(...){std::fprintf(stderr,"469 worker unknown failure\n");}
  {std::lock_guard l(mutex_);poisoned_.store(true);info_[Errors]++;}worker_drain();std::lock_guard l(mutex_);initialized_=true;cv_.notify_all();
 }
}
void Resources::arm(uint64_t g,uint64_t f){
 check_idle();cpu_set_t a;CPU_ZERO(&a);need(!sched_getaffinity(0,sizeof(a),&a)&&CPU_COUNT(&a)==1&&CPU_ISSET(255,&a),"469 ARM actual main255");std::lock_guard l(mutex_);need(!armed_.load()&&g==generation_&&f==format_,"469 ARM once actualbank");armed_.store(true);info_[Armed]=1;
}
void Resources::poison()noexcept{std::lock_guard l(mutex_);poisoned_.store(true);info_[Errors]++;forward_tid_=0;pending_forward_.fill(0);observing_=false;stop_=true;cv_.notify_all();}
void Resources::publish_ready(unsigned layer,const uint8_t*counts){
 std::unique_lock l(mutex_,std::try_to_lock);if(!l.owns_lock()||!info_[Pending])return;need(!poisoned_.load(),"469 publication after poison forbidden");Device device;
 for(auto&j:jobs_)if(j.state==State::Ready){
  if(j.victim/384==layer&&counts[j.victim%384]){info_[ProtectedReady]++;continue;}
  auto rc=cudaEventQuery(j.done);if(rc==cudaErrorNotReady)continue;ck(rc);int old=map_[j.victim];need(old>=0&&slot_id_[unsigned(old)]==int(j.victim)&&map_[j.candidate]<0&&slot_id_[j.slot]<0&&!spare_[j.slot]&&j.nonce&&j.generation==generation_&&j.format==format_,"469 Ready exactmapping/nonce/bank");
  map_[j.candidate]=int(j.slot);slot_id_[j.slot]=int(j.candidate);map_[j.victim]=-1;slot_id_[unsigned(old)]=-1;spare_[unsigned(old)]=true;j.state=State::Unused;info_[Published]++;info_[Pending]--;cv_.notify_all();
 }
}
void Resources::before(uintptr_t gate,unsigned rows,const uint8_t*counts,unsigned n){
 if(!armed_.load())return;
 need(!task_active_.exchange(true),"469 N1 Task overlap refused");
 try{
  check_idle();cpu_set_t a;CPU_ZERO(&a);need(!sched_getaffinity(0,sizeof(a),&a)&&CPU_COUNT(&a)==1&&CPU_ISSET(254,&a),"469 publication/observer actualTask254");
  auto it=gate_id_.find(gate);need(it!=gate_id_.end()&&it->second%384==0&&rows>=1&&rows<=6&&counts&&n==384,"469 actual expert0/layer/M/count witness");unsigned layer=it->second/384,sum=0;for(unsigned e=0;e<384;e++){need(counts[e]<=rows,"469 original positive Count bound");sum+=counts[e];}need(sum<=rows*6,"469 positive route census");
  if(!sealed_.load()){std::lock_guard l(mutex_);need(!poisoned_.load()&&!closed_.load()&&!observing_,"469 clean before callback");
   if(!sealed_){need(completed_<32&&layer==layer_cursor_,"469 exact ordered0..39 training");if(layer==0){need(!current_m_&&!forward_tid_,"469 no partialforward restart");current_m_=rows;forward_tid_=uint64_t(syscall(SYS_gettid));}else need(current_m_==rows&&forward_tid_==uint64_t(syscall(SYS_gettid)),"469 sameM/issuer within forward; rebind only layer0");std::copy_n(counts,384,pending_count_.begin());observing_=true;observer_tid_=uint64_t(syscall(SYS_gettid));info_[BeforeCalls]++;info_[CurrentM]=current_m_;info_[PartialForward]=1;}
  }
  // Worker never publishes; current positive victims are protected BEFORE
  // actual Task427 firstcontains. No CUDA path when pending is zero.
  publish_ready(layer,counts);
 }catch(...){task_active_.store(false);poison();throw;}
}
void Resources::after(int status){
 if(!armed_.load())return;
 try{
  need(task_active_.load(),"469 matching before/after Task");
  need(!poisoned_.load()&&!closed_.load(),"469 after clean owner");
  if(status){std::lock_guard l(mutex_);info_[ObserverErrors]++;throw std::runtime_error("469 failedTask cannot commit training");}
  if(sealed_.load()){task_active_.store(false);return;}
  {std::lock_guard l(mutex_);
   if(observing_){
    need(observer_tid_==uint64_t(syscall(SYS_gettid))&&completed_<32&&!sealed_,"469 same callback issuer, finite training");
    std::copy(pending_count_.begin(),pending_count_.end(),pending_forward_.begin()+size_t(layer_cursor_)*384);info_[AfterCalls]++;observing_=false;observer_tid_=0;
    if(layer_cursor_==39){
     // Commit ALL40 successful layers atomically as one completed forward.
     for(unsigned id=0;id<15360;id++)if(auto c=pending_forward_[id]){need(score_[id]<=256u-(unsigned(c)+2)&&groups_[id]<32&&routes_[id]<=192u-c,"469 finite score overflow guard");score_[id]+=unsigned(c)+2;groups_[id]++;routes_[id]+=c;}
     pending_forward_.fill(0);completed_++;info_[TrainingForwards]=completed_;info_[TrainingRows]+=current_m_;layer_cursor_=0;current_m_=0;forward_tid_=0;info_[CurrentM]=0;info_[PartialForward]=0;
     if(completed_==32){need(info_[BeforeCalls]==1280&&info_[AfterCalls]==1280,"469 exact finite training census");info_[Frozen]=1;sealed_.store(true);cv_.notify_all();}
    }else layer_cursor_++;
    info_[LayerCursor]=layer_cursor_;
   }
  }task_active_.store(false);
 }catch(...){task_active_.store(false);poison();throw;}
}
int Resources::contains(int e,uintptr_t g,uintptr_t u,uintptr_t d)const{need(!closed_.load()&&!poisoned_.load(),"469 contains open clean owner");unsigned id=id_for(e,g,u,d);return map_[id]>=0?2:0;}
void Resources::begin(const DsWholeCommandV299*c,void**ticket){
 check_idle();need(ticket&&!*ticket&&c&&c->struct_size==sizeof(*c)&&c->version==299&&c->epoch&&c->epoch<std::numeric_limits<uint64_t>::max()&&c->rows>=1&&c->rows<=6&&c->dim==5120&&c->inter==2304&&c->groups>=1&&c->groups<=36&&c->route_words==c->rows*6*5120&&!c->reserved,"469 actual Task command/epoch");
 ds303::Command cmd{};cmd.epoch=c->epoch+1;cmd.rows=c->rows;cmd.groups=c->groups;std::copy(std::begin(c->view),std::end(c->view),std::begin(cmd.view));route_selected_.fill(false);std::fill_n(result_.data(),c->route_words,uint16_t(0));int layer=-1;uint64_t fresh=0,old=0;
 for(unsigned i=0;i<c->groups;i++){
  const auto&g=c->group[i];unsigned id=id_for(int(g.expert),g.gate_key,g.up_key,g.down_key);need(map_[id]>=0&&(!i||layer==int(id/384))&&!g.reserved[0]&&!g.reserved[1],"469 begin current membership/same layer");layer=int(id/384);auto&v=cmd.group[i];v.slot=unsigned(map_[id]);v.expert=g.expert;v.count=g.count;std::copy(std::begin(g.query),std::end(g.query),std::begin(v.query));std::copy(std::begin(g.route_slot),std::end(g.route_slot),std::begin(v.route_slot));std::copy(std::begin(g.alpha),std::end(g.alpha),std::begin(v.alpha));
  need(g.count>=1&&g.count<=6,"469 Count1..6");for(unsigned j=0;j<g.count;j++){need(g.route_slot[j]<36&&!route_selected_[g.route_slot[j]],"469 original route single ownership");route_selected_[g.route_slot[j]]=true;}
  if(v.slot>=ActiveSlots)fresh+=g.count;else{bool was_initial=prompt_rank_[v.slot]==id;if(was_initial)old+=g.count;else fresh+=g.count;}
 }
 Device device;op_->begin(cmd);active_epoch_.store(c->epoch);issuer_=uint64_t(syscall(SYS_gettid));if(last_issuer_&&last_issuer_!=issuer_)issuer_rebinds_++;borrowed_.store(false);*ticket=this;
 bool pending=false;{std::lock_guard l(mutex_);for(const auto&j:jobs_)if(j.state==State::Submitted){auto rc=cudaEventQuery(j.done);if(rc==cudaErrorNotReady)pending=true;else ck(rc);}info_[ComputeBegins]++;info_[ComputeUploadPending]+=pending;info_[NewRoutesUsed]+=fresh;info_[OldRoutesUsed]+=old;}
}
void Resources::finish(void*t,uint64_t epoch,const uint16_t**out,uint64_t words){need(t==this&&epoch==active_epoch_.load()&&issuer_==uint64_t(syscall(SYS_gettid))&&out&&!borrowed_.load(),"469 exact finish ticket");Device device;auto view=op_->finish(epoch+1);need(view.words==words&&!poisoned_.load(),"469 actual391 result/worker clean");for(unsigned r=0;r<view.rows*6;r++)if(route_selected_[r])std::memcpy(result_.data()+r*5120,view.data+r*5120,5120*2);{std::lock_guard l(mutex_);reported_op_=op_->stats();reported_finish_=true;}borrowed_.store(true);*out=result_.data();}
void Resources::release(void*t,uint64_t epoch){need(t==this&&epoch==active_epoch_.load()&&issuer_==uint64_t(syscall(SYS_gettid))&&borrowed_.load(),"469 release borrowed ticket");Device device;op_->release(epoch+1);borrowed_.store(false);active_epoch_.store(0);last_issuer_=issuer_;issuer_=0;}
void Resources::abort(void*t,uint64_t epoch){need(t==this&&epoch==active_epoch_.load()&&issuer_==uint64_t(syscall(SYS_gettid)),"469 abort actual ticket");Device device;op_->abort(epoch+1);borrowed_.store(false);active_epoch_.store(0);issuer_=0;}
Info Resources::info()const{
 std::unique_lock l(mutex_,std::try_to_lock);if(!l.owns_lock())throw Busy{};if(closing_.load()||closed_.load())throw Closing{};if(active_epoch_.load()||task_active_.load())throw Busy{};
 if(op_&&op_->stats()[ds303::Active])throw Busy{};auto r=info_;r[ActiveTicket]=active_epoch_.load()!=0;r[Borrowed]=borrowed_.load();r[Poisoned]=poisoned_.load();r[Closed]=closed_.load();r[HookRefs]=hook_refs_.load();return r;
}
ds370::Receipt370 Resources::legacy()const{
 std::lock_guard l(mutex_);ds370::Receipt370 r;r.adaptive=0;r.component_only=0;r.catalog=15360;r.physical=PhysicalSlots;r.active=ActiveSlots;r.spares=8;r.static_entries=0;r.pinned_bytes=info_[LivePinnedBytes];r.armed=info_[Armed];r.completed_forwards=info_[TrainingForwards];r.observer_calls=info_[BeforeCalls];r.observer_failures=info_[ObserverErrors];r.planner_wall_ns=info_[PlannerNs];r.planner_thread_ns=info_[PlannerThreadNs];r.stage_wall_ns=info_[ValidationNs]+info_[CopyNs];r.stage_thread_ns=info_[StageThreadNs];r.started=info_[Started];r.upload_completed=info_[Ready];r.published=info_[Published];r.protected_ready=info_[ProtectedReady];r.H2D_bytes=info_[H2DBytes];r.inverse_read_bytes=info_[InverseReadBytes];r.inverse_write_bytes=info_[InverseWriteBytes];r.pending=info_[Pending];r.close_cancelled=info_[Cancelled];r.closed=closed_.load();r.failed=info_[Errors];r.positive_drains=r.closed?1:0;
 for(const auto&j:jobs_)r.stages_inflight+=j.state==State::Copying||j.state==State::Submitted;
 auto s=op_?op_->stats():last_op_;auto&v=r.whole;
 v[ds299::Version]=370;v[ds299::Layout]=139;v[ds299::Generation]=generation_;v[ds299::FormatEpoch]=format_;v[ds299::Entries1]=ActiveSlots;v[ds299::WeightBytes1]=PhysicalSlots*TripleBytes;v[ds299::FinalBytes1]=DeviceBytes-Matrix;v[ds299::DeclaredPeak1]=DeviceBytes;v[ds299::RequestedPeak1]=20ull<<30;v[ds299::FreeAfterWorkspace1]=free_after_workspace_;
 v[ds299::Begin]=info_[ComputeBegins];v[ds299::Submitted]=info_[ComputeBegins];v[ds299::Finished]=s[ds303::Finishes]?s[ds303::Finishes]-1:0;v[ds299::Released]=s[ds303::Releases]?s[ds303::Releases]-1:0;v[ds299::Aborted]=s[ds303::Aborts];v[ds299::Failed]=info_[Errors];v[ds299::LiveTickets]=active_epoch_.load()!=0;v[ds299::Borrowed]=borrowed_.load();v[ds299::Groups1]=s[ds303::Groups]?s[ds303::Groups]-1:0;v[ds299::Routes1]=s[ds303::Routes]?s[ds303::Routes]-1:0;
 // Exact legacy transport contract: only actual391 successful finish snapshots
 // publish these counters; they include prime, and NEVER uploader weightbytes.
 if(reported_finish_){v[ds299::H2D1]=reported_op_[ds303::InputCopyBytes];v[ds299::D2H1]=reported_op_[ds303::OutputCopyBytes];v[ds299::Completed1]=reported_op_[ds303::CompletedEvents]-1;v[ds299::GuChunks1]=reported_op_[ds303::GuChunks]-1;v[ds299::DownChunks1]=reported_op_[ds303::DownChunks]-1;}
 v[ds299::IssuerRebinds]=issuer_rebinds_;v[ds299::InverseCalls]=info_[StartupInverseCalls];v[ds299::InverseBytes]=info_[StartupInverseBytes];v[ds299::StartupScratchLive]=closed_.load()?0:Matrix;v[ds299::StartupJobs1]=info_[StartupJobs];v[ds299::Closed]=closed_.load();return r;
}
void Resources::close_impl(){
 if(closed_.load())return;need(!hook_refs_.load()&&!active_epoch_.load()&&!task_active_.load(),"469 detach hooks/release Task before close");if(op_){auto s=op_->stats();need(!s[ds303::Active]&&!s[ds303::BorrowedView],"469 actual391 idle close");}
 {std::lock_guard l(mutex_);need(!closing_.load(),"469 serialized close");closing_.store(true);}
 unregister_diagnostic(this); // Retire diagnostics at CLOSING, before positive drains.
 {std::lock_guard l(mutex_);stop_=true;cv_.notify_all();}
 if(worker_.joinable())worker_.join(); // NEVER hold worker/metadata mutex.
 Device device;{std::lock_guard l(mutex_);info_[WorkerJoined]=1;}
 if(op_){need(!op_->shutdown(),"469 positive391 shutdown");last_op_=op_->stats();op_.reset();}if(compute_)ck(cudaStreamSynchronize(compute_));{std::lock_guard l(mutex_);info_[ComputeDrained]=1;}
 if(upload_)ck(cudaStreamSynchronize(upload_));for(auto&j:jobs_)if(j.done){ck(cudaEventDestroy(j.done));j.done=nullptr;}
 if(pins_)ck(cudaFreeHost(pins_));pins_=nullptr;for(void*p:{static_cast<void*>(scratch_),static_cast<void*>(bank_),static_cast<void*>(gu_),static_cast<void*>(down_)})if(p)ck(cudaFree(p));scratch_=bank_=nullptr;gu_=down_=nullptr;
 if(upload_)ck(cudaStreamDestroy(upload_));if(compute_)ck(cudaStreamDestroy(compute_));upload_=compute_=nullptr;weights_.reset();closed_.store(true);{std::lock_guard l(mutex_);info_[Closed]=1;info_[LiveOwner]=0;info_[LivePinnedBytes]=0;info_[LiveDeviceBytes]=0;}
}
void Resources::close(){close_impl();}
Resources::~Resources()noexcept{try{close_impl();}catch(...){std::terminate();}}
namespace {std::mutex diagnostics_mutex;std::vector<std::weak_ptr<Resources>>diagnostics;}
void register_diagnostic(const std::shared_ptr<Resources>&r){std::lock_guard l(diagnostics_mutex);diagnostics.erase(std::remove_if(diagnostics.begin(),diagnostics.end(),[](auto&w){return w.expired();}),diagnostics.end());need(diagnostics.empty(),"469 N1 owner registry");diagnostics.push_back(r);}
void unregister_diagnostic(const Resources*r)noexcept{std::lock_guard l(diagnostics_mutex);diagnostics.erase(std::remove_if(diagnostics.begin(),diagnostics.end(),[&](auto&w){auto x=w.lock();return !x||x.get()==r;}),diagnostics.end());}
int read_info(uint64_t*out,int words)noexcept{
 if(!out||words!=InfoWords)return 1;std::fill_n(out,InfoWords,uint64_t(0));
 try{std::shared_ptr<Resources>one;unsigned n=0;{std::lock_guard l(diagnostics_mutex);for(auto&w:diagnostics)if(auto x=w.lock()){one=std::move(x);n++;}}
  // Strong diagnostic borrow drops OUTSIDE registry lock; its final destructor
  // may itself unregister/close. Closing is serialized with info mutex.
  if(n!=1)return n?3:2;auto r=one->info();if(r[Closed])return 2;std::copy(r.begin(),r.end(),out);return 0;}catch(const Closing&){return 2;}catch(const Busy&){return 4;}catch(...){return 5;}
}
}
extern "C" int ds469_fixed_plan_info(uint64_t*out,int words)noexcept{return ds469::read_info(out,words);}
