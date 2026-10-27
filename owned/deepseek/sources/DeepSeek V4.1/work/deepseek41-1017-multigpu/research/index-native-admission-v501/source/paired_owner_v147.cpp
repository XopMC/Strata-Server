#include "index_pair_admission_v501.hpp"
#include "../../gpu-gu-paired-resident-owner-v147/paired_owner_v147.hpp"
#include "../../pair-packed-inverse-v141-r2/source/pair_inverse_v141.h"
#include "../../gpu-gu-byteperm-v13/batch_v5.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <sys/syscall.h>
#include <unistd.h>
#include <xmmintrin.h>

namespace ds1017_gu_paired_owner_v147 {
class UploadStats147 final {
public:
 std::array<std::atomic<uint64_t>,StatCount147>value{};
 void set(Stat147 s,uint64_t n)noexcept{value[s].store(n,std::memory_order_relaxed);}
 void add(Stat147 s,uint64_t n=1)noexcept{value[s].fetch_add(n,std::memory_order_relaxed);}
};
namespace {
// Only this TU can manufacture the typed deleter. get_deleter is type-safe;
// no raw void layout guess, TLS, global next-owner table or reference cycle.
struct Retention147 {
 std::shared_ptr<const void> original_weights;
 std::vector<Sidecar147>sidecar;
 std::shared_ptr<UploadStats147>stats;
 size_t cache_budget=0,startup_scratch_budget=0;
 void operator()(const void*)const noexcept{} // original_weights owns original object.
};
void optin147(){
 const char*v=std::getenv("DS1017_GPU_GU_PAIR32_OWNER_V147");
 const char*i=std::getenv("DS1017_PAIR32_BYTE_INVERSE_V141");
 const char*c=std::getenv("DS1017_COLD_FOOTER_MAX_V108");
 if(!v||std::strcmp(v,"139")||!i||std::strcmp(i,"1")||(c&&std::strcmp(c,"0")))
  throw std::invalid_argument("explicit paired147=139/inverse141=1 and literal cold19 policy required");
}
}
std::array<uint64_t,StatCount147>stats147(const std::shared_ptr<const UploadStats147>&s)noexcept{
 std::array<uint64_t,StatCount147>r{};if(s)for(size_t i=0;i<r.size();i++)r[i]=s->value[i].load(std::memory_order_relaxed);return r;
}
}

namespace ds1017_gu_resident_owner_v19 {
namespace {
namespace B=ds1017_gu_batch_v5;
namespace P147=ds1017_gu_paired_owner_v147;
static_assert(StatCount==32,"originalOwner19statisticsABIchanged");
constexpr size_t GiB=size_t(1)<<30,Reserve=size_t(512)<<20,Stage=size_t(4)<<20;
constexpr int Guard=16,MaxPairs=36;
constexpr size_t PairBytes=2*B::WeightBytes,ResultWords=2ull*MaxPairs*B::Rows;
constexpr size_t ResultDeviceBytes=(ResultWords+2*Guard)*sizeof(float);
constexpr size_t ResultHostBytes=(ResultWords+2*Guard)*sizeof(float);
constexpr size_t SentinelBytes=(1+2*Guard)*sizeof(int32_t);
constexpr uint32_t Canary=0x6b19c4a5U;
constexpr uint64_t Generation=0x4755524553494419ULL;
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
uint64_t tid() noexcept{return uint64_t(::syscall(SYS_gettid));}
void fp_mode(){require(std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&0xe040u)==0,"GU owner requires original RN/noFTZ/noDAZ");}
struct Device {
 int previous=0;Device(){ck(cudaGetDevice(&previous));ck(cudaSetDevice(1));}
 ~Device()noexcept{if(cudaSetDevice(previous)!=cudaSuccess)std::terminate();}
};
struct Key {uintptr_t gate,up;bool operator==(const Key&o)const noexcept{return gate==o.gate&&up==o.up;}};
struct Hash {size_t operator()(Key k)const noexcept{uint64_t x=uint64_t(k.gate)^(uint64_t(k.up)+(uint64_t(k.gate)<<6)+(uint64_t(k.gate)>>2));x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;return size_t(x);}};
bool range(uintptr_t p,size_t n)noexcept{return p&&n&&n<=std::numeric_limits<uintptr_t>::max()-p;}
void topology(){
 const char*owner=std::getenv("DS1017_GPU_GU_RESIDENT_OWNER_V19");
 const char*kernel=std::getenv("DS1017_GPU_GU_BYTEPERM_V13");
 ds501::require_environment();
 require(owner&&!std::strcmp(owner,"1")&&kernel&&!std::strcmp(kernel,"1"),"explicit GU owner/V13 opt-ins required");
 ds501::validate_all();
 cudaStreamCaptureStatus state{};ck(cudaStreamIsCapturing(nullptr,&state));require(state==cudaStreamCaptureStatusNone,"GU owner construction requires ordinary startup");
 // Qualified V13 returns before enqueue for this deliberately invalid geometry.
 // Detect its already-latched OFF policy before substantial allocations.
 require(B::launch(nullptr,nullptr,0,0,nullptr,nullptr,nullptr)==cudaErrorInvalidValue,"selected V13 launch policy is not enabled");
}
}
struct Owner::Impl {
 std::shared_ptr<const void> retained_weights;
 std::shared_ptr<P147::UploadStats147> paired_stats;
 uint8_t*paired_upload147=nullptr;
 std::vector<Descriptor> resident;
 std::vector<float> max_factor;
 std::unordered_map<Key,uint32_t,Hash> slots;
 std::unordered_map<uintptr_t,uint32_t> gate_slots,up_slots;
 mutable std::mutex mutex,ref_mutex;
 std::unique_lock<std::mutex> lease;
 std::shared_ptr<Owner> hook_hold;uint64_t hook_refs=0;
 std::array<std::atomic<uint64_t>,StatCount> counters{};
 DsGuHooksV18 hooks{};
 cudaStream_t stream=nullptr;cudaEvent_t done=nullptr;
 uint8_t*bank=nullptr;const uint8_t**device_table=nullptr;
 B::Request*device_request=nullptr,*host_request=nullptr;
 float*device_result_base=nullptr,*device_result=nullptr,*host_result_base=nullptr,*host_result=nullptr;
 int32_t*sentinel_base=nullptr,*sentinel=nullptr;
 uint8_t*upload_stage=nullptr;
 std::atomic<uint64_t> issuer{0};uint64_t last_issuer=0,epoch=0;
 uint32_t output_words=0;bool active=false,borrowed=false,poison=false,closed=false,guards_initialized=false;
 void add(Stat s,uint64_t n=1)noexcept{counters[s].fetch_add(n,std::memory_order_relaxed);}
 int failure(int code)noexcept{add(Failures);return code;}
 void set(Stat s,uint64_t n)noexcept{counters[s].store(n,std::memory_order_relaxed);}
 static void metadata(const std::vector<Descriptor>&descriptors){
  require(!descriptors.empty()&&descriptors.size()<=40u*384u,"bounded GU descriptor list required");
  std::set<std::pair<int,int>> identities;std::vector<std::pair<uintptr_t,uintptr_t>> ranges;ranges.reserve(descriptors.size()*2);
  for(const auto&d:descriptors){
   require(d.layer>=0&&d.layer<40&&d.expert>=0&&d.expert<384&&d.rows==B::Rows&&d.columns==B::K&&d.bits==0,"GU descriptor logical identity/geometry differs");
   require(d.gate_bytes==B::WeightBytes&&d.up_bytes==B::WeightBytes&&d.gate_key==reinterpret_cast<uintptr_t>(d.gate)&&d.up_key==reinterpret_cast<uintptr_t>(d.up)&&d.gate_key!=d.up_key&&range(d.gate_key,d.gate_bytes)&&range(d.up_key,d.up_bytes),"GU descriptor exact compact keys/ranges required");
   require(identities.emplace(d.layer,d.expert).second,"duplicate layer/expert GU descriptor");ranges.emplace_back(d.gate_key,d.gate_key+d.gate_bytes);ranges.emplace_back(d.up_key,d.up_key+d.up_bytes);
  }
  std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();i++)require(ranges[i-1].second<=ranges[i].first,"overlapping compact GU source allocations");
 }
 float inspect_pair(const Descriptor&d){
  float maximum=0.f;
  // SAME logical oldtile/group/row sequence and original FP max/half order.
  // Only byte address mapping changes to original pair32 footer512/544.
  for(const uint8_t*p:{d.gate,d.up})for(size_t offset=256;offset<B::WeightBytes;offset+=288){
   const size_t block=(offset-256)/288,tile=block/B::Groups,group=block%B::Groups;
   const size_t paired_footer=((tile/2)*B::Groups+group)*576+512+(tile&1)*32;
   for(int row=0;row<16;row++){
    uint16_t code;std::memcpy(&code,p+paired_footer+row*2,2);require(code<=254,"GU paired139 footer must be zero-extended E8M0<=254");
    const uint32_t bits=code?uint32_t(code)<<23:0x00400000U;float factor;std::memcpy(&factor,&bits,4);maximum=std::max(maximum,factor*.5f);
   }
  }
  paired_stats->add(P147::ValidatedFooterWords147,2*(B::WeightBytes/288)*16);
  return maximum;
 }
 void upload(uint8_t*destination,const uint8_t*source,size_t bytes){
  for(size_t at=0;at<bytes;at+=Stage){size_t n=std::min(Stage,bytes-at);std::memcpy(upload_stage,source+at,n);ck(cudaMemcpyAsync(destination+at,upload_stage,n,cudaMemcpyHostToDevice,stream));add(UploadH2DBytes,n);add(UploadCopies);ck(cudaStreamSynchronize(stream));}
 }
 void upload_paired147(uint8_t*destination,const uint8_t*source,size_t bytes){
  require(paired_upload147&&bytes==B::WeightBytes,"one bounded paired147 upload scratch required");
  upload(paired_upload147,source,bytes);paired_stats->add(P147::PairedH2DBytes147,bytes);
  const int code=ds141_pair32_to_output16(paired_upload147,bytes,B::Rows,B::K,139,destination,bytes,1,reinterpret_cast<uint64_t>(stream));
  paired_stats->set(P147::LastInverseStatus147,uint64_t(int64_t(code)));
  if(code)throw std::runtime_error("paired147 purebyte inverse refused status "+std::to_string(code));
  paired_stats->add(P147::InverseSubmitted147);paired_stats->add(P147::InverseInputBytes147,bytes);paired_stats->add(P147::InverseOutputBytes147,bytes);
  // No eager extra synchronize: next upload and final table upload use SAME
  // stream, so scratch overwrite follows inverse reads. Final sync proves all.
 }
 void init(std::vector<Descriptor> descriptors,size_t budget,std::shared_ptr<const void> token){
  P147::optin147();
  const auto*typed=std::get_deleter<P147::Retention147>(token);
  require(typed&&typed->original_weights&&typed->stats,"paired147 typed retained sidecar required, untagged Owner::create refused");
  paired_stats=typed->stats;
  require(typed->cache_budget==budget&&typed->startup_scratch_budget==B::WeightBytes&&budget<=22*GiB-B::WeightBytes,"explicit paired147 finalcache+startup scratch budget <=22GiB required");
  require(typed->sidecar.size()==descriptors.size()&&!typed->sidecar.empty(),"one paired147 sidecar for each descriptor required");
  uint64_t owner_generation=0,format_epoch=0;
  for(size_t i=0;i<descriptors.size();i++){
   const auto&d=descriptors[i];const auto&s=typed->sidecar[i];
   require(s.layer==d.layer&&s.expert==d.expert&&s.gate_key==d.gate_key&&s.up_key==d.up_key&&s.layout_tag==139&&!s.reserved&&s.owner_generation&&s.bank_format_epoch,"paired147 descriptor/sidecar identity/tag differs");
   require(d.gate_key%(size_t(2)<<20)==0&&d.up_key%(size_t(2)<<20)==0,"paired147 CPU bank allocations must remain2MiB aligned");
   if(!i){owner_generation=s.owner_generation;format_epoch=s.bank_format_epoch;}
   require(owner_generation==s.owner_generation&&format_epoch==s.bank_format_epoch,"paired147 mixed owner generation/format epoch refused");
  }
  metadata(descriptors);fp_mode();topology();
  paired_stats->set(P147::OwnerGeneration147,owner_generation);paired_stats->set(P147::FormatEpoch147,format_epoch);paired_stats->set(P147::ValidatedDescriptors147,descriptors.size());
  retained_weights=std::move(token);set(Version,19);set(BudgetBytes,budget);
  const size_t fixed=sizeof(B::Request)+ResultDeviceBytes+SentinelBytes;
  require(budget>=fixed+PairBytes+sizeof(const uint8_t*),"GU budget cannot hold one full pair and work buffers");
  const size_t capacity=(budget-fixed)/(PairBytes+sizeof(const uint8_t*));
  std::stable_sort(descriptors.begin(),descriptors.end(),[](const Descriptor&a,const Descriptor&b){if(a.frequency!=b.frequency)return a.frequency>b.frequency;if(a.layer!=b.layer)return a.layer<b.layer;return a.expert<b.expert;});
  descriptors.resize(std::min(capacity,descriptors.size()));resident=std::move(descriptors);max_factor.resize(resident.size());
  const size_t weight_bytes=resident.size()*PairBytes,table_bytes=resident.size()*sizeof(const uint8_t*),total=fixed+weight_bytes+table_bytes;
  Device device;cudaStreamCaptureStatus capture{};ck(cudaStreamIsCapturing(nullptr,&capture));require(capture==cudaStreamCaptureStatusNone,"GPU1 startup may not be captured");
  const size_t peak=total+B::WeightBytes;
  paired_stats->set(P147::FinalRequestedDeviceBytes147,total);paired_stats->set(P147::PeakRequestedDeviceBytes147,peak);paired_stats->set(P147::ResidentEntries147,resident.size());
  size_t free=0,physical=0;ck(cudaMemGetInfo(&free,&physical));set(VRAMFreeBefore,free);require(free>=peak&&free-peak>=Reserve,"insufficient GPU1 VRAM including explicit paired147 startup scratch+512MiB reserve");
  set(RequestedDeviceBytes,total);set(WeightDeviceBytes,weight_bytes);set(ResidentEntries,resident.size());set(StagingPinnedBytes,Stage);set(RequestPinnedBytes,sizeof(B::Request));set(ResultPinnedBytes,ResultHostBytes);set(ScratchSentinelBytes,SentinelBytes);
  ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
  ck(cudaMalloc(reinterpret_cast<void**>(&bank),weight_bytes));ck(cudaMalloc(reinterpret_cast<void**>(&device_table),table_bytes));ck(cudaMalloc(reinterpret_cast<void**>(&device_request),sizeof(B::Request)));
  ck(cudaMalloc(reinterpret_cast<void**>(&device_result_base),ResultDeviceBytes));device_result=device_result_base+Guard;
  ck(cudaMalloc(reinterpret_cast<void**>(&sentinel_base),SentinelBytes));sentinel=sentinel_base+Guard;
  ck(cudaMalloc(reinterpret_cast<void**>(&paired_upload147),B::WeightBytes));paired_stats->set(P147::StartupScratchPeak147,B::WeightBytes);paired_stats->set(P147::StartupScratchLive147,B::WeightBytes);
  ck(cudaHostAlloc(reinterpret_cast<void**>(&upload_stage),Stage,cudaHostAllocDefault));ck(cudaHostAlloc(reinterpret_cast<void**>(&host_request),sizeof(B::Request),cudaHostAllocDefault));ck(cudaHostAlloc(reinterpret_cast<void**>(&host_result_base),ResultHostBytes,cudaHostAllocDefault));host_result=host_result_base+Guard;
  std::array<uint32_t,1+2*Guard> canary{};canary.fill(Canary);ck(cudaMemcpyAsync(sentinel_base,canary.data(),SentinelBytes,cudaMemcpyHostToDevice,stream));
  for(int i=0;i<Guard;i++){std::memcpy(host_result_base+i,&Canary,4);std::memcpy(host_result+ResultWords+i,&Canary,4);}ck(cudaMemcpyAsync(device_result_base,host_result_base,Guard*4,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(device_result+ResultWords,host_result+ResultWords,Guard*4,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));guards_initialized=true;
  std::vector<const uint8_t*> table(resident.size());
  for(uint32_t i=0;i<resident.size();i++){
   const auto&d=resident[i];max_factor[i]=inspect_pair(d);slots.emplace(Key{d.gate_key,d.up_key},i);gate_slots.emplace(d.gate_key,i);up_slots.emplace(d.up_key,i);table[i]=bank+size_t(i)*PairBytes;
   upload_paired147(bank+size_t(i)*PairBytes,d.gate,d.gate_bytes);upload_paired147(bank+size_t(i)*PairBytes+B::WeightBytes,d.up,d.up_bytes);
  }
  ck(cudaMemcpyAsync(device_table,table.data(),table_bytes,cudaMemcpyHostToDevice,stream));add(UploadH2DBytes,table_bytes);add(UploadCopies);ck(cudaStreamSynchronize(stream));
  paired_stats->set(P147::InverseCompleted147,paired_stats->value[P147::InverseSubmitted147].load(std::memory_order_relaxed));
  ck(cudaFree(paired_upload147));paired_upload147=nullptr;paired_stats->set(P147::StartupScratchLive147,0);
  ck(cudaFreeHost(upload_stage));upload_stage=nullptr;set(StagingPinnedBytes,0);ck(cudaMemGetInfo(&free,&physical));set(VRAMFreeAfter,free);paired_stats->set(P147::InitComplete147,1);
 }
 // Destructor is allowed only after detach and every post/read lease has ended.
 // A failed driver drain is terminal; never free in-use buffers or reset hardware.
 void close()noexcept{
  if(closed)return;
  if(!stream&&!done&&!bank&&!device_table&&!device_request&&!device_result_base&&!sentinel_base&&!host_request&&!host_result_base&&!upload_stage&&!paired_upload147){closed=true;set(Closed,1);if(paired_stats)paired_stats->set(P147::Closed147,1);retained_weights.reset();return;}
  std::unique_lock<std::mutex> lock(mutex,std::try_to_lock);if(!lock.owns_lock()||active)std::terminate();
  try{Device device;if(stream)ck(cudaStreamSynchronize(stream));
   if(guards_initialized&&sentinel_base){std::array<uint32_t,1+2*Guard> s{};ck(cudaMemcpy(s.data(),sentinel_base,SentinelBytes,cudaMemcpyDeviceToHost));for(auto x:s)require(x==Canary,"V13 scratch sentinel unexpectedly modified");}
   if(guards_initialized&&device_result_base){std::array<uint32_t,2*Guard> s{};ck(cudaMemcpy(s.data(),device_result_base,Guard*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(s.data()+Guard,device_result+ResultWords,Guard*4,cudaMemcpyDeviceToHost));for(auto x:s)require(x==Canary,"GU result device guard changed");}
   for(void*p:{static_cast<void*>(bank),static_cast<void*>(device_table),static_cast<void*>(device_request),static_cast<void*>(device_result_base),static_cast<void*>(sentinel_base),static_cast<void*>(paired_upload147)})if(p)ck(cudaFree(p));
   for(void*p:{static_cast<void*>(host_request),static_cast<void*>(host_result_base),static_cast<void*>(upload_stage)})if(p)ck(cudaFreeHost(p));
   if(done)ck(cudaEventDestroy(done));if(stream)ck(cudaStreamDestroy(stream));
   bank=nullptr;device_table=nullptr;device_request=nullptr;device_result_base=nullptr;sentinel_base=nullptr;host_request=nullptr;host_result_base=nullptr;upload_stage=nullptr;paired_upload147=nullptr;done=nullptr;stream=nullptr;closed=true;set(Closed,1);if(paired_stats){paired_stats->set(P147::StartupScratchLive147,0);paired_stats->set(P147::Closed147,1);}retained_weights.reset();
  }catch(...){std::terminate();}
 }
 int contains(int expert,uintptr_t gate,uintptr_t up)noexcept{
  if(counters[Closed].load(std::memory_order_relaxed)||expert<0||expert>=384||!gate||!up){add(ContainsInvalid);return -1;}
  const auto found=slots.find(Key{gate,up});
  if(found==slots.end()){if(gate_slots.count(gate)||up_slots.count(up)){add(ContainsInvalid);return -1;}add(ContainsMisses);return 0;}
  if(resident[found->second].expert!=expert){add(ContainsInvalid);return -1;}add(ContainsHits);return 1;
 }
 bool same_issuer(void*ticket,uint64_t expected)const noexcept{return ticket==this&&issuer.load(std::memory_order_acquire)==tid()&&expected==epoch;}
 int begin(const DsGuCommandV18*c,void**ticket)noexcept{
  add(BeginCalls);if(!ticket)return failure(1);*ticket=nullptr;
  try{
   std::unique_lock<std::mutex> lock(mutex,std::try_to_lock);require(lock.owns_lock()&&!active&&!closed&&!poison,"GU N1 owner unavailable");fp_mode();
   require(c&&c->version==18&&c->struct_size==sizeof(*c)&&c->epoch>epoch&&c->rows>=1&&c->rows<=6&&c->dim==B::K&&c->inter==B::Rows&&c->groups>=1&&c->groups<=B::MaxExperts&&!c->reserved,"GU command ABI/epoch/geometry differs");
   B::Request request{};request.generation=Generation;request.experts=c->groups;std::array<bool,6> views{};
   for(unsigned q=0;q<6;q++){
    const auto&v=c->view[q];if(q>=c->rows||!v.codes){require(!v.codes&&!v.scales&&!v.sums&&!v.columns,"inactive GU view must be zero");continue;}
    require(v.columns==B::K&&v.scales&&v.sums,"missing actual A8 view");views[q]=true;std::memcpy(request.codes+q*B::K,v.codes,B::K);std::memcpy(request.scales+q*B::Groups,v.scales,B::Groups*4);std::memcpy(request.sums+q*(B::K/4),v.sums,B::K);
   }
   unsigned scratch_offset=0;std::array<bool,36> routes{};std::array<unsigned,6> row_uses{};int layer=-1,last_expert=-1;
   std::array<uint32_t,B::MaxExperts> used_slots{};unsigned ones=0,twos=0,sixes=0;
   for(unsigned j=0;j<c->groups;j++){
    const auto&g=c->group[j];const auto it=slots.find(Key{g.gate_key,g.up_key});require(it!=slots.end(),"uncached GU keys submitted");const uint32_t slot=it->second;const auto&d=resident[slot];
    require(g.expert==unsigned(d.expert)&&d.expert>last_expert&&!g.reserved&&(g.count==1||g.count==2||g.count==6)&&g.output_words==request.output_words,"GU group identity/count/offset differs");last_expert=d.expert;
    if(layer<0)layer=d.layer;require(layer==d.layer,"one GU command must contain exactly one layer");for(unsigned k=0;k<j;k++)require(used_slots[k]!=slot,"duplicate resident GU slot");used_slots[j]=slot;
    auto&e=request.entry[j];e.weight=slot;e.count=g.count;e.output_words=request.output_words;e.scratch_words=scratch_offset;std::array<bool,6> queries{};
    for(unsigned t=0;t<6;t++){if(t>=g.count){require(!g.query[t]&&!g.route_slot[t],"inactive GU query metadata must be zero");continue;}unsigned q=g.query[t],r=g.route_slot[t];require(q<c->rows&&views[q]&&!queries[q]&&r<c->rows*6&&r/6==q&&!routes[r],"GU query/scatter conservation differs");queries[q]=true;routes[r]=true;row_uses[q]++;e.query[t]=q;}
    request.pairs+=g.count;request.output_words+=2*g.count*B::Rows;scratch_offset+=2*g.count*B::Rows*B::Groups;ones+=g.count==1;twos+=g.count==2;sixes+=g.count==6;
   }
   require(request.pairs<=MaxPairs&&request.output_words==c->output_words&&request.output_words<=ResultWords&&*std::max_element(row_uses.begin(),row_uses.end())<=6&&scratch_offset<=B::MaxScratchWords,"GU batch conservation differs");
   unsigned bad_codes=0,bad_sums=0,bad_scales=0;float maximum=0.f;
#pragma clang loop vectorize(enable)
   for(size_t j=0;j<sizeof(request.codes);j+=4){int sum=0;for(int k=0;k<4;k++){bad_codes|=unsigned(request.codes[j+k]==-128);sum+=request.codes[j+k];}bad_sums|=unsigned(sum!=request.sums[j/4]);}
#pragma clang loop vectorize(enable)
   for(float scale:request.scales){bad_scales|=unsigned(!std::isfinite(scale)||scale<0.f);maximum=std::max(maximum,scale);}
   require(!bad_codes&&!bad_sums&&!bad_scales,"GU A8 code/sum/scale certificate differs");
   for(unsigned j=0;j<c->groups;j++)require(double(max_factor[used_slots[j]])*maximum*48768.0*B::Groups<=double(std::numeric_limits<float>::max())*.5,"GU sufficient finite output bound failed");
   Device device;std::memcpy(host_request,&request,sizeof(request));lease=std::move(lock);active=true;borrowed=false;epoch=c->epoch;output_words=request.output_words;const uint64_t current=tid();if(last_issuer&&last_issuer!=current)add(IssuerRebinds);last_issuer=current;issuer.store(current,std::memory_order_release);set(LiveTickets,1);
   try{
    ck(cudaMemcpyAsync(device_request,host_request,sizeof(request),cudaMemcpyHostToDevice,stream));add(CommandH2DBytes,sizeof(request));ck(B::launch(device_table,device_request,request.experts,request.output_words,sentinel,device_result,stream));add(SubmittedCommands);add(SubmittedGroups,c->groups);add(SubmittedRoutePairs,request.pairs);add(Count1Groups,ones);add(Count2Groups,twos);add(Count6Groups,sixes);
    ck(cudaMemcpyAsync(host_result,device_result,request.output_words*4,cudaMemcpyDeviceToHost,stream));add(ResultD2HBytes,request.output_words*4);ck(cudaEventRecord(done,stream));
   }catch(...){poison=true;if(cudaStreamSynchronize(stream)!=cudaSuccess)std::terminate();active=false;borrowed=false;issuer.store(0,std::memory_order_release);set(LiveTickets,0);add(AbortedCommands);lease.unlock();throw;}
   *ticket=this;return 0;
  }catch(...){return failure(72);}
 }
 int finish(void*ticket,uint64_t expected,const float**result,uint64_t words)noexcept{
  if(!same_issuer(ticket,expected))return failure(74);
  try{require(active&&!borrowed&&result&&words==output_words,"GU finish ticket/view extent differs");
   const uintptr_t address=reinterpret_cast<uintptr_t>(result),begin=reinterpret_cast<uintptr_t>(host_result_base);
   require(range(address,sizeof(*result))&&!(address<begin+ResultHostBytes&&begin<address+sizeof(*result)),"GU result pointer may not alias borrowed pinned result");
   *result=nullptr;Device device;ck(cudaEventSynchronize(done));unsigned bad=0;
#pragma clang loop vectorize(enable)
   for(size_t j=0;j<words;j++){uint32_t bits;std::memcpy(&bits,host_result+j,4);bad|=unsigned((bits&0x7f800000U)==0x7f800000U);}
   require(!bad,"GPU finite output");borrowed=true;set(BorrowedViews,1);*result=host_result;return 0;
  }catch(...){poison=true;return failure(74);}
 }
 int release_view(void*ticket,uint64_t expected)noexcept{
  if(!same_issuer(ticket,expected))return failure(76);
  try{require(active&&borrowed,"GU result is not borrowed");borrowed=false;active=false;set(BorrowedViews,0);set(LiveTickets,0);add(ReleasedViews);issuer.store(0,std::memory_order_release);lease.unlock();return 0;}catch(...){poison=true;return failure(76);}
 }
 int abort(void*ticket,uint64_t expected)noexcept{
  if(!same_issuer(ticket,expected))return failure(75);
  try{require(active,"GU abort ticket inactive");Device device;ck(cudaStreamSynchronize(stream));active=false;borrowed=false;set(BorrowedViews,0);set(LiveTickets,0);add(AbortedCommands);issuer.store(0,std::memory_order_release);lease.unlock();return 0;}catch(...){poison=true;return failure(75);}
 }
};
std::shared_ptr<Owner> Owner::create(std::vector<Descriptor>d,size_t b,std::shared_ptr<const void>token){return std::shared_ptr<Owner>(new Owner(std::move(d),b,std::move(token)));}
Owner::Owner(std::vector<Descriptor>d,size_t b,std::shared_ptr<const void>token):impl_(new Impl){
 try{impl_->init(std::move(d),b,std::move(token));impl_->hooks={sizeof(DsGuHooksV18),18,this,retain_hook,release_hook,contains_hook,begin_hook,finish_hook,release_view_hook,abort_hook};}
 catch(...){if(impl_->paired_stats)impl_->paired_stats->add(P147::Failures147);impl_->close();throw;}
}
Owner::~Owner()noexcept{impl_->close();}
const DsGuHooksV18&Owner::hooks()const noexcept{return impl_->hooks;}
std::array<uint64_t,StatCount>Owner::stats()const noexcept{std::array<uint64_t,StatCount>r{};for(size_t i=0;i<r.size();i++)r[i]=impl_->counters[i].load(std::memory_order_relaxed);return r;}
void Owner::retain_hook(void*p)noexcept{try{auto*o=static_cast<Owner*>(p);std::lock_guard<std::mutex>lock(o->impl_->ref_mutex);if(!o->impl_->hook_refs)o->impl_->hook_hold=o->shared_from_this();o->impl_->hook_refs++;}catch(...){std::terminate();}}
void Owner::release_hook(void*p)noexcept{std::shared_ptr<Owner>last;try{auto*o=static_cast<Owner*>(p);std::lock_guard<std::mutex>lock(o->impl_->ref_mutex);require(o->impl_->hook_refs>0,"unbalanced GU hook release");if(!--o->impl_->hook_refs){require(o->impl_->counters[LiveTickets].load(std::memory_order_acquire)==0,"cannot detach GU owner with live post readers");last=std::move(o->impl_->hook_hold);}}catch(...){std::terminate();}}
int Owner::contains_hook(void*p,int e,uintptr_t g,uintptr_t u)noexcept{return p?static_cast<Owner*>(p)->impl_->contains(e,g,u):-1;}
int Owner::begin_hook(void*p,const DsGuCommandV18*c,void**t)noexcept{if(!p){if(t)*t=nullptr;return 1;}return static_cast<Owner*>(p)->impl_->begin(c,t);}
int Owner::finish_hook(void*p,void*t,uint64_t e,const float**r,uint64_t n)noexcept{if(!p){if(r)*r=nullptr;return 1;}return static_cast<Owner*>(p)->impl_->finish(t,e,r,n);}
int Owner::release_view_hook(void*p,void*t,uint64_t e)noexcept{return p?static_cast<Owner*>(p)->impl_->release_view(t,e):1;}
int Owner::abort_hook(void*p,void*t,uint64_t e)noexcept{return p?static_cast<Owner*>(p)->impl_->abort(t,e):1;}
}

namespace ds1017_gu_paired_owner_v147 {
Created147 create_paired147(std::vector<Descriptor>d,std::vector<Sidecar147>s,size_t cache,size_t scratch,std::shared_ptr<const void>token){
 optin147();
 constexpr size_t GiB=size_t(1)<<30,Bytes=6635520;
 if(!token||scratch!=Bytes||cache>22*GiB-scratch)throw std::invalid_argument("explicit paired147 retained bank/finalcache+6635520scratch required");
 auto receipt=std::make_shared<UploadStats147>();receipt->set(Version147,147);receipt->set(Layout147,139);receipt->set(CacheBudget147,cache);receipt->set(StartupScratchBudget147,scratch);receipt->set(DeclaredPeakBudget147,cache+scratch);receipt->set(OriginalStatWords147,ds1017_gu_resident_owner_v19::StatCount);
 const void*identity=token.get();
 std::shared_ptr<const void>typed(identity,Retention147{std::move(token),std::move(s),receipt,cache,scratch});
 auto owner=Owner::create(std::move(d),cache,std::move(typed));
 return Created147{std::move(owner),std::move(receipt),cache,scratch};
}
}
