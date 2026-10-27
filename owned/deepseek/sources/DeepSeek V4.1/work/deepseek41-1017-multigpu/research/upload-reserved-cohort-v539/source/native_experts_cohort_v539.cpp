#include "upload_cohort_v539.h"
#include "bulk_home_v418.h"
#include "fixed_plan_info_v469.h"
#include "../../gpu-gu-resident-owner-v19/resident_owner_v19.hpp"
#include "strata/ds41/native_experts.hpp"
#include "../../gpu-gu-paired-resident-owner-v147/paired_owner_v147.hpp"
#include <algorithm>
#include <chrono>
#include <thread>
#include <array>
#include <cmath>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <sched.h>
#include <set>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <stdexcept>
#include "pair_registry_v145.hpp"
#include "native_adaptive_lease_v387.hpp"
#include "../../whole-adaptive-task-gap-v386/source/policy_hooks_v386.hpp"

namespace {
using Upload151=ds1017_gu_paired_owner_v147::UploadStats147;
std::mutex upload151_mutex;
std::vector<std::weak_ptr<const Upload151>> upload151_live;
void publish_upload151(const std::shared_ptr<const Upload151>& stats){
 if(!stats)throw std::invalid_argument("missing paired147 upload receipt");
 std::lock_guard<std::mutex> lock(upload151_mutex);
 upload151_live.erase(std::remove_if(upload151_live.begin(),upload151_live.end(),
  [](const auto&w){return w.expired();}),upload151_live.end());
 upload151_live.emplace_back(stats); // weak only: never retains Owner or bank
}
}
extern "C" int ds151_last_upload_stats147(uint64_t*out,int count)noexcept{
 if(!out||count!=24)return 1;
 try{
  std::lock_guard<std::mutex> lock(upload151_mutex);
  std::shared_ptr<const Upload151> selected;
  size_t live=0;
  upload151_live.erase(std::remove_if(upload151_live.begin(),upload151_live.end(),
   [](const auto&w){return w.expired();}),upload151_live.end());
  for(const auto&w:upload151_live)if(auto x=w.lock()){++live;selected=std::move(x);}
  if(live!=1)return live?3:2;
  const auto values=ds1017_gu_paired_owner_v147::stats147(selected);
  if(values[0]!=147||values[1]!=139||values[19]!=1||values[20]||values[21]||values[12])return 4;
  std::copy(values.begin(),values.end(),out);return 0;
 }catch(...){return 5;}
}

namespace {
using Cohort539=std::array<uint64_t,264>;
std::mutex cohort539_mutex;
std::vector<std::weak_ptr<const Cohort539>>cohort539_live;
std::shared_ptr<const Cohort539>publish_cohort539(const std::vector<int>&cpus){
 auto row=std::make_shared<Cohort539>();
 const uint64_t header[16]={539,1,248,123,125,127,254,253,255,68,4,64,65,66,67,0};
 std::copy(std::begin(header),std::end(header),row->begin());
 if(cpus.size()!=248)throw std::logic_error("539 exact worker count");
 for(size_t i=0;i<cpus.size();++i)(*row)[16+i]=uint64_t(cpus[i]);
 std::lock_guard<std::mutex>lock(cohort539_mutex);
 cohort539_live.erase(std::remove_if(cohort539_live.begin(),cohort539_live.end(),[](const auto&w){return w.expired();}),cohort539_live.end());
 cohort539_live.emplace_back(row);return row;
}
}
extern "C" int ds539_worker_cohort_info(uint64_t*out,int words)noexcept{
 if(!out||words!=264)return 1;
 try{std::lock_guard<std::mutex>lock(cohort539_mutex);std::shared_ptr<const Cohort539>selected;size_t live=0;
  cohort539_live.erase(std::remove_if(cohort539_live.begin(),cohort539_live.end(),[](const auto&w){return w.expired();}),cohort539_live.end());
  for(const auto&w:cohort539_live)if(auto r=w.lock()){selected=std::move(r);++live;}
  if(live!=1)return live?3:2;std::copy(selected->begin(),selected->end(),out);return 0;
 }catch(...){return 5;}
}

namespace strata::ds41 {
namespace {
[[noreturn]] void bad(const std::string& s){throw std::runtime_error("native experts: "+s);}
struct Library {
    void* value=nullptr;
    explicit Library(const std::string& path){
        if(path.empty()||path[0]!='/')bad("library must have an absolute path");
        // GCC's OpenMP runtime keeps process-global allocations. Unloading and
        // reinitializing it for each engine leaks that state. Keep code and its
        // dependency globals mapped, while still closing this handle and
        // destroying every pool/task/checkpoint allocation on teardown.
        value=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE);if(!value)bad(dlerror());
    }
    ~Library(){if(value)dlclose(value);}
    template<class T> T symbol(const char* name){
        dlerror();void* v=dlsym(value,name);const char* error=dlerror();
        if(error||!v)bad(std::string("missing library symbol ")+name);
        return reinterpret_cast<T>(v);
    }
};
using PoolCreate=void*(*)(const int*,int,int*);
using Destroy=void(*)(void*);
using EngineCreate=void*(*)(void*,uintptr_t,int,int,int);
using TaskCreate=void*(*)(void*,const uintptr_t*,const uintptr_t*,const int*,int,int,int,int,int,float,const uint16_t*,const int64_t*,const float*,uint16_t*,uint64_t*);
using Run=int(*)(void*,void*);
using Work=int(*)(void*,uint64_t*,int);
using Prime=int(*)(void*,const void*,int,int,int);
using Prime2=int(*)(void*,const void*,const void*,int,int,int);
using Prime3=int(*)(void*,const char*,const void*,const void*,int,int,int);
void validate_cpus(const std::vector<int>& cpus,int coordinator){
    if(cpus.empty()||cpus.size()>256)bad("invalid worker count");
    cpu_set_t allowed;CPU_ZERO(&allowed);
    if(sched_getaffinity(0,sizeof(allowed),&allowed))bad("cannot read allowed CPU affinity");
    std::set<int>seen;
    for(int cpu:cpus)
        if(cpu<0||cpu>=CPU_SETSIZE||!CPU_ISSET(cpu,&allowed)||!seen.insert(cpu).second)bad("invalid, disallowed or duplicate worker CPU");
    if(coordinator<0||coordinator>=CPU_SETSIZE||!CPU_ISSET(coordinator,&allowed))bad("disallowed coordinator CPU");
}
}
struct NativeExperts::Impl {
    struct Layer {int experts,dim,inter,topk;float limit;std::vector<uintptr_t>w,s;std::vector<int>bits;std::vector<StagedProjection>staged;};
    struct Task {
        std::vector<uint16_t>x,out;std::vector<int64_t>ids;std::vector<float>routing;
        alignas(8) std::array<uint64_t,2> flags{};void* handle=nullptr;
    };
    std::shared_ptr<const BorrowedWeights> weights;
    NativeExperts*identity387=nullptr;
    const bool adaptive_engine387=ds387::enabled();
    const bool adaptive_policy387=adaptive_engine387?ds387::adaptive_mode():false;
    std::shared_ptr<ds387::Lease>adaptive_lease387;
    using WholeAttach387=int(*)(void*,const DsWholeHooksV299*);
    using PolicyAttach387=int(*)(void*,const ds370::PolicyHooks370*)noexcept;
    WholeAttach387 attach_whole387=nullptr;PolicyAttach387 attach_policy387=nullptr;
    std::shared_ptr<ds1017_gu_resident_owner_v19::Owner>gu_owner_v19;
    std::shared_ptr<const ds1017_gu_paired_owner_v147::UploadStats147>gu_upload_stats147;
    // Libraries and borrowed payload outlive the engine and all of its tasks.
    std::unique_ptr<Library> task_lib,pool_lib;
    Destroy destroy_pool=nullptr,destroy_engine=nullptr;
    TaskCreate create_task=nullptr;Run run_task=nullptr;Work work=nullptr;Prime prime=nullptr;Prime2 prime2=nullptr;Prime3 prime3=nullptr;
    std::shared_ptr<const Cohort539>cohort539;
    bool bulk418=false;void*home418[2]{};
    void* pool=nullptr;void* engine=nullptr;bool a8=false,mixed_f32=false;int workers=0;size_t staged_bytes=0;
    using PairInfo145=int(*)(void*,uint64_t*,int);PairInfo145 pair_info145=nullptr;
    std::array<uint64_t,8> pair_binding145{};
    std::map<int,Layer>layers;
    std::map<std::tuple<int,int,int>,std::unique_ptr<Task>>tasks;
    mutable std::mutex mutex;
    void*pool_for418(int layer,int expert){return bulk418?home418[(layer+expert)%2]:pool;}
    int pool_info418(uint64_t*out,int words){
        if(!bulk418)return pair_info145(pool,out,words);
        if(!out||words!=12)return 1;std::array<uint64_t,12>a{},b{};
        if(pair_info145(home418[0],a.data(),12)||pair_info145(home418[1],b.data(),12))return 1;
        for(int i:{0,1,2,3,4,5,9,10,11})if(a[i]!=b[i])return 1;
        std::copy(a.begin(),a.end(),out);out[6]+=b[6];out[7]+=b[7];out[8]+=b[8];return 0;
    }
    Task& task(int layer,int rows,int topk){
        const auto&l=layers.at(layer);
    auto key=std::make_tuple(layer,rows,topk);auto position=tasks.find(key);
    if(position==tasks.end()){
        auto task=std::make_unique<Task>();task->x.resize(rows*l.dim);task->ids.resize(rows*topk);
        task->routing.resize(rows*topk);task->out.resize(rows*topk*l.dim);
        // Publish owned buffers before the engine retains their addresses. On
        // failure erase this task; the C ABI promises nullptr means no task.
        position=tasks.emplace(key,std::move(task)).first;auto& t=*position->second;
        t.handle=create_task(engine,l.w.data(),l.s.data(),l.bits.data(),l.experts,l.dim,l.inter,topk,rows,l.limit,
                               t.x.data(),t.ids.data(),t.routing.data(),t.out.data(),t.flags.data());
        if(!t.handle){tasks.erase(position);bad("task creation failed");}
    }
    return *position->second;
    }
    ~Impl(){ds387::cleanup(identity387);adaptive_lease387.reset();if(engine)destroy_engine(engine);gu_owner_v19.reset();gu_upload_stats147.reset();if(pool)destroy_pool(pool);for(void*home:home418)if(home)destroy_pool(home);}
};
NativeExperts::NativeExperts(std::shared_ptr<const BorrowedWeights> weights,const std::string& task_library,
                             const std::string& pool_library,bool a8,std::vector<int> cpus,int coordinator,bool mixed_f32)
    :impl_(std::make_unique<Impl>()){
    if(!weights)bad("missing borrowed checkpoint");if(a8&&mixed_f32)bad("ambiguous activation arithmetic");validate_cpus(cpus,coordinator);
    auto& p=*impl_;p.identity387=this;p.weights=std::move(weights);p.a8=a8;p.mixed_f32=mixed_f32;p.workers=int(cpus.size());
    if(p.adaptive_engine387){
        auto exact=[](const char*k,const char*value){const char*v=std::getenv(k);if(!v||std::strcmp(v,value))bad(std::string("387 requires ")+k+"="+value);};
        exact("DS1017_GPU_WHOLE_EXPERT_OWNER_V299","1");exact("DS1017_GPU_GU_OVERLAP_V6","0");
        exact("DS1017_GPU_GU_PAIR32_OWNER_V147","0");exact("DS1017_ADAPTIVE_FUSED_STAGE_COPY_V378","1");
        const char*component=std::getenv("DS1017_WHOLE_ADAPTIVE_COMPONENT_V370");
        if(component&&std::strcmp(component,"0"))bad("387 production forbids component fixture mode");
    }
    const char*bulk=std::getenv("DS1017_CPU_NUMA_BULK_V418");if(bulk&&std::strcmp(bulk,"0")&&std::strcmp(bulk,"1"))bad("literal418 flag required");p.bulk418=bulk&&!std::strcmp(bulk,"1");
    const char*upload465=std::getenv("DS1017_FIXED_PLAN_UPLOAD_V465");
    if(upload465&&std::strcmp(upload465,"0")&&std::strcmp(upload465,"1"))bad("466 uploader flag must be0 or1");
    const bool reserve253465=upload465&&!std::strcmp(upload465,"1");
    if(!reserve253465||!p.bulk418||!p.adaptive_engine387)bad("539 requires fixed uploader253/bulk418/paired469 owner");
    const char*reserve539=std::getenv("DS1017_UPLOAD_RESERVED_COHORT_V539");
    if(!reserve539||std::strcmp(reserve539,"1"))bad("539 requires literal reserved cohort1; no252 fallback");
    const char*refresh469=std::getenv("DS1017_FIXED_PLAN_REFRESH_V469");
    if(!refresh469||(std::strcmp(refresh469,"0")&&std::strcmp(refresh469,"1")))bad("469 literal refresh0 or1 required");
    if(p.adaptive_policy387)bad("469 old370 adaptive policy must remain0");
    if(p.bulk418){
        if(!a8||mixed_f32||cpus.size()!=248u)bad("539 exact248 physical workers required");
        // Selected Engine passes cpus.front() as its historical constructor
        // placeholder. Its real outer issuer is pinned later by this setting;
        // Task418 separately enforces sched_getcpu()==254 at every bulk job.
        const char*issuer=std::getenv("DS41_NATIVE_COORDINATOR");
        if(!issuer||std::strcmp(issuer,"254"))bad("424 actual outer coordinator setting must be254");
        if(coordinator!=cpus.front()&&coordinator!=254)bad("424 unknown legacy coordinator placeholder");
        std::vector<int>wanted;for(int i=0;i<127;i++)if(i<64||i>67)wanted.push_back(i);for(int i=128;i<253;i++)wanted.push_back(i);if(cpus!=wanted)bad("539 exact CPU tuple excludes64..67 and preserves both homes");
        const char*test=std::getenv("DS1017_CPU_NUMA_BULK_COMPONENT_V419");if(test&&std::strcmp(test,"0"))bad("418 production rejects component mode");
    }
    p.task_lib=std::make_unique<Library>(task_library);
    if(p.adaptive_engine387){
        // Refuse a missing Task386 policy ABI BEFORE pools, CUDA allocations or prefill.
        p.attach_whole387=p.task_lib->symbol<Impl::WholeAttach387>("ds41_moe_whole_attach_v299");
        p.attach_policy387=p.task_lib->symbol<Impl::PolicyAttach387>("ds386_moe_policy_attach");
    }p.pool_lib=std::make_unique<Library>(pool_library);
    const std::string prefix=mixed_f32?"ds41_mixedf32_batch_":(a8?"ds41_wxa8_batch_":"ds41_mxfp4_batch_");
    auto create_pool=p.pool_lib->symbol<PoolCreate>((prefix+"create").c_str());
    p.destroy_pool=p.pool_lib->symbol<Destroy>((prefix+"destroy").c_str());
    auto function=p.pool_lib->symbol<void*>((prefix+"gemv").c_str());
    auto create_engine=p.task_lib->symbol<EngineCreate>("ds41_moe_engine_create");
    p.destroy_engine=p.task_lib->symbol<Destroy>("ds41_moe_engine_destroy");
    p.create_task=p.task_lib->symbol<TaskCreate>("ds41_moe_task_create");
    p.run_task=p.task_lib->symbol<Run>("ds41_moe_task_run");p.work=p.task_lib->symbol<Work>("ds41_moe_engine_work");
    p.pair_binding145=pair145::info(*p.weights);
    if(p.pair_binding145[2]){
        if(!a8||mixed_f32||p.pair_binding145[2]!=46080||p.pair_binding145[3]!=305764761600ull||!p.pair_binding145[4]||!p.pair_binding145[5]||p.pair_binding145[6]!=1)bad("complete typed139 registry required");
        p.pair_info145=p.pool_lib->symbol<Impl::PairInfo145>("ds143_pair32_pool_info");
    }
    int status=0;
    if(p.bulk418){
        if(!p.pair_info145)bad("418 complete typed139 registry required");
        auto create_home=p.pool_lib->symbol<decltype(&ds418_home_pool_create)>("ds418_home_pool_create");
        for(int h=0;h<2;h++){std::vector<int>local;for(int cpu:cpus)if(cpu/128==h)local.push_back(cpu);p.home418[h]=create_home(local.data(),int(local.size()),h,&status);if(!p.home418[h]||status)bad("418 home pool creation failed: "+std::to_string(status));}
    }else{p.pool=create_pool(cpus.data(),p.workers,&status);if(!p.pool||status)bad("worker pool creation failed: "+std::to_string(status));}
    if(p.pair_info145){std::array<uint64_t,12>v{};if(p.pool_info418(v.data(),12)||v[0]!=143||v[1]!=139||v[2]!=32||v[3]!=576||v[4]!=288||v[5]!=1||v[6]!=uint64_t(p.workers)||v[7]||v[8]||v[9]||v[10]||v[11])bad("actual fresh CPU143 paired pool required");}
    p.prime=reinterpret_cast<Prime>(dlsym(p.pool_lib->value,"ds41_wxa8_prime"));dlerror();
    p.prime2=reinterpret_cast<Prime2>(dlsym(p.pool_lib->value,"ds41_wxa8_prime_v2"));dlerror();
    p.prime3=reinterpret_cast<Prime3>(dlsym(p.pool_lib->value,"ds41_wxa8_prime_v3"));dlerror();
    // The task library's legacy flag selects the ABI carrying weight-format
    // bits. Actual activation arithmetic belongs to the named projection DSO.
    if(p.bulk418){
        auto create_bulk=p.task_lib->symbol<decltype(&ds418_moe_engine_create)>("ds418_moe_engine_create");
        Ds418Pools b{sizeof(b),418,{p.home418[0],p.home418[1]},reinterpret_cast<uintptr_t>(function),{123,125},{127,254},{0,0}};
        p.engine=create_bulk(&b);
    }else p.engine=create_engine(p.pool,reinterpret_cast<uintptr_t>(function),a8||mixed_f32,p.workers,coordinator);
    if(!p.engine)bad("task engine creation failed");
    p.cohort539=publish_cohort539(cpus);
}
NativeExperts::~NativeExperts()=default;
void NativeExperts::add_layer(int layer,const std::string& prefix,int experts,int dim,int inter,int topk,float limit){
    auto& p=*impl_;std::lock_guard<std::mutex>lock(p.mutex);
    if(layer<0||layer>=40||p.layers.count(layer)||experts<1||experts>384||dim<32||dim>8192||dim%32||
       inter<32||inter>8192||inter%32||topk<1||topk>6||topk>experts||!std::isfinite(limit)||limit<0)bad("invalid layer geometry");
    if(p.bulk418&&(experts!=384||dim!=5120||inter!=2304||topk!=6||limit!=10.f))bad("418 exact production geometry required");
    Impl::Layer l{experts,dim,inter,topk,limit,{},{},{},{}};size_t new_staged=0;
    for(int e=0;e<experts;e++)for(auto which:{"w1","w3","w2"}){
        bool down=std::string(which)=="w2";
        const std::string projection=prefix+"."+std::to_string(e)+"."+which;
        if(pair145::contains(*p.weights,projection)){
            const char*borrowed=std::getenv("DS41_CPU_BLOCK_CACHE_BORROWED");
            if(!p.pair_info145||!p.a8||p.mixed_f32||!p.prime3||!borrowed||std::strcmp(borrowed,"1"))bad("paired139 requires actual CPU143 and named borrowed prime_v3");
            const auto d=pair145::projection(*p.weights,projection,down?dim:inter,down?inter:dim);
            const auto*data=static_cast<const uint8_t*>(d.data);const auto*footer=data+512;
            if(p.prime3(p.pool_for418(d.layer,d.expert),(projection+".weight").c_str(),data,footer,d.rows,d.columns,0))bad("CPU143 paired named binding failed");
            l.w.push_back(reinterpret_cast<uintptr_t>(data));l.s.push_back(reinterpret_cast<uintptr_t>(footer));l.bits.push_back(0);continue;
        }
        if(p.bulk418)bad("418 refuses untyped/non139 expert before inference");
        if(p.weights->is_output16_compact(projection)){
            const char*borrowed=std::getenv("DS41_CPU_BLOCK_CACHE_BORROWED");
            if(!p.a8 || p.mixed_f32 || !p.prime3 || !borrowed || std::strcmp(borrowed,"1"))
                bad("output16 descriptors require unchanged CPU118 A8 named-borrowed cache mode");
            const auto compact=p.weights->compact_projection(projection,down?dim:inter,down?inter:dim);
            // Actual compact.data is a stable KEY, not a raw row address. The
            // native scale-footer address belongs to the same real allocation
            // and is passed only for CPU118's legacy nonnull argument check.
            // prime_v3 binds the named cache; CPU118 requires that cache on run,
            // and its 16-row scheduling consumes the real blocked tile pointer.
            const auto*scale_footer=compact.bits?nullptr:compact.first_scale_footer();
            if(p.prime3(p.pool,(projection+".weight").c_str(),compact.data,scale_footer,
                        compact.rows,compact.columns,compact.bits))bad("CPU compact named cache binding failed");
            l.w.push_back(reinterpret_cast<uintptr_t>(compact.data));
            l.s.push_back(reinterpret_cast<uintptr_t>(scale_footer));l.bits.push_back(compact.bits);
            continue; // No StagedProjection and no retained duplicate raw bytes.
        }
        auto owned=p.weights->stage_projection(prefix+"."+std::to_string(e)+"."+which,down?dim:inter,down?inter:dim);
        const auto&v=owned.projection;new_staged+=owned.owned_bytes();
        if(new_staged>(1ull<<30)-p.staged_bytes)bad("cross-shard expert copies exceed1GiB model budget");
        if(v.bits&&!p.a8&&!p.mixed_f32)bad("own quant requires an explicit mixed-format arithmetic implementation");
        if(p.prime3){if(p.prime3(p.pool,(prefix+"."+std::to_string(e)+"."+which+".weight").c_str(),v.weights,v.scales,v.rows,v.columns,v.bits))bad("CPU named packed cache preparation failed");}
        else if(p.prime2){if(p.prime2(p.pool,v.weights,v.scales,v.rows,v.columns,v.bits))bad("CPU packed cache preparation failed");}
        else if(p.prime&&p.prime(p.pool,v.weights,v.rows,v.columns,v.bits))bad("CPU locality preparation failed");
        l.w.push_back(reinterpret_cast<uintptr_t>(v.weights));l.s.push_back(reinterpret_cast<uintptr_t>(v.scales));l.bits.push_back(v.bits);l.staged.push_back(std::move(owned));
    }
    p.layers.emplace(layer,std::move(l));p.staged_bytes+=new_staged;
    if(p.pair_info145&&p.layers.size()==40){std::array<uint64_t,12>v{};if(p.pool_info418(v.data(),12)||v[0]!=143||v[1]!=139||v[2]!=32||v[3]!=576||v[4]!=288||v[5]!=1||v[6]!=uint64_t(p.workers)||v[7]!=46080||v[8]!=305764761600ull||v[9]!=p.pair_binding145[4]||v[10]!=p.pair_binding145[5]||v[11])bad("actual complete CPU143 paired pool binding differs");}
}
std::vector<uint16_t> NativeExperts::run_selected(int layer,int rows,const std::vector<uint16_t>& input,
                                       const std::vector<int64_t>& ids,const std::vector<float>& routing,int topk){
    auto& p=*impl_;std::lock_guard<std::mutex>lock(p.mutex);
    const auto found=p.layers.find(layer);if(found==p.layers.end()||rows<1||rows>8)bad("unknown layer or unsupported batch");
    const auto& l=found->second;
    if(topk==0)topk=l.topk;
    if(topk<1||topk>l.topk)bad("invalid expert subset size");
    if(input.size()!=size_t(rows*l.dim)||ids.size()!=size_t(rows*topk)||routing.size()!=ids.size())bad("activation/routing shape differs");
    for(auto value:input)if((value&0x7f80)==0x7f80)bad("nonfinite activation");
    for(int row=0;row<rows;row++)for(int i=0;i<topk;i++){
        const auto at=row*topk+i;
        if(ids[at]<0||ids[at]>=l.experts||!std::isfinite(routing[at])||routing[at]<0)bad("invalid routing");
        for(int j=0;j<i;j++)if(ids[row*topk+j]==ids[at])bad("duplicate routed expert");
    }
    auto&t=p.task(layer,rows,topk);
    std::copy(input.begin(),input.end(),t.x.begin());std::copy(ids.begin(),ids.end(),t.ids.begin());std::copy(routing.begin(),routing.end(),t.routing.begin());
    const int status=p.run_task(p.engine,t.handle);if(status)bad("expert computation failed: "+std::to_string(status));
    return t.out;
}
void NativeExperts::prepare_batch(int rows){
    auto&p=*impl_;std::lock_guard<std::mutex>lock(p.mutex);
    if(rows<1||rows>8||p.layers.size()!=40)bad("complete native layers and bounded prepared batch required");
    for(const auto&entry:p.layers)p.task(entry.first,rows,entry.second.topk);
}
std::vector<uint16_t> NativeExperts::run(int layer,int rows,const std::vector<uint16_t>&input,const std::vector<int64_t>&ids,const std::vector<float>&routing){
    return run_selected(layer,rows,input,ids,routing,0);
}
std::vector<uint64_t> NativeExperts::worker_rows()const{
    auto& p=*impl_;std::lock_guard<std::mutex>lock(p.mutex);std::vector<uint64_t>counts(p.workers);
    int status=p.work(p.engine,counts.data(),p.workers);if(status)bad("worker affinity failed: "+std::to_string(status));return counts;
}
}

namespace strata::ds41 {
void NativeExperts::build_gu_cache_v19(size_t bytes,const std::vector<std::array<uint64_t,384>>&frequency){
 auto&p=*impl_;std::lock_guard<std::mutex>lock(p.mutex);
 if(p.adaptive_engine387){
  if(p.adaptive_lease387){if(p.adaptive_lease387->closed())bad("387 terminal owner already closed");return;}
  if(p.gu_owner_v19||p.gu_upload_stats147||!p.a8||p.mixed_f32||p.layers.size()!=40||frequency.size()!=40||bytes!=(size_t(20)<<30))bad("387 requires fresh exact A8 paired model and20GiB GPU1 budget");
  const auto binding=pair145::info(*p.weights);
  if(binding!=p.pair_binding145||binding[0]!=145||binding[1]!=139||binding[2]!=46080||binding[3]!=305764761600ull||!binding[4]||!binding[5]||binding[6]!=1||binding[7])bad("387 complete immutable typed139 registry binding required");
  std::array<uint64_t,12>pool_info{};
  if(!p.pair_info145||p.pool_info418(pool_info.data(),12)||pool_info[0]!=143||pool_info[1]!=139||pool_info[2]!=32||pool_info[3]!=576||pool_info[4]!=288||pool_info[5]!=1||pool_info[6]!=uint64_t(p.workers)||pool_info[7]!=46080||pool_info[8]!=305764761600ull||pool_info[9]!=binding[4]||pool_info[10]!=binding[5]||pool_info[11])bad("387 actual complete CPU143 pool binding required");
  std::vector<ds299::Descriptor>descriptors;descriptors.reserve(15360);
  for(const auto&item:p.layers){
   const int layer=item.first;const auto&l=item.second;
   if(layer<0||layer>=40||l.experts!=384||l.dim!=5120||l.inter!=2304||l.topk!=6||l.limit!=10.f)bad("387 literal whole-expert layer geometry");
   uint64_t total=0;for(auto count:frequency[layer]){if(count>524288)bad("387 frequency count bound");total+=count;}if(!total)bad("387 actual prefill frequencies required for every layer");
   for(int e=0;e<384;e++){
    const std::string prefix="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(e);
    for(int role=0;role<3;role++){
     const auto v=pair145::projection(*p.weights,prefix+(role==0?".w1":role==1?".w3":".w2"),role==2?5120:2304,role==2?2304:5120);
     if(v.struct_bytes!=sizeof(v)||v.layout_tag!=139||v.bits||v.bytes!=6635520||v.layer!=layer||v.expert!=e||v.role!=role||v.reserved||v.owner_generation!=binding[4]||v.bank_format_epoch!=binding[5]||l.bits[e*3+role]||l.w[e*3+role]!=reinterpret_cast<uintptr_t>(v.data)||l.s[e*3+role]!=reinterpret_cast<uintptr_t>(static_cast<const uint8_t*>(v.data)+512))bad("387 all actual Task triple keys/roles/epochs must match typed registry");
    }
    // EXACT original19 policy: normalize by actual per-layer routed uses,
    // including shorter decoder suffix layers. Never use future decode tokens.
    descriptors.push_back({layer,e,(frequency[layer][e]*1000000000000ull)/total});
   }
  }
  // Local table buffers outlive Owner::create;303 validates/uploads them before
  // returning and does not retain their host addresses afterward.
  std::array<uint32_t,65536>values{};std::array<uint8_t,65536>domain{};std::array<uint64_t,16>table_info{};Ds276ExpTable table{};
  if(ds276_exp_table_build(&table,values.data(),values.size(),domain.data(),domain.size(),table_info.data(),16)||ds276_exp_table_validate(&table))bad("387 qualified same-process literal libmvec table276");

  ds370::Config370 config;config.adaptive=p.adaptive_policy387;config.component_only=false;config.budget=bytes;
  auto owner=ds370::Owner::create(p.weights,std::move(descriptors),table,config);
  const auto receipt=owner->receipt();const auto&v=receipt.whole;
  if(receipt.version!=370||receipt.struct_bytes!=672||receipt.component_only||receipt.catalog!=15360||receipt.physical!=1078||
     receipt.adaptive||receipt.active!=1070||receipt.spares!=8||receipt.static_entries!=0||
     v[ds299::Version]!=370||v[ds299::Layout]!=139||v[ds299::Generation]!=binding[4]||v[ds299::FormatEpoch]!=binding[5]||
     v[ds299::Entries0]||v[ds299::Entries1]!=receipt.active||v[ds299::DeclaredPeak1]>bytes||
     v[ds299::LiveTickets]||v[ds299::Borrowed]||v[ds299::Failed]||v[ds299::StartupJobs1]!=1||v[ds299::Closed])bad("387 actual370 startup receipt differs");
  std::array<uint64_t,64>actual469{};
  int actual469rc=ds469_fixed_plan_info(actual469.data(),64);
  const auto snapshot469deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(20);
  while(actual469rc==4&&std::chrono::steady_clock::now()<snapshot469deadline){std::this_thread::yield();actual469rc=ds469_fixed_plan_info(actual469.data(),64);}
  if(actual469rc||actual469[0]!=469||actual469[1]!=64||
     actual469[2]!=uint64_t(!std::strcmp(std::getenv("DS1017_FIXED_PLAN_REFRESH_V469"),"1"))||
     actual469[3]!=1||actual469[4]!=binding[4]||actual469[5]!=binding[5]||actual469[6]!=15360||
     actual469[7]!=1078||actual469[8]!=1070||actual469[9]!=8||actual469[10]||
     actual469[11]!=253||actual469[12]!=1||actual469[13]||actual469[52]||actual469[53]||
     actual469[54]||actual469[55]||actual469[56]||actual469[60]!=1)
      bad("469 actual fixed-plan startup receipt differs");
  auto attach_whole=p.attach_whole387;auto attach_policy=p.attach_policy387;
  if(!attach_whole||!attach_policy)bad("387 missing early-qualified Task386 hook ABI");
  bool whole=false,policy=false;
  try{
   if(attach_whole(p.engine,&owner->hooks()))bad("387 whole hook attach rejected");whole=true;
   if(attach_policy(p.engine,&owner->policy_hooks()))bad("387 policy hook attach rejected");policy=true;
   // CapturedImpl staysalive until explicit registered lease cleanup. The closure
   // is cleared during close BEFORE Implfree. No external raw pointer is exposed.
   p.adaptive_lease387=ds387::install(*this,owner,[&p,attach_whole,attach_policy]{
    std::lock_guard<std::mutex>lock(p.mutex);
    if(attach_policy(p.engine,nullptr))bad("387 policy detach requires idle Taskengine");
    if(attach_whole(p.engine,nullptr))bad("387 whole detach requires idle Taskengine");
   });
  }catch(...){
   if(policy&&attach_policy(p.engine,nullptr))std::terminate();
   if(whole&&attach_whole(p.engine,nullptr))std::terminate();
   if(owner->close())std::terminate();throw;
  }
  return;
 }
 if(p.adaptive_lease387)bad("387 immutable owner mode changed");
 if(p.gu_owner_v19)return; // fixed membership for this Engine lifetime
 const bool paired=p.pair_binding145[2]!=0;
 if(paired){
  const char*scratch=std::getenv("DS1017_GPU_GU_STARTUP_SCRATCH_BYTES_V147");
  if(!scratch||std::strcmp(scratch,"6635520"))throw std::invalid_argument("paired GU requires explicit startup scratch budget6635520");
  if(p.pair_binding145[0]!=145||p.pair_binding145[1]!=139||p.pair_binding145[2]!=46080||
     p.pair_binding145[3]!=305764761600ull||!p.pair_binding145[4]||!p.pair_binding145[5])
   throw std::invalid_argument("paired GU complete typed registry binding required");
 }
 if(!p.a8||p.mixed_f32||p.layers.size()!=40||frequency.size()!=40||!bytes||bytes>(22ull<<30))throw std::invalid_argument("GU owner requires exact complete A8 model and private budget");
 std::vector<ds1017_gu_resident_owner_v19::Descriptor>descriptors;descriptors.reserve(15360);
  std::vector<ds1017_gu_paired_owner_v147::Sidecar147>sidecars;if(paired)sidecars.reserve(15360);
 for(const auto&item:p.layers){
  const int layer=item.first;const auto&l=item.second;
  if(layer<0||layer>=40||l.experts!=384||l.dim!=5120||l.inter!=2304||l.topk!=6)throw std::invalid_argument("GU layer geometry");
  uint64_t total=0;for(auto count:frequency[layer]){if(count>524288)throw std::invalid_argument("GU frequency count bound");total+=count;}
  if(!total)throw std::invalid_argument("GU layer must have actual prefill frequencies");
  for(int e=0;e<384;e++){
   const std::string prefix="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(e);
   if(paired){
    const auto gate=pair145::projection(*p.weights,prefix+".w1",2304,5120);
    const auto up=pair145::projection(*p.weights,prefix+".w3",2304,5120);
    if(gate.layout_tag!=139||up.layout_tag!=139||gate.bits||up.bits||
       gate.layer!=layer||up.layer!=layer||gate.expert!=e||up.expert!=e||gate.role!=0||up.role!=1||
       gate.owner_generation!=p.pair_binding145[4]||up.owner_generation!=gate.owner_generation||
       gate.bank_format_epoch!=p.pair_binding145[5]||up.bank_format_epoch!=gate.bank_format_epoch||
       l.bits[e*3]||l.bits[e*3+1]||l.w[e*3]!=reinterpret_cast<uintptr_t>(gate.data)||
       l.w[e*3+1]!=reinterpret_cast<uintptr_t>(up.data)||
       l.s[e*3]!=reinterpret_cast<uintptr_t>(static_cast<const uint8_t*>(gate.data)+512)||
       l.s[e*3+1]!=reinterpret_cast<uintptr_t>(static_cast<const uint8_t*>(up.data)+512))
     throw std::invalid_argument("paired GU typed keys/generation/epoch differ from CPU Task");
    ds1017_gu_resident_owner_v19::Descriptor out;out.layer=layer;out.expert=e;
    out.gate_key=l.w[e*3];out.up_key=l.w[e*3+1];
    out.gate=static_cast<const uint8_t*>(gate.data);out.up=static_cast<const uint8_t*>(up.data);
    out.gate_bytes=gate.bytes;out.up_bytes=up.bytes;
    out.frequency=(frequency[layer][e]*1000000000000ull)/total;
    descriptors.push_back(out);
    ds1017_gu_paired_owner_v147::Sidecar147 tag;tag.layer=layer;tag.expert=e;
    tag.gate_key=out.gate_key;tag.up_key=out.up_key;tag.layout_tag=139;
    tag.owner_generation=gate.owner_generation;tag.bank_format_epoch=gate.bank_format_epoch;
    sidecars.push_back(tag);continue;
   }
   if(!p.weights->is_output16_compact(prefix+".w1")||!p.weights->is_output16_compact(prefix+".w3"))throw std::invalid_argument("GU cache requires checked exact Output16 compact descriptors");
   const auto gate=p.weights->compact_projection(prefix+".w1",2304,5120),up=p.weights->compact_projection(prefix+".w3",2304,5120);
   if(gate.bits||up.bits||l.bits[e*3]||l.bits[e*3+1]||l.w[e*3]!=reinterpret_cast<uintptr_t>(gate.data)||l.w[e*3+1]!=reinterpret_cast<uintptr_t>(up.data))throw std::invalid_argument("GU Task weight keys differ from protected compact allocations");
   ds1017_gu_resident_owner_v19::Descriptor out;out.layer=layer;out.expert=e;out.gate_key=l.w[e*3];out.up_key=l.w[e*3+1];out.gate=gate.data;out.up=up.data;out.gate_bytes=gate.bytes;out.up_bytes=up.bytes;
   // Normalize by actual rows per layer: decoder suffix layers intentionally
   // process fewer prompt rows. Rank policy does not alter inference math.
   out.frequency=(frequency[layer][e]*1000000000000ull)/total;
   descriptors.push_back(out);
  }
 }
 std::shared_ptr<ds1017_gu_resident_owner_v19::Owner>owner;
 std::shared_ptr<const ds1017_gu_paired_owner_v147::UploadStats147>upload_stats;
 if(paired){
  auto made=ds1017_gu_paired_owner_v147::create_paired147(std::move(descriptors),std::move(sidecars),
   bytes,6635520,std::shared_ptr<const void>(p.weights));
  owner=std::move(made.owner);upload_stats=std::move(made.upload_stats);
 }else{
  // Historical mathematical path retained; private147 rejects untyped creation.
  owner=ds1017_gu_resident_owner_v19::Owner::create(std::move(descriptors),bytes,std::shared_ptr<const void>(p.weights));
 }
 auto attach=p.task_lib->symbol<int(*)(void*,const DsGuHooksV18*)>("ds41_moe_gu_attach_v18");
 if(attach(p.engine,&owner->hooks()))throw std::runtime_error("GU owner Task attach rejected");
 p.gu_owner_v19=std::move(owner);p.gu_upload_stats147=std::move(upload_stats);
 if(p.gu_upload_stats147)publish_upload151(p.gu_upload_stats147);
}
std::array<uint64_t,32> NativeExperts::gu_statistics_v19()const{
 const auto&p=*impl_;std::lock_guard<std::mutex>lock(p.mutex);
 static_assert(ds1017_gu_resident_owner_v19::StatCount==32);
 return p.gu_owner_v19?p.gu_owner_v19->stats():std::array<uint64_t,32>{};
}
}
