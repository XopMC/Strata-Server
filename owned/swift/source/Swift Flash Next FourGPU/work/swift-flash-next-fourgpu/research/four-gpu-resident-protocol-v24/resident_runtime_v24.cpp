#include "resident_protocol_v24.hpp"
#include "resident_runtime_v11.hpp"
#include "cpu_fenv_admission_v11.hpp"
#include "../four-gpu-expert-dispatch-v10-r2/dispatch_v10.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "peer_matrix.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
namespace resident_runtime_v11 {
namespace D=four_dispatch_v10;
namespace G=cpu_grouped_v6;
namespace R=cpu_rows_v5;
namespace B=four_banks_v3;
namespace N=strata::kernels;
namespace {
void need(bool yes,const char*why) { if(!yes) throw std::runtime_error(why); }
void ck(cudaError_t e,const char*why) { if(e!=cudaSuccess) throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e)); }
[[noreturn]] void fatal() { std::fprintf(stderr,"V11 fatal unchecked runtime retirement\n"); std::_Exit(2); }
struct Current {
    int device=-1;
    Current() { ck(cudaGetDevice(&device),"save runtime device"); }
    ~Current() { if(cudaSetDevice(device)!=cudaSuccess) fatal(); }
};
struct Memory {
    int device; size_t bytes; void*pointer=nullptr;
    Memory(int d,size_t n):device(d),bytes(n) { ck(cudaSetDevice(d),"allocate runtime device"); ck(cudaMalloc(&pointer,n),"bounded resident scratch"); }
    ~Memory() { if(pointer&&(cudaSetDevice(device)!=cudaSuccess||cudaFree(pointer)!=cudaSuccess)) fatal(); }
};
struct Arena {
    std::shared_ptr<const void>physical;
    std::vector<std::unique_ptr<Memory>>memory;
    std::array<cudaStream_t,4>streams{};
    std::array<uint64_t,4>bytes{};
    explicit Arena(std::shared_ptr<const void>p):physical(std::move(p)) {
        try {
            for(int d=0;d<4;++d) { ck(cudaSetDevice(d),"create source capture stream"); ck(cudaStreamCreateWithFlags(&streams[d],cudaStreamNonBlocking),"source capture stream"); }
        } catch(...) {
            for(int d=0;d<4;++d) if(streams[d] && (cudaSetDevice(d)!=cudaSuccess || cudaStreamDestroy(streams[d])!=cudaSuccess)) fatal();
            throw;
        }
    }
    template<class T>T*make(int d,size_t n) {
        auto b=std::make_unique<Memory>(d,n*sizeof(T)); auto*p=static_cast<T*>(b->pointer);
        bytes[d]+=b->bytes; memory.push_back(std::move(b)); return p;
    }
    ~Arena() {
        for(int d=0;d<4;++d) { if(cudaSetDevice(d)!=cudaSuccess||cudaStreamSynchronize(streams[d])!=cudaSuccess) fatal(); }
        memory.clear();
        for(int d=0;d<4;++d) { if(cudaSetDevice(d)!=cudaSuccess||cudaStreamDestroy(streams[d])!=cudaSuccess) fatal(); }
    }
};
struct Stage {
    StageView view;
    std::shared_ptr<Arena>arena;
    std::array<D::BankBuffers,4>banks{};
    int32_t*device_ids_v24=nullptr;
    std::vector<int32_t>host_ids_v24;
    bool device_complete_v24=false,ledger_ready_v24=false;
    uint64_t complete_frames_v24=0;
    R::Inputs*device_inputs=nullptr;
    std::vector<R::Inputs>host_inputs;
    D::NativeMask*masks=nullptr;
    D::LayerReceipt*receipts=nullptr;
    uint32_t*stage_error=nullptr;
    std::array<std::vector<std::unique_ptr<D::Dispatcher>>,9>layers;
    std::vector<std::shared_ptr<const void>>retained;
    uint64_t sequence=0;
    int T=0,accounted=0;
    bool running=false;
    Stage(StageView v,const Config&cfg):view(v),arena(std::make_shared<Arena>(cfg.physical_owner)),host_inputs(size_t(v.max_t)*24) {
        host_ids_v24.assign(24*80,-1);
        for(int d=0;d<4;++d) {
            auto&b=banks[d]; auto&c=b.cpu;
            c.plan=arena->make<R::Plan>(d,1);
            c.plan_error=arena->make<uint32_t>(d,1);
            c.physical=arena->make<B::PhysicalDeviceView>(d,1);
            c.input=arena->make<float>(d,8*2560);
            c.input_q=arena->make<uint8_t>(d,8*2720);
            c.gate_raw=arena->make<float>(d,80*640);
            c.up_raw=arena->make<float>(d,80*640);
            c.gate_scaled=arena->make<float>(d,80*640);
            c.up_scaled=arena->make<float>(d,80*640);
            c.hidden=arena->make<float>(d,80*640);
            c.hidden_q=arena->make<uint8_t>(d,80*680);
            c.down_raw=arena->make<float>(d,80*2560);
            c.output=arena->make<float>(d,80*2560);
            c.status=arena->make<G::Status>(d,1);
            b.compact=arena->make<cpu_compact_v8::WorkList>(d,1);
            b.native_q8=arena->make<uint8_t>(d,8*D::Qrow);
            b.native_plan=arena->make<swift_ep_v26::Plan>(d,1);
            b.native_scratch_bytes=N::native_expert_scratch_bytes(80,640);
            b.native_scratch=arena->make<uint8_t>(d,b.native_scratch_bytes);
            b.native_rows=arena->make<float>(d,80*2560);
            b.status=arena->make<D::BankStatus>(d,1);
            b.capture_stream=arena->streams[d];
            b.persistent_ctas=1; // Base policy; Compact is not promoted by this model bridge.
            ck(cudaSetDevice(d),"publish physical view device");
            ck(cudaMemcpyAsync(const_cast<B::PhysicalDeviceView*>(c.physical),&cfg.physical_views[d],sizeof(B::PhysicalDeviceView),cudaMemcpyHostToDevice,b.capture_stream),"publish leased physical view");
            ck(cudaStreamSynchronize(b.capture_stream),"physical view positive publication");
        }
        device_ids_v24=arena->make<int32_t>(v.device,size_t(v.max_t)*24*80);
        device_inputs=arena->make<R::Inputs>(v.device,host_inputs.size());
        masks=arena->make<D::NativeMask>(v.device,24);
        receipts=arena->make<D::LayerReceipt>(v.device,24);
        stage_error=arena->make<uint32_t>(v.device,1);
        for(int T=1;T<=v.max_t;++T) for(int l=0;l<24;++l) {
            auto&i=host_inputs[size_t(T-1)*24+l];
            i.physical=cfg.physical_views[v.device];
            i.logical={v.logical_res,cfg.logical_epoch}; i.ids=v.ids;
            i.expected_physical_epoch=i.physical.physical_epoch;
            i.expected_logical_epoch=cfg.logical_epoch; i.sequence=1; i.layer=v.lb+l; i.T=T;
        }
        ck(cudaSetDevice(v.device),"publish stage inputs");
        ck(cudaMemcpyAsync(device_inputs,host_inputs.data(),host_inputs.size()*sizeof(R::Inputs),cudaMemcpyHostToDevice,v.stream),"initial immutable layer input bindings");
        ck(cudaStreamSynchronize(v.stream),"initial layer bindings complete");
        for(int T=1;T<=v.max_t;++T) for(int l=0;l<24;++l) {
            D::OwnerView o{v.device,v.lb+l,T,v.stream,device_inputs+size_t(T-1)*24+l,v.original_x,v.native_q8,v.parts,v.native_hits,masks+l,receipts+l,stage_error};
            auto dispatch=std::make_unique<D::Dispatcher>(o,banks,arena,D::CpuPolicy::Base);
            retained.push_back(dispatch->retain());
            layers[T].push_back(std::move(dispatch));
        }
    }
    void close() {
        need(!running,"V11 stage still running at unregister");
        ck(cudaSetDevice(view.device),"retire original stage");
        ck(cudaStreamSynchronize(view.stream),"positive original stage retirement");
        // Caller has destroyed every original parent graph. Release its typed
        // retention tokens before closing the source graph dispatchers.
        retained.clear();
        for(auto&set:layers) { for(auto&d:set) d->close(); set.clear(); }
        std::fprintf(stderr,"V24_LEDGER_RETIRE owner=%d completed_frames=%llu parent_graphs_gone=1 original_stream_drained=1\n",view.device,(unsigned long long)complete_frames_v24);
        arena.reset();
    }
};
struct Runtime {
    Config cfg;
    std::vector<int32_t>original_r;
    std::array<std::unique_ptr<Stage>,2>stage;
    uint64_t next_sequence=0,window_sequence=0;
    int next_stage=0,window_T=0;
    bool active=false,poison=false;
};
std::map<const void*,StageView>registrations;
std::unique_ptr<Runtime>runtime;
std::atomic<bool>prepared_flag{false},active_flag{false},poison_flag{false};
Stage&get(const void*self) {
    need(runtime!=nullptr,"V11 resident runtime not prepared");
    for(auto&s:runtime->stage) if(s&&s->view.self==self) return *s;
    throw std::runtime_error("V11 unknown or retired Verifier");
}
void check_host_r() {
    auto&r=*runtime;
    need(r.cfg.host_res&&r.original_r.size()==r.cfg.host_res_count&&std::memcmp(r.original_r.data(),r.cfg.host_res,r.original_r.size()*4)==0,"V11 fixed0 original logical R changed");
}
void check_device_r(const StageView&v,const Config&cfg) {
    std::vector<int32_t>back(B::Keys);
    ck(cudaSetDevice(v.device),"logical R readback device");
    ck(cudaMemcpyAsync(back.data(),v.logical_res,back.size()*4,cudaMemcpyDeviceToHost,v.stream),"actual original R readback");
    ck(cudaStreamSynchronize(v.stream),"positive logical R readback");
    need(std::memcmp(back.data(),cfg.host_res,back.size()*4)==0,"V11 device logical R differs from exact profile");
}
void check_physical(const Config&cfg) {
    std::vector<B::PhysicalRef>canonical(B::Keys),back(B::Keys);
    uint64_t epoch=cfg.physical_views[0].physical_epoch;
    need(epoch!=0,"V11 missing physical epoch");
    for(int d=0;d<4;++d) {
        const auto&v=cfg.physical_views[d];
        need(v.refs&&v.physical_epoch==epoch,"V11 physical metadata epoch/view");
        ck(cudaSetDevice(d),"full physical coverage readback");
        ck(cudaMemcpy(back.data(),v.refs,back.size()*sizeof(B::PhysicalRef),cudaMemcpyDeviceToHost),"physical refs actual readback");
        if(d==0) canonical=back;
        else need(std::memcmp(back.data(),canonical.data(),back.size()*sizeof(B::PhysicalRef))==0,"V11 physical metadata mirrors differ");
        for(int b=0;b<4;++b) need(v.base[b]==cfg.physical_views[0].base[b]&&v.bank_bytes[b]==cfg.physical_views[0].bank_bytes[b],"V11 physical bank bases/extents disagree");
    }
    std::array<std::vector<bool>,4>used;
    const auto&v=cfg.physical_views[0];
    for(int d=0;d<4;++d) {
        need(v.base[d]&&v.bank_bytes[d]>=B::BlobBytes,"V11 missing physical bank");
        used[d].assign(size_t(v.bank_bytes[d]/B::SlotStride),false);
    }
    for(int key=0;key<B::Keys;++key) {
        const auto&ref=canonical[key];
        need(ref.key==uint32_t(key)&&ref.device>=0&&ref.device<4&&ref.physical_slot>=0&&ref.physical_epoch==epoch&&ref.blob_bytes==B::BlobBytes,"V11 incomplete physical expert union");
        const int d=ref.device; const size_t slot=size_t(ref.physical_slot);
        need(slot<used[d].size()&&!used[d][slot]&&ref.byte_offset==uint64_t(slot)*B::SlotStride&&ref.byte_offset<=v.bank_bytes[d]&&B::BlobBytes<=v.bank_bytes[d]-ref.byte_offset,"V11 duplicate/out-of-bounds physical expert");
        used[d][slot]=true;
    }
}
}
void register_stage(const StageView&v) {
    need(!prepared_flag.load()&&v.self&&v.device>=0&&v.device<2&&v.lb==v.device*24&&v.le==v.lb+24&&v.max_t>=2&&v.max_t<=8&&v.original_x&&v.native_q8&&v.ids&&v.parts&&v.native_hits&&v.logical_res&&v.stream,"V11 stage registration shape/lifetime");
    for(auto&entry:registrations) need(entry.first!=v.self&&entry.second.device!=v.device,"V11 duplicate stage registration");
    registrations.emplace(v.self,v);
}
void unregister_stage(const void*self) {
    Current restore;
    if(runtime) for(auto&s:runtime->stage) if(s&&s->view.self==self) {
        s->close(); s.reset();
    }
    registrations.erase(self);
}
void prepare(const Config&cfg) {
    Current restore;
    need(!runtime&&!prepared_flag.load()&&registrations.size()==2&&cfg.physical_owner&&cfg.host_res&&cfg.host_res_count==B::Keys&&cfg.logical_epoch&&cfg.adapt_swaps==0,"V11 requires complete fixed0 original profile and both stages");
    require_protocol_v24();
    cpu_fenv_admission_v11::require_process();
    std::string err; const bool peers=strata_peer_default::require_all(4,err); need(peers,err.c_str());
    for(int key=0;key<B::Keys;++key) need(cfg.host_res[key]>=-1,"V11 invalid original logical R value");
    check_physical(cfg);
    auto next=std::make_unique<Runtime>(); next->cfg=cfg;
    next->original_r.assign(cfg.host_res,cfg.host_res+cfg.host_res_count);
    for(const auto&entry:registrations) {
        check_device_r(entry.second,cfg);
        next->stage[entry.second.device]=std::make_unique<Stage>(entry.second,cfg);
    }
    runtime=std::move(next); prepared_flag.store(true,std::memory_order_release);
    std::fprintf(stderr,"V24_RESIDENT_PREPARED no_layer_host_handshake=1 ledger_capacity_per_owner=61440 full_unique_experts=24576 stages=2 banks=4 fixed0=1 native_and_CPU_sources=all supported_T=1..max active=0\n");
}
void activate() {
    Current restore;
    need(runtime&&prepared_flag.load()&&!runtime->active&&!runtime->poison,"V11 activate state");
    cpu_fenv_admission_v11::require_process(); check_host_r();
    for(auto&s:runtime->stage) { need(bool(s),"V11 incomplete stage set"); check_device_r(s->view,runtime->cfg); }
    runtime->active=true; active_flag.store(true,std::memory_order_release);
}
void deactivate() {
    need(runtime!=nullptr,"V11 deactivate without preparation");
    for(auto&s:runtime->stage) if(s) need(!s->running,"V11 deactivate while stage running");
    runtime->active=false; active_flag.store(false,std::memory_order_release);
}
void close() {
    need(registrations.empty(),"V11 close before individual Verifier graph/stream retirement");
    active_flag.store(false); prepared_flag.store(false);
    runtime.reset();
}
bool prepared()noexcept { return prepared_flag.load(std::memory_order_acquire); }
bool active(const void*self)noexcept { return active_flag.load(std::memory_order_acquire)&&!poison_flag.load()&&registrations.count(self)!=0; }
bool active_layer(int layer)noexcept { return active_flag.load(std::memory_order_acquire)&&!poison_flag.load()&&layer>=0&&layer<48; }
void begin_stage(const void*self,int T) {
    Current restore;
    auto&s=get(self); auto&r=*runtime;
    need(active(self)&&!s.running&&s.view.device==r.next_stage&&T>=1&&T<=s.view.max_t,"V11 stage order/shape/active gate");
    cpu_fenv_admission_v11::require_ready(); check_host_r();
    if(s.view.device==0) { need(r.next_sequence!=UINT64_MAX,"V11 epoch exhaustion"); r.window_sequence=++r.next_sequence; r.window_T=T; }
    else need(T==r.window_T,"V11 dense stages disagree on T");
    s.sequence=r.window_sequence; s.T=T; s.accounted=0;
    s.device_complete_v24=false;s.ledger_ready_v24=false;
    auto*host=s.host_inputs.data()+size_t(T-1)*24;
    for(int l=0;l<24;++l) host[l].sequence=s.sequence;
    ck(cudaSetDevice(s.view.device),"stage input publication");
    // Stable host storage lives through finish; same stream orders H2D before
    // the ONE subsequent verify graph launch. No pageable-return assumption.
    ck(cudaMemcpyAsync(s.device_inputs+size_t(T-1)*24,host,24*sizeof(R::Inputs),cudaMemcpyHostToDevice,s.view.stream),"one bounded current-stage input publication");
    // Missing ledger copies must remain invalid, not silently reuse prior IDs.
    ck(cudaMemsetAsync(s.device_ids_v24+size_t(T-1)*24*80,0xff,24*80*sizeof(int32_t),s.view.stream),"V24 current ledger sentinel publication");
    for(auto&d:s.layers[T]) d->begin(s.sequence);
    s.running=true;
}
Rows capture_pre(const void*self,int layer,int T) {
    auto&s=get(self); need(active(self)&&T>=1&&T<=s.view.max_t&&layer>=s.view.lb&&layer<s.view.le,"V11 pre capture shape");
    if(layer==s.view.lb) ck(D::reset_stage_error(s.stage_error,s.view.stream),"stage error entry reset");
    auto rows=s.layers[T][size_t(layer-s.view.lb)]->capture_pre();
    // Original IDs are still live here. This node joins the complete original
    // pre/plan frontier before any next layer can overwrite the router buffer.
    auto*to=s.device_ids_v24+(size_t(T-1)*24+size_t(layer-s.view.lb))*80;
    ck(cudaMemcpyAsync(to,s.view.ids,size_t(T)*10*sizeof(int32_t),cudaMemcpyDeviceToDevice,s.view.stream),"V24 original route ledger capture");
    return {rows.dst,rows.count,rows.error};
}
Rows capture_post(const void*self,int layer,int T) {
    auto&s=get(self); need(active(self)&&T>=1&&T<=s.view.max_t&&layer>=s.view.lb&&layer<s.view.le,"V11 post capture shape");
    auto rows=s.layers[T][size_t(layer-s.view.lb)]->capture_post(); return {rows.dst,rows.count,rows.error};
}
void capture_cpu_rows(const void*self,int layer,int T) {
    auto&s=get(self); need(active(self)&&T>=1&&T<=s.view.max_t&&layer>=s.view.lb&&layer<s.view.le,"V11 CPU gather capture shape");
    s.layers[T][size_t(layer-s.view.lb)]->capture_cpu_rows();
}
void cancel_protocol_v24(const char*reason)noexcept {
    // Watchdog/error thread may request cancellation while launch is pending.
    // Publish only the existing atomic poison; never touch mutable stage state.
    poison_flag.store(true,std::memory_order_release);
    std::fprintf(stderr,"V24_CANCEL %s\n",reason?reason:"external release");
}
void require_protocol_v24() {
    const char*p=std::getenv("STRATA_RESIDENT_PROTOCOL_V24");
    need(p&&!std::strcmp(p,"1"),"V24 explicit resident protocol gate required");
    const char*a=std::getenv("STRATA_VERIFY_ASYNC_LAUNCH");
    need(!a||!std::strcmp(a,"0"),"V24 isolated synchronous launch policy; async/pin are separate candidates");
}
bool complete_device_stage_v24(const void*self,std::string&error) {
    Current restore;auto&s=get(self);
    need(active(self)&&s.running&&!s.device_complete_v24&&!s.ledger_ready_v24&&s.accounted==0,"V24 device-completion phase refused");
    std::vector<D::Dispatcher*>all;for(auto&d:s.layers[s.T])all.push_back(d.get());
    const bool device_ok=D::Dispatcher::finish_all(all,error);
    s.device_complete_v24=true;
    if(!device_ok||poison_flag.load()) {runtime->poison=true;poison_flag.store(true);if(error.empty())error="V24 GPU receipt/epoch failure";return false;}
    ck(cudaSetDevice(s.view.device),"V24 ledger readback owner");
    ck(cudaMemcpyAsync(s.host_ids_v24.data(),s.device_ids_v24+size_t(s.T-1)*24*80,24*80*sizeof(int32_t),cudaMemcpyDeviceToHost,s.view.stream),"V24 one complete ledger readback");
    ck(cudaStreamSynchronize(s.view.stream),"V24 positive original ledger readback");
    for(int layer=0;layer<24;++layer)for(int row=0;row<s.T*10;++row) {
        const int id=s.host_ids_v24[size_t(layer)*80+row];
        if(id<0||id>=512){runtime->poison=true;poison_flag.store(true);error="V24 incomplete or invalid original route ledger";return false;}
    }
    check_host_r();s.ledger_ready_v24=true;++s.complete_frames_v24;
    if(s.complete_frames_v24==1)std::fprintf(stderr,"V24_LEDGER_COMPLETE owner=%d layers=24 current_T=%d receipt_checks=24 D2H_bytes=7680 positive_completion=1 before_accounting=1\n",s.view.device,s.T);
    return true;
}
const int32_t*completed_layer_ids_v24(const void*self,int layer,int T) {
    auto&s=get(self);need(active(self)&&s.running&&s.device_complete_v24&&s.ledger_ready_v24&&T==s.T&&
        layer==s.view.lb+s.accounted&&s.accounted>=0&&s.accounted<24,"V24 completed ledger row order refused");
    return s.host_ids_v24.data()+size_t(s.accounted)*80;
}
void require_deferred_accounting_v24(int layer,int T,const int32_t*ids) {
    need(runtime&&layer>=0&&layer<48&&runtime->stage[size_t(layer/24)],"V24 deferred accounting stage");
    auto&s=*runtime->stage[size_t(layer/24)];
    need(ids==completed_layer_ids_v24(s.view.self,layer,T),"V24 accounting requires positively completed original ID ledger");
}
bool finish_stage(const void*self,std::string&error) {
    auto&s=get(self);need(s.running&&s.device_complete_v24&&s.ledger_ready_v24,"V24 finish before complete GPU ledger");
    const bool ok=s.accounted==24&&!poison_flag.load();s.running=false;
    if(!ok){runtime->poison=true;poison_flag.store(true);if(error.empty())error="V24 original deferred accounting refused";return false;}
    runtime->next_stage=(s.view.device+1)%2;
    if(s.complete_frames_v24==1)std::fprintf(stderr,"V24_ACCOUNTED owner=%d original_order_layers=24 current_T=%d native_and_CPU_R_unchanged=1\n",s.view.device,s.T);
    return true;
}
void abort_stage(const void*self) {
    auto&s=get(self); poison_flag.store(true); runtime->poison=true;
    if(s.running) { for(auto&d:s.layers[s.T]) d->abort(); s.running=false; }
}
void require_accounting(int layer,int T,int K,const int32_t*host_res,const int32_t*ids) {
    need(active_layer(layer)&&runtime&&host_res==runtime->cfg.host_res&&ids&&K==10&&T>=1&&T<=8,"V11 logical accounting binding");
    auto&s=*runtime->stage[size_t(layer/24)];
    need(s.running&&s.T==T&&layer==s.view.lb+s.accounted,"V11 original callback layer order");
    require_deferred_accounting_v24(layer,T,ids);
    cpu_fenv_admission_v11::require_ready();
    for(int row=0;row<T*K;++row) need(ids[row]>=0&&ids[row]<512,"V11 routed expert outside original IDs");
    ++s.accounted;
}
void fail_accounting(const char*reason)noexcept { poison_flag.store(true); std::fprintf(stderr,"V11_ACCOUNTING_REFUSED %s\n",reason?reason:"unknown"); }
uint64_t planned_device_bytes(int device) {
    need(device>=0&&device<4,"V11 buffer-budget device");
    const uint64_t bank=sizeof(R::Plan)+4+sizeof(B::PhysicalDeviceView)+8ull*2560*4+8ull*2720+5ull*80*640*4+80ull*680+2ull*80*2560*4+sizeof(G::Status)+sizeof(cpu_compact_v8::WorkList)+8ull*D::Qrow+sizeof(swift_ep_v26::Plan)+N::native_expert_scratch_bytes(80,640)+80ull*2560*4+sizeof(D::BankStatus);
    return 2*bank+(device<2?8ull*24*80*sizeof(int32_t)+8ull*24*sizeof(R::Inputs)+24*sizeof(D::NativeMask)+24*sizeof(D::LayerReceipt)+4:0);
}
uint64_t device_bytes(int device) {
    need(device>=0&&device<4&&runtime,"V11 actual buffer-budget state"); uint64_t total=0;
    for(auto&s:runtime->stage) if(s&&s->arena) total+=s->arena->bytes[device]; return total;
}
}
