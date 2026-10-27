// Three real vocabulary shards. Original Q6_K/Q8_1 math; no unsplit-head retry.
#include "run_devices.hpp"
#include "head_tp_v200.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>
namespace head_tp_v200 {
namespace K=strata::kernels;
namespace {
constexpr int N=2560,V=248320,MT=8,OwnerDevice=2;
constexpr size_t RowBytes=2100,Spare=512ull<<20;
constexpr std::array<int,4> Starts={0,82688,165504,V};
constexpr int rows(int d){return Starts[d+1]-Starts[d];}
static_assert(rows(0)+rows(1)+rows(2)==V);
static_assert((Starts[1]*RowBytes)%256==0&&(Starts[2]*RowBytes)%256==0);
std::atomic<Owner*>active{nullptr};
std::atomic<const strata::core::Verifier*>bound{nullptr};
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
void ck(cudaError_t e,const char*s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
void device(int d){ck(cudaSetDevice(d),"head3 device");}
struct Restore{int d=-1;Restore(){ck(cudaGetDevice(&d),"head3 save device");}~Restore(){if(d>=0&&cudaSetDevice(d)!=cudaSuccess)std::_Exit(2);}};
void project(const void*w,const void*q,void*il,float*y,int r,int t,cudaStream_t s){
    // Select the actual native algorithm for this T, preserving its per-row class.
    need(K::native_mmvq_il_supported(14,t,r)==K::native_mmvq_il_supported(14,t,V),"head3 original IL admission differs");
    need(K::native_mmvq_il_rows_for(89,14,t,r)==K::native_mmvq_il_rows_for(89,14,t,V),"head3 original row class differs");
    if(K::native_mmvq_il_supported(14,t,r)){
        if(t>=2)K::native_q8_1_interleave(q,il,N,t,s);
        K::native_mmvq_il(14,w,q,il,y,N,r,t,s);
    }else K::native_mmvq(14,w,q,y,N,r,t,s);
    ck(cudaGetLastError(),"head3 unchanged Q6_K projection");
}
struct Graph{
    int d;cudaStream_t s;cudaGraphExec_t exec=nullptr;
    Graph(int dev,cudaStream_t stream):d(dev),s(stream){}
    ~Graph(){if(exec){cudaSetDevice(d);cudaGraphExecDestroy(exec);}}
    void capture(const std::function<void()>&fn,int mmvq_expected,int copies_expected,int record_expected,int wait_expected){
        device(d);cudaGraph_t g=nullptr;ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"head3 capture begin");
        try{fn();ck(cudaGetLastError(),"head3 capture body");ck(cudaStreamEndCapture(s,&g),"head3 capture end");}
        catch(...){cudaStreamEndCapture(s,&g);if(g)cudaGraphDestroy(g);throw;}
        try{
            size_t n=0;ck(cudaGraphGetNodes(g,nullptr,&n),"head3 node count");std::vector<cudaGraphNode_t>ns(n);ck(cudaGraphGetNodes(g,ns.data(),&n),"head3 nodes");
            int mm=0,cp=0,rec=0,wait=0;
            for(auto node:ns){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type),"head3 node type");
                cp+=type==cudaGraphNodeTypeMemcpy;rec+=type==cudaGraphNodeTypeEventRecord;wait+=type==cudaGraphNodeTypeWaitEvent;
                if(type==cudaGraphNodeTypeKernel){cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p),"head3 kernel params");const char*name=nullptr;ck(cudaFuncGetName(&name,p.func),"head3 kernel name");need(name,"head3 named kernel");
                    mm+=bool(std::strstr(name,"native_mmvq_multi_kernel")||std::strstr(name,"native_mmvq_il_kernel")||std::strstr(name,"native_q6_k_mmvq_kernel"));}
            }
            need(mm==mmvq_expected&&cp==copies_expected&&rec==record_expected&&wait==wait_expected,"head3 actual graph DMA/kernel/event contract");
            ck(cudaGraphInstantiate(&exec,g,0),"head3 instantiate");
        }catch(...){cudaGraphDestroy(g);throw;}
        ck(cudaGraphDestroy(g),"head3 destroy source graph");
    }
    void launch(){device(d);ck(cudaGraphLaunch(exec,s),"head3 replay");}
};
struct PerT{
    std::unique_ptr<Graph>producer,tail;std::array<std::unique_ptr<Graph>,2>remote;
    cudaEvent_t ready=nullptr;std::array<cudaEvent_t,2>done{};bool prepared=false;unsigned cuts=0;uint64_t calls=0;
    void close()noexcept{
        tail.reset();for(auto&g:remote)g.reset();producer.reset();
        if(ready){cudaSetDevice(OwnerDevice);cudaEventDestroy(ready);ready=nullptr;}
        for(int d=0;d<2;++d)if(done[d]){cudaSetDevice(d);cudaEventDestroy(done[d]);done[d]=nullptr;}
        prepared=false;
    }
    ~PerT(){close();}
};
}
struct Owner::Impl{
    const strata::core::Verifier*ver;const strata::core::NativeHead*head;cudaStream_t owner;
    std::array<cudaStream_t,2>streams{};
    std::array<void*,2>weights{},q{},il{},result{};void*local=nullptr;
    std::array<PerT,9>graph{};View base{};bool ready=false,failed=false;uint64_t windows=0;
    Impl(const strata::core::Verifier*v,const strata::core::NativeHead*h,cudaStream_t s):ver(v),head(h),owner(s){}
    void drain()noexcept{
        // Owner tail depends on remotes; all queued work belongs to this immutable binding.
        for(int d=0;d<2;++d)if(streams[d]){if(cudaSetDevice(d)!=cudaSuccess||cudaStreamSynchronize(streams[d])!=cudaSuccess)std::_Exit(2);}
        if(owner&&(cudaSetDevice(OwnerDevice)!=cudaSuccess||cudaStreamSynchronize(owner)!=cudaSuccess))std::_Exit(2);
    }
    ~Impl(){
        drain();for(auto&g:graph)g.close();
        for(int d=0;d<2;++d){cudaSetDevice(d);for(void*p:{weights[d],q[d],il[d],result[d]})if(p)cudaFree(p);if(streams[d])cudaStreamDestroy(streams[d]);}
        cudaSetDevice(OwnerDevice);if(local)cudaFree(local);
    }
    void setup(){
        std::string e;need(threegpu_scope::count()==3&&threegpu_scope::hardware(e),e.empty()?"head3 current three-card scope":e.c_str());
        need(ver&&head&&owner&&head->loaded()&&head->device()==OwnerDevice&&head->type()==14&&head->row_bytes()==RowBytes&&head->weight_bytes()==V*RowBytes,"head3 exact owning native Q6_K head");
        need(K::native_mmvq_multi_exact()&&K::argmax_rows_wanted()&&!std::getenv("CUDA_LAUNCH_BLOCKING"),"head3 selected numeric/nonblocking contract");
        for(int d=0;d<3;++d){device(d);size_t f=0,t=0;ck(cudaMemGetInfo(&f,&t),"head3 budget");
            const size_t required=d<2?rows(d)*RowBytes+K::native_q8_1_bytes(N,MT)+K::native_q8_1_il_bytes(N,MT)+rows(d)*MT*4ull:rows(2)*MT*4ull;
            need(f>=required+Spare,"head3 complete budget including512MiB reserve");}
        for(int d=0;d<2;++d){
            for(auto pair:{std::array<int,2>{d,OwnerDevice},std::array<int,2>{OwnerDevice,d}}){
                device(pair[0]);int yes=0;ck(cudaDeviceCanAccessPeer(&yes,pair[0],pair[1]),"head3 P2P capability");need(yes,"head3 direct P2P required; no host-staging fallback");
                auto rc=cudaDeviceEnablePeerAccess(pair[1],0);if(rc==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(rc,"head3 P2P enable");
            }
            device(d);ck(cudaStreamCreateWithFlags(&streams[d],cudaStreamNonBlocking),"head3 remote stream");
            ck(cudaMalloc(&weights[d],rows(d)*RowBytes),"head3 raw shard weights");ck(cudaMalloc(&q[d],K::native_q8_1_bytes(N,MT)),"head3 Q8 scratch");
            ck(cudaMalloc(&il[d],K::native_q8_1_il_bytes(N,MT)),"head3 IL scratch");ck(cudaMalloc(&result[d],rows(d)*MT*4ull),"head3 remote FP32 scratch");
            ck(cudaMemcpyPeerAsync(weights[d],d,(const unsigned char*)head->weights()+Starts[d]*RowBytes,OwnerDevice,rows(d)*RowBytes,streams[d]),"head3 raw startup P2P");
            ck(cudaStreamSynchronize(streams[d]),"head3 shard startup complete");
        }
        device(OwnerDevice);ck(cudaMalloc(&local,rows(2)*MT*4ull),"head3 local FP32 scratch");
        std::fprintf(stderr,"HEAD3_SETUP current_ordinals=0,1,2 shards=%d/%d/%d raw_extra_bytes=%zu no_weight_requant=1 host_mailboxes=0\n",rows(0),rows(1),rows(2),(rows(0)+rows(1))*RowBytes);
    }
    bool same(const View&v)const{return v.verifier==ver&&v.head==head&&v.device==OwnerDevice&&v.capacity>=2&&v.capacity<=MT&&v.T>=1&&v.T<=v.capacity&&v.stream==owner&&v.mixed&&v.q8&&v.il&&v.logits&&v.arg&&v.out&&(!v.nan||v.nf)&&!v.batch&&!v.split&&!v.profile&&!v.prelaunched;}
    void produce(const View&v,int t,PerT&g,bool capture){
        K::native_quantize_q8_1(v.mixed,v.q8,N,t,owner);
        ck(cudaEventRecordWithFlags(g.ready,owner,capture?cudaEventRecordExternal:cudaEventRecordDefault),"head3 input-ready");
        project((const unsigned char*)head->weights()+Starts[2]*RowBytes,v.q8,v.il,(float*)local,rows(2),t,owner);
        ck(cudaMemcpy2DAsync(v.logits+Starts[2],V*4,local,rows(2)*4,rows(2)*4,t,cudaMemcpyDeviceToDevice,owner),"head3 owner full-stride assembly");
    }
    void remote(const View&v,int t,int d,PerT&g,bool capture){
        device(d);auto s=streams[d];ck(cudaStreamWaitEvent(s,g.ready,capture?cudaEventWaitExternal:cudaEventWaitDefault),"head3 input dependency");
        ck(cudaMemcpyAsync(q[d],v.q8,K::native_q8_1_bytes(N,t),cudaMemcpyDeviceToDevice,s),"head3 direct raw Q8 input");
        project(weights[d],q[d],il[d],(float*)result[d],rows(d),t,s);
        for(int row=0;row<t;++row)ck(cudaMemcpyAsync(v.logits+row*V+Starts[d],(float*)result[d]+row*rows(d),rows(d)*4,cudaMemcpyDeviceToDevice,s),"head3 direct FP32 shard result");
        ck(cudaEventRecordWithFlags(g.done[d],s,capture?cudaEventRecordExternal:cudaEventRecordDefault),"head3 shard ready");
    }
    void tail(const View&v,int t,PerT&g,bool capture){
        device(OwnerDevice);for(int d=0;d<2;++d)ck(cudaStreamWaitEvent(owner,g.done[d],capture?cudaEventWaitExternal:cudaEventWaitDefault),"head3 result dependency");
        K::argmax_rows(v.logits,t,V,v.arg,v.out,owner);if(v.nan)K::logits_nonfinite_rows(v.logits,t,V,v.nf,owner);
    }
    void prepare_all(const View&v){
        need(same(v),"head3 immutable owning view");base=v;device(OwnerDevice);ck(cudaMemsetAsync(v.mixed,0,N*v.capacity*4,owner),"head3 finite warmup input");
        for(int t=1;t<=v.capacity;++t){
            auto&g=graph[t];device(OwnerDevice);ck(cudaEventCreateWithFlags(&g.ready,cudaEventDisableTiming),"head3 ready event");
            for(int d=0;d<2;++d){device(d);ck(cudaEventCreateWithFlags(&g.done[d],cudaEventDisableTiming),"head3 done event");}
            device(OwnerDevice);produce(v,t,g,false);for(int d=0;d<2;++d)remote(v,t,d,g,false);tail(v,t,g,false);drain();
            g.producer=std::make_unique<Graph>(OwnerDevice,owner);g.producer->capture([&]{produce(v,t,g,true);},1,1,1,0);
            for(int d=0;d<2;++d){g.remote[d]=std::make_unique<Graph>(d,streams[d]);g.remote[d]->capture([&,d]{remote(v,t,d,g,true);},1,t+1,1,1);}
            g.tail=std::make_unique<Graph>(OwnerDevice,owner);g.tail->capture([&]{tail(v,t,g,true);},0,0,0,2);g.prepared=true;
        }
        drain();ready=true;std::fprintf(stderr,"HEAD3_READY T1_%d=1 parallel_shards=3 direct_P2P=1 fallback=0\n",v.capacity);
    }
};
bool requested(){const char*v=std::getenv("STRATA_HEAD_TP_V200");need(!v||!std::strcmp(v,"1"),"head3 mandatory head split");return true;}
Owner::Owner(std::unique_ptr<Impl>p):p_(std::move(p)){}
Owner::~Owner(){Restore restore;Owner*self=this;active.compare_exchange_strong(self,nullptr);auto*v=p_->ver;bound.compare_exchange_strong(v,nullptr);p_->drain();std::fprintf(stderr,"HEAD3_CLOSE windows=%llu T7_legacy=0\n",(unsigned long long)p_->windows);p_.reset();}
std::unique_ptr<Owner> Owner::create(const strata::core::Verifier*v,const strata::core::NativeHead*h,cudaStream_t s,std::string&e){
    Restore restore;try{need(!active.load()&&!bound.load(),"head3 unique immutable binding");auto p=std::make_unique<Impl>(v,h,s);p->setup();auto o=std::unique_ptr<Owner>(new Owner(std::move(p)));bound.store(v);active.store(o.get());return o;}catch(const std::exception&x){e=x.what();return nullptr;}
}
bool Owner::prepare(const View&v,std::string&e){Restore restore;try{need(!p_->failed&&p_->same(v),"head3 owning view refused");if(!p_->ready)p_->prepare_all(v);else need(v.mixed==p_->base.mixed&&v.q8==p_->base.q8&&v.il==p_->base.il&&v.logits==p_->base.logits&&v.arg==p_->base.arg&&v.out==p_->base.out&&v.nf==p_->base.nf&&v.nan==p_->base.nan&&v.capacity==p_->base.capacity,"head3 capture binding changed");return true;}catch(const std::exception&x){e=x.what();p_->failed=true;p_->drain();return false;}}
bool Owner::cut(int t,bool ar,bool batch,bool split,bool profile){
    need(!batch&&!split&&!profile&&t>=1&&t<=MT&&p_->ready&&!p_->failed&&p_->graph[t].prepared,"head3 mandatory cut refused; unsplit retry forbidden");
    p_->graph[t].cuts|=1u<<unsigned(ar);return true;
}
bool Owner::finish(const View&v,bool ar,std::string&e){Restore restore;try{
    need(!p_->failed&&p_->same(v),"head3 finish view refused");auto&g=p_->graph[v.T];need(p_->ready&&g.prepared&&(g.cuts&(1u<<unsigned(ar))),"head3 cut/graph epoch missing");
    g.producer->launch();for(int d=0;d<2;++d)g.remote[d]->launch();g.tail->launch();device(OwnerDevice);ck(cudaStreamSynchronize(p_->owner),"head3 complete full logits/picks");
    ++g.calls;++p_->windows;if(g.calls==1)std::fprintf(stderr,"HEAD3_ADMITTED T%d simultaneous_shards=3 direct_input_bytes=%zu complete_FP32_return=%zu fallback=0\n",v.T,2*K::native_q8_1_bytes(N,v.T),(rows(0)+rows(1))*v.T*4ull);return true;
}catch(const std::exception&x){e=x.what();p_->failed=true;p_->drain();return false;}}
bool prepare(const View&v,std::string&e){auto*p=active.load();return !p||bound.load()!=v.verifier||p->prepare(v,e);}
bool cut(const strata::core::Verifier*v,int t,bool ar,bool b,bool s,bool profile){auto*p=active.load();return p&&bound.load()==v&&p->cut(t,ar,b,s,profile);}
bool finish(const View&v,bool ar,std::string&e){auto*p=active.load();return !p||bound.load()!=v.verifier||p->finish(v,ar,e);}
bool reject_nonordinary(const strata::core::Verifier*v,const char*api,std::string&e){if(active.load()&&bound.load()==v){e=std::string("head3 owning ordinary API required: ")+api;return true;}return false;}
}
