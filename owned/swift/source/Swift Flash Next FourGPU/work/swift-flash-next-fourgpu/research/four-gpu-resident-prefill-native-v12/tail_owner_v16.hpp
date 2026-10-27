#pragma once
// Private production integration source; Root build/device/model proof pending.
#include "packet/tail_api.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace swift_tail_v16 {
class Context {
    using Geometry=swift_tail_r027::Geometry;
    struct Entry { bool valid=false; uint64_t age=0; swift_tail_r027::HostPlan plan; };
    struct Upload { unsigned char* source=nullptr; cudaEvent_t credit=nullptr; bool used=false; };
    int owner_=-1,max_rows_=0,next_upload_=0;
    cudaStream_t stream_=nullptr;
    swift_tail_r027::LaunchProfile profile_;
    std::array<Entry,8> cache_;
    std::vector<uint8_t> terminal_;
    std::array<Upload,2> upload_;
    uint64_t clock_=0,epoch_=1;
    size_t max_tiles_=0,max_previous_=0,partial_bytes_=0,upload_bytes_=0;
    float* partial_=nullptr;
    swift_tail_r027::TileSpan* spans_=nullptr;
    int32_t* previous_=nullptr;
    bool pending_=false,poisoned_=false;
    static size_t align16(size_t n){return (n+15)&~size_t(15);}
    static void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
    static void check(cudaError_t e,const char* text){if(e!=cudaSuccess)throw std::runtime_error(std::string(text)+": "+cudaGetErrorString(e));}
    [[noreturn]] static void fatal(const char* text,cudaError_t e)noexcept{
        std::fprintf(stderr,"R027 V16 unsafe retirement refused: %s: %s\n",text,cudaGetErrorString(e));
        std::fflush(stderr);std::_Exit(2);
    }
    void issuing_owner()const{
        int current=-1;check(cudaGetDevice(&current),"issuing device");
        need(current==owner_&&stream_,"R027 V16 owner/stream mismatch");
    }
    void outside_capture()const{
        cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
        check(cudaStreamIsCapturing(stream_,&status),"capture state");
        need(status==cudaStreamCaptureStatusNone,
             "R027 V16 dynamic host-routed Prefill must prepare/submit outside capture");
    }
    static bool same(const Geometry&a,const Geometry&b){
        return a.owner==b.owner&&a.J==b.J&&a.groups==b.groups&&a.ntx==b.ntx&&
            a.tiles==b.tiles&&a.grid==b.grid&&a.k_blocks==b.k_blocks&&
            a.blocks_per_iter==b.blocks_per_iter&&a.max_rows==b.max_rows&&
            a.total_rows==b.total_rows&&a.ld_dst==b.ld_dst;
    }
    // Literal original make_plan traversal into already reserved vectors. No
    // host or device allocation on a miss; the bounded cache is HOST metadata.
    const swift_tail_r027::HostPlan& plan(const Geometry&g){
        need(clock_!=UINT64_MAX,"R027 V16 plan epoch exhausted");++clock_;
        for(auto&e:cache_)if(e.valid&&same(e.plan.geometry,g)){e.age=clock_;return e.plan;}
        auto* e=&cache_[0];
        for(auto&x:cache_)if(!x.valid){e=&x;break;}else if(x.age<e->age)e=&x;
        need(g.J>=8&&g.J<=128&&g.groups>=1&&g.groups<=16&&g.ntx>=1&&
             g.tiles==int64_t(10)*g.groups*g.ntx&&g.grid>=1&&g.k_blocks==40&&
             g.blocks_per_iter==4&&int64_t(g.tiles)*40<(int64_t(1)<<30)&&
             size_t(g.tiles)<=max_tiles_,"R027 V16 original plan bounds");
        auto&p=e->plan;p.geometry=g;p.spans.resize(g.tiles);
        std::fill(p.spans.begin(),p.spans.end(),swift_tail_r027::TileSpan{});
        p.previous_blocks.clear();terminal_.resize(g.tiles);std::fill(terminal_.begin(),terminal_.end(),0);
        for(int b0=0;b0<g.grid;++b0){
            const int64_t start=swift_tail_r027::range_edge(g,b0),stop=swift_tail_r027::range_edge(g,b0+1);
            if(start==stop||start%40==0||(start/40==stop/40&&stop%40!=0))continue;
            const int tile=int(start/40);
            need(tile>=0&&tile<g.tiles&&!terminal_[tile]++,"R027 V16 terminal fixup owner");
            auto&span=p.spans[tile];span.offset=uint32_t(p.previous_blocks.size());
            int b=b0-1;int64_t end=start;
            for(;;){
                need(b>=0,"R027 V16 preceding partial missing");
                const int64_t begin=swift_tail_r027::range_edge(g,b);
                if(begin==end){--b;end=begin;continue;}
                need(p.previous_blocks.size()<max_previous_,"R027 V16 previous-block capacity");
                p.previous_blocks.push_back(b);++span.count;
                if(begin%40==0||begin/40<start/40)break;
                --b;end=begin;
            }
        }
        need(p.previous_blocks.size()<=size_t(g.grid),"R027 V16 incomplete tail bound");
        e->valid=true;e->age=clock_;return p;
    }
    void upload(const swift_tail_r027::HostPlan&p){
        auto&u=upload_[next_upload_];next_upload_^=1;
        if(u.used)check(cudaEventSynchronize(u.credit),"pinned descriptor source credit");
        const size_t bytes=p.spans.size()*sizeof(swift_tail_r027::TileSpan),off=align16(bytes);
        const size_t prev=p.previous_blocks.size()*sizeof(int32_t);
        need(off+prev<=upload_bytes_,"R027 V16 pinned descriptor capacity");
        std::memcpy(u.source,p.spans.data(),bytes);
        if(prev)std::memcpy(u.source+off,p.previous_blocks.data(),prev);
        pending_=true; // BEFORE first uncertain enqueue, including a later event-record failure
        check(cudaMemcpyAsync(spans_,u.source,bytes,cudaMemcpyHostToDevice,stream_),"span upload");
        if(prev)check(cudaMemcpyAsync(previous_,u.source+off,prev,cudaMemcpyHostToDevice,stream_),"previous-block upload");
        check(cudaEventRecord(u.credit,stream_),"pinned source copy completion credit");u.used=true;
    }
    void close_noexcept()noexcept{
        if(owner_<0)return;int previous=-1;auto e=cudaGetDevice(&previous);
        if(e==cudaSuccess)e=cudaSetDevice(owner_);
        if(e==cudaSuccess&&stream_)e=cudaStreamSynchronize(stream_);
        if(e!=cudaSuccess)fatal("owner stream drain",e);
        pending_=false;
        for(auto&u:upload_){
            if(u.credit){e=cudaEventDestroy(u.credit);if(e!=cudaSuccess)fatal("credit destroy",e);u.credit=nullptr;}
            if(u.source){e=cudaFreeHost(u.source);if(e!=cudaSuccess)fatal("descriptor free",e);u.source=nullptr;}
        }
        e=cudaSetDevice(previous);if(e!=cudaSuccess)fatal("restore caller",e);
        owner_=-1; // borrowed partial/descriptors are never freed here
    }
public:
    Context(int owner,cudaStream_t stream,int max_rows):max_rows_(max_rows),stream_(stream){
        try{
            need(owner>=0&&owner<2&&stream&&max_rows>=1&&max_rows<=32768,"R027 V16 owner/max T");
            int current=-1;check(cudaGetDevice(&current),"initial owner");
            need(current==owner,"R027 V16 initial owner mismatch");owner_=owner;
            issuing_owner();outside_capture();profile_=swift_tail_r027::prepare_launch_profile();
            need(profile_.owner==owner_,"R027 V16 prepared owner mismatch");
            max_tiles_=size_t(160)*size_t((max_rows+profile_.largest_J-1)/profile_.largest_J);
            max_previous_=size_t(profile_.sm_count);
            partial_bytes_=max_previous_*128*128*sizeof(float);
            upload_bytes_=align16(max_tiles_*sizeof(swift_tail_r027::TileSpan))+max_previous_*sizeof(int32_t);
            terminal_.reserve(max_tiles_);
            for(auto&e:cache_){e.plan.spans.reserve(max_tiles_);e.plan.previous_blocks.reserve(max_previous_);}
            for(auto&u:upload_){
                check(cudaHostAlloc(reinterpret_cast<void**>(&u.source),upload_bytes_,cudaHostAllocPortable),"pinned descriptor slot");
                check(cudaEventCreateWithFlags(&u.credit,cudaEventDisableTiming),"descriptor copy event");
            }
        }catch(...){close_noexcept();throw;}
    }
    ~Context(){close_noexcept();}
    Context(const Context&)=delete;Context&operator=(const Context&)=delete;
    void admit(cudaStream_t stream)const{
        issuing_owner();outside_capture();
        need(!poisoned_&&stream==stream_&&partial_&&spans_&&previous_,
             "R027 V16 producer owner is unbound or poisoned");
    }
    void bind_after_drain(void* scratch,size_t bytes,cudaStream_t stream){
        issuing_owner();outside_capture();need(stream==stream_&&!poisoned_,"R027 V16 stream rebind/state mismatch");
        check(cudaStreamSynchronize(stream_),"borrowed scratch rebind drain");pending_=false;
        need(epoch_!=UINT64_MAX,"R027 V16 borrowed epoch exhausted");++epoch_;
        const size_t spans_at=align16(partial_bytes_),previous_at=spans_at+align16(max_tiles_*sizeof(swift_tail_r027::TileSpan));
        need(scratch&&!(reinterpret_cast<uintptr_t>(scratch)&15u)&&
             previous_at+max_previous_*sizeof(int32_t)<=bytes,"R027 V16 GEMM scratch lease too small");
        auto*p=static_cast<unsigned char*>(scratch);
        partial_=reinterpret_cast<float*>(p);spans_=reinterpret_cast<swift_tail_r027::TileSpan*>(p+spans_at);
        previous_=reinterpret_cast<int32_t*>(p+previous_at);
    }
    void run(const strata::prefill::mmq::Product&p,const int32_t*host_bounds,
             const int32_t*identity,int64_t row0,int64_t nr,const float*tails,
             void*hq,size_t hq_capacity,float*row_sd,size_t row_sd_capacity,cudaStream_t stream){
        try{
            admit(stream);
            need(!poisoned_&&stream==stream_&&partial_&&spans_&&previous_&&host_bounds&&
                 p.type==40&&p.n>=1&&p.n<=16&&p.w_rows==1280&&p.w_cols==2560&&
                 p.expert_bytes==1843200&&p.ld_dst==2560&&p.ids==identity&&identity&&
                 p.total_rows>=1&&p.total_rows<=int64_t(max_rows_)*10&&
                 p.total_rows%10==0&&p.max_rows>=1&&p.max_rows<=max_rows_&&
                 row0>=0&&nr>0&&nr<=p.total_rows&&row0<=p.total_rows-nr&&tails&&hq,
                 "R027 V16 strict native40 owning product/row span");
            need(host_bounds[0]==row0&&host_bounds[p.n]==row0+nr,"R027 V16 absolute bounds endpoints");
            int64_t maxr=0;
            for(int i=0;i<p.n;++i){
                need(host_bounds[i]>=row0&&host_bounds[i+1]>=host_bounds[i]&&host_bounds[i+1]<=row0+nr,
                     "R027 V16 nonmonotone/out-of-range routed bounds");
                maxr=std::max<int64_t>(maxr,int64_t(host_bounds[i+1])-host_bounds[i]);
            }
            need(maxr==p.max_rows,"R027 V16 original max-row mismatch");
            const auto shape=swift_tail_r027::describe_shape(p,profile_);
            need(shape.partial_bytes<=partial_bytes_,"R027 V16 partial capacity");
            const auto&hp=plan(shape.geometry);
            swift_tail_r027::Binding b;b.owner=owner_;b.stream=stream_;b.epoch=epoch_;b.identity_ids=identity;
            b.partial=partial_;b.partial_capacity=partial_bytes_;b.device_spans=spans_;b.device_previous_blocks=previous_;
            b.span_count=hp.spans.size();b.previous_count=hp.previous_blocks.size();b.tails=tails;b.row0=row0;b.nr=nr;
            b.hq=hq;b.hq_capacity=hq_capacity;b.row_sd=row_sd;b.row_sd_capacity=row_sd_capacity;
            upload(hp);pending_=true;
            swift_tail_r027::enqueue_raw(p,shape,b);
            swift_tail_r027::enqueue_completion(p,shape,b);
        }catch(...){poisoned_=true;throw;} // no ordinary GU/R56 retry, including partial submission
    }
};
} // namespace swift_tail_v16
