#include "strata/ds41/native_host_scatter.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <limits>
#include <sched.h>
#include <set>
#include <stdexcept>
#include <thread>
namespace strata::ds41 {
struct NativeHostScatter::Impl {
    float*out;int rows,dim;std::vector<int>cpus;std::vector<std::thread>threads;std::vector<uint64_t>counts;
    mutable std::mutex command;std::mutex state;std::condition_variable ready,complete;
    bool stopping=false,clear=false,failed=false,startup_failed=false,layer_started=false;uint64_t generation=0;
    int initialized=0,pending=0,last_expert=-1,last_row=-1;std::atomic<bool>operation_failed{false};
    const int32_t*ids=nullptr;const uint16_t*values=nullptr;size_t count=0;
    Impl(float*o,size_t elements,int r,int d,std::vector<int>c):out(o),rows(r),dim(d),cpus(std::move(c)),counts(cpus.size()){
        if(!out||rows<1||rows>1048576||dim<32||dim>8192||dim%32||elements!=size_t(rows)*dim||cpus.empty()||cpus.size()>256)
            throw std::invalid_argument("invalid native host accumulator shape");
        cpu_set_t allowed;CPU_ZERO(&allowed);if(sched_getaffinity(0,sizeof(allowed),&allowed))throw std::runtime_error("cannot read CPU affinity");
        std::set<int>seen;for(int cpu:cpus)if(cpu<0||cpu>=CPU_SETSIZE||!CPU_ISSET(cpu,&allowed)||!seen.insert(cpu).second)
            throw std::invalid_argument("invalid native host scatter CPUs");
    }
    ~Impl(){
        {std::lock_guard<std::mutex>lock(state);stopping=true;}ready.notify_all();
        for(auto&t:threads)if(t.joinable())t.join();
    }
    void start(){
        for(size_t i=0;i<cpus.size();i++)threads.emplace_back([this,i]{worker(i);});
        std::unique_lock<std::mutex>lock(state);complete.wait(lock,[&]{return initialized==int(cpus.size());});
        if(startup_failed)throw std::runtime_error("host scatter worker affinity failed");
    }
    void worker(size_t index){
        cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpus[index],&set);const bool bad=sched_setaffinity(0,sizeof(set),&set)!=0;
        std::unique_lock<std::mutex>lock(state);startup_failed|=bad;initialized++;complete.notify_all();uint64_t seen=0;
        while(true){
            ready.wait(lock,[&]{return stopping||generation!=seen;});if(stopping)return;
            const auto current=generation;const bool zero=clear;const auto n=count;const auto*row_ids=ids;const auto*input=values;
            lock.unlock();
            if(zero){
                const size_t first=size_t(rows)*index/cpus.size(),end=size_t(rows)*(index+1)/cpus.size();
                std::fill(out+first*dim,out+end*dim,0.f);
            }else{
                const size_t first=n*index/cpus.size(),end=n*(index+1)/cpus.size();
                for(size_t row=first;row<end;row++){
                    float*target=out+size_t(row_ids[row])*dim;
                    for(int d=0;d<dim;d++){
                        const uint32_t bits=uint32_t(input[row*dim+d])<<16;float value;std::memcpy(&value,&bits,4);
                        const float sum=target[d]+value;
                        if(!std::isfinite(sum))operation_failed.store(true,std::memory_order_relaxed);
                        target[d]=sum;
                    }
                }
                counts[index]+=end-first;
            }
            lock.lock();seen=current;if(--pending==0)complete.notify_all();
        }
    }
    void dispatch(bool zero,const int32_t*row_ids,const uint16_t*input,size_t n){
        std::unique_lock<std::mutex>lock(state);clear=zero;ids=row_ids;values=input;count=n;pending=int(cpus.size());
        operation_failed=false;generation++;ready.notify_all();complete.wait(lock,[&]{return pending==0;});
        if(operation_failed){failed=true;throw std::runtime_error("nonfinite native host routed accumulator");}
    }
};
NativeHostScatter::NativeHostScatter(float*out,size_t elements,int rows,int dim,std::vector<int>cpus)
    :impl_(std::make_unique<Impl>(out,elements,rows,dim,std::move(cpus))){impl_->start();}
NativeHostScatter::~NativeHostScatter()=default;
void NativeHostScatter::begin_layer(){auto&p=*impl_;std::lock_guard<std::mutex>lock(p.command);p.dispatch(true,nullptr,nullptr,0);p.failed=false;p.layer_started=true;p.last_expert=p.last_row=-1;}
void NativeHostScatter::add(int expert,const std::vector<int32_t>&ids,const std::vector<uint16_t>&values){
    add_view(expert,ids.data(),ids.size(),values.data(),values.size());
}
void NativeHostScatter::add_view(int expert,const int32_t*ids,size_t count,const uint16_t*values,size_t elements){
    auto&p=*impl_;std::lock_guard<std::mutex>lock(p.command);
    if(!p.layer_started)throw std::logic_error("begin_layer is required before native host scatter");
    if(p.failed)throw std::runtime_error("failed host scatter must be reset");
    if(expert<0||expert>=384||expert<p.last_expert||!ids||!values||!count||count>16384||elements!=count*size_t(p.dim))
        throw std::invalid_argument("invalid native host scatter packet or expert order");
    const uintptr_t destination=reinterpret_cast<uintptr_t>(p.out);
    const size_t destination_bytes=size_t(p.rows)*p.dim*sizeof(float);
    auto distinct=[&](const void*data,size_t bytes){
        const uintptr_t source=reinterpret_cast<uintptr_t>(data),maximum=std::numeric_limits<uintptr_t>::max();
        if(destination_bytes>maximum-destination||bytes>maximum-source||
           (source<destination+destination_bytes&&destination<source+bytes))
            throw std::invalid_argument("host scatter packet aliases its destination or overflows address range");
    };
    distinct(ids,count*sizeof(int32_t));distinct(values,elements*sizeof(uint16_t));
    int previous=expert==p.last_expert?p.last_row:-1;
    for(size_t i=0;i<count;i++){int row=ids[i];if(row<=previous||row<0||row>=p.rows)throw std::invalid_argument("native scatter rows must be unique ascending positions");previous=row;}
    p.dispatch(false,ids,values,count);p.last_expert=expert;p.last_row=previous;
}
std::vector<uint64_t>NativeHostScatter::worker_rows()const{auto&p=*impl_;std::lock_guard<std::mutex>lock(p.command);return p.counts;}
}
