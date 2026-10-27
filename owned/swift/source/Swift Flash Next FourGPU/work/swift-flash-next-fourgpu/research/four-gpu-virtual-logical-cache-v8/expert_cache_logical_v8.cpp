#include "strata/core/expert_cache.hpp"
#include <algorithm>
#include <limits>
#include <exception>
#include <utility>
namespace strata::core {
bool ExpertCache::open_logical_sized(const std::vector<int64_t>&slot_bytes,int64_t nl,int64_t ne,std::string&err){
    // A caller cannot turn a live physical cache into metadata or silently free
    // it. Use a fresh object before frontend payload allocation/capture.
    if(base_!=nullptr||!segs_.empty()||vmm_){err="ExpertCache V8: logical open requires fresh/no-payload cache";return false;}
    if(nl!=48||ne!=512||slot_bytes.empty()||slot_bytes.size()>size_t(nl*ne)||seg_req_!=0){err="ExpertCache V8: unsupported logical geometry/elastic cache";return false;}
    try{
        std::vector<uint64_t>off(slot_bytes.size()+1,0);
        for(size_t i=0;i<slot_bytes.size();++i){
            if(slot_bytes[i]!=2764816||guard_req_>std::numeric_limits<int64_t>::max()-slot_bytes[i]-255){err="ExpertCache V8: native40 slot/guard mismatch";return false;}
            uint64_t stride=uint64_t(slot_stride(slot_bytes[i],guard_req_));
            if(stride>uint64_t(std::numeric_limits<int64_t>::max())-off[i]){err="ExpertCache V8: virtual offset overflow";return false;}
            off[i+1]=off[i]+stride;
        }
        std::vector<int32_t>res(size_t(nl*ne),kNotResident),next(size_t(nl),0);
        int64_t q=int64_t(slot_bytes.size())/nl;
        for(int64_t l=0;l<nl;++l)next[size_t(l)]=int32_t(l*q);
        close(); // previously empty/logical, so no CUDA/payload action
        logical_only_=true;slots_=live_slots_=int64_t(slot_bytes.size());
        n_layers_=nl;n_expert_=ne;blob_=2764816;guard_=guard_req_;
        off_=std::move(off);residency_=std::move(res);layer_next_=std::move(next);
        next_free_=fills_=admitted_=0;
        return true;
    }catch(const std::exception&e){err=std::string("ExpertCache V8: metadata allocation: ")+e.what();return false;}
}
bool ExpertCache::install_logical_selection(const std::vector<int32_t>&selected,int64_t lb,int64_t le,std::string&err){
    if(!logical_only_||resident()!=0||selected.size()!=size_t(n_layers_*n_expert_)||lb<0||lb>=le||le>n_layers_||per_layer_){err="ExpertCache V8: selection must bind empty global-profile logical cache";return false;}
    try{
        std::vector<uint8_t>seen(size_t(slots_),0);
        for(int64_t l=0;l<n_layers_;++l)for(int64_t e=0;e<n_expert_;++e){
            int32_t s=selected[size_t(l*n_expert_+e)];
            if(s<0)continue;
            if(l<lb||l>=le||s>=slots_||++seen[size_t(s)]!=1){err="ExpertCache V8: selected map home/virtual-slot/duplicate mismatch";return false;}
        }
        if(std::find(seen.begin(),seen.end(),0)!=seen.end()){err="ExpertCache V8: selected virtual slot missing";return false;}
        residency_=selected;next_free_=slots_;admitted_=slots_;return true;
    }catch(const std::exception&e){err=std::string("ExpertCache V8: selection metadata allocation: ")+e.what();return false;}
}
}
