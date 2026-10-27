#include "../../../../../../../include/strata_ds/fresh_diagnostic_visibility.hpp"
// Appended to the private Pack host TU, outside strata::ds41 namespace. This
// keeps the exact five-step recipe; original151 CABI/Handle definitions untouched.
#include "headless_resource_v320.h"
#include "strata/ds41/engine_causal_batch_v310.hpp"
#include "pair_registry_v145.hpp"
extern "C" int ds320_headless_resource_component(void*checkpoint,const ds41_native_hash*hash,
 uint64_t budget,uint64_t reserve,uint64_t*out,int count,char*message,size_t message_bytes){
    using namespace strata::ds41;
    if(!out||count!=32||!message||message_bytes<2)return -1;
    std::fill(out,out+32,uint64_t(0));out[0]=320;out[1]=1;out[2]=524288;out[6]=budget;out[7]=reserve;message[0]=0;
    int caller=-1,rc=1;bool switched=false;
    auto fail=[](bool ok,const char*why){if(!ok)throw std::invalid_argument(why);};
    auto cuda_check=[](cudaError_t e,const char*why){if(e!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e));};
    try{
        const char*flag=std::getenv("DS1017_HEADLESS_PREFILL_RESOURCE_V320");
        fail(flag&&!std::strcmp(flag,"1"),"320 resource component explicit opt-in required");
        fail(checkpoint&&hash&&hash->token_map&&hash->token_count==kVocab&&hash->multipliers&&hash->multiplier_count==8&&hash->primes&&hash->offsets&&hash->modulus_count==48,"320 actual checkpoint/hash arguments");
        const char*pair=std::getenv("DS41_NATIVE_PAIR_MASK"),*stage=std::getenv("DEEPSEEK41_GPU_STAGE");
        fail(pair&&!std::strcmp(pair,"1")&&stage&&!std::strcmp(stage,"0")&&strata::ds41::publication::metadata_snapshot(2),"320 fresh pair snapshot BEFORE CUDA");
        out[1]=2;int devices=0;cuda_check(cudaGetDeviceCount(&devices),"320 visible count");fail(devices==2,"320 only admitted pair visible");
        fail(strata::ds41::publication::runtime_snapshot(2),"320 fresh pair CUDA UUID/order mismatch");
        cuda_check(cudaGetDevice(&caller),"320 caller device");fail(caller==0,"320 resource fixture called from owning GPU0 child");
        const auto checkpoint_owner=std::make_shared<NativeCheckpoint>(*static_cast<NativeCheckpoint*>(checkpoint));
        const auto binding=pair145::info(*checkpoint_owner->weights());
        fail(binding[2]==46080&&binding[3]==305764761600ull&&binding[4]&&binding[5]&&binding[6]==1,"320 real SAME-DSO typed139 checkpoint required");
        out[28]=binding[4];out[29]=binding[5];
        EngramHash h;h.max_ngram=4;h.n_heads=8;h.pad=hash->pad;h.layers={1,14};h.token_map.assign(hash->token_map,hash->token_map+hash->token_count);
        for(int i=0;i<2;++i){h.multipliers.emplace_back(hash->multipliers+i*4,hash->multipliers+(i+1)*4);h.primes.emplace_back(hash->primes+i*24,hash->primes+(i+1)*24);h.offsets.emplace_back(hash->offsets+i*24,hash->offsets+(i+1)*24);}
        EngineOptions options;options.max_seq=524288;options.cpu_threads=0;options.prefill_chunk=0;options.vram_expert_slots=0;options.ram_budget_gib=0;
        NativeEngineOptions native;native.graph=false;HeadlessOptionsV320 limits;limits.max_device_bytes=budget;limits.reserve_bytes=reserve;
        cuda_check(cudaSetDevice(1),"320 explicit fixture selectGPU1");switched=true;size_t free=0,total=0;
        cuda_check(cudaMemGetInfo(&free,&total),"320 before constructor free");out[3]=free;out[1]=3;
        {
            auto context=std::make_unique<Engine>(checkpoint_owner,std::move(h),options,native,EngineRoleV320::HeadlessPrefillRank1,limits);
            out[1]=4;const auto info=context->headless_info_v320();std::copy(info.begin(),info.end(),out+8);
            const auto dense=context->pack_headless_info_v320();out[24]=dense[6];out[25]=dense[3];out[26]=dense[4];out[27]=dense[5];
            fail(info[0]==320&&info[1]==320&&info[2]==1&&info[3]==1&&!info[4]&&!info[5]&&!info[6]&&!info[7]&&info[8]==1323827200ull,"320 actual headless owner/worker/doorbell/head omission guard");
            cuda_check(cudaMemGetInfo(&free,&total),"320 constructed free");out[4]=free;
            context->headless_close_v320();out[1]=5;context.reset();out[30]=1;
        }
        cuda_check(cudaMemGetInfo(&free,&total),"320 destroyed free");out[5]=free;out[1]=6;rc=0;
    }catch(const std::exception&e){std::snprintf(message,message_bytes,"%s",e.what());}
     catch(...){std::snprintf(message,message_bytes,"unknown320 resource exception");rc=2;}
    if(switched){const auto status=cudaSetDevice(caller);if(status!=cudaSuccess){std::snprintf(message,message_bytes,"320 restoreGPU0: %s",cudaGetErrorString(status));rc=3;}else out[31]=1;}
    return rc;
}
