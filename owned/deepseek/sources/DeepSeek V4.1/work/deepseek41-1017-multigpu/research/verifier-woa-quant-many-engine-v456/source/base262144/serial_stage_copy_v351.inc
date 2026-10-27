// Private defaultOFF correction for the actual335 serial256 caller only.
// The325 rank pool below is unchanged: actual currentCUDA device, PCI41 NUMA
// sysfs validation, explicit singleton CPUs, worker pinning and joined copies.
namespace {
static std::array<std::atomic<uint64_t>,12>serial_copy_counts351{};
static bool serial_rank_config351(const char*flag,bool fixed256,bool serial,const char*wave,const char*headless,const char*cpus){
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1")||!fixed256||!serial||!wave||std::strcmp(wave,"0")||!headless||std::strcmp(headless,"0")||!cpus||std::strcmp(cpus,"0,1,2,3"))
        throw std::invalid_argument("351 requires explicit serial335256, wave/headlessOFF and exact GPU0 copy CPUs0,1,2,3");
    return true;
}
static bool serial_rank_enabled351(){
    static const bool enabled=serial_rank_config351(std::getenv("DS1017_SERIAL_RANK_STAGE_COPY_V351"),geometry335::selected().is256(),geometry335::serial(),std::getenv("DS1017_PREFILL_WAVEFRONT_V325"),std::getenv("DS1017_HEADLESS_PREFILL_V320"),std::getenv("DS1017_WAVEFRONT_COPY0_CPUS_V325"));
    return enabled;
}
static NativeStageCopyPoolV2&stage_copy_pool_v2(){
    if(serial_rank_enabled351()){
        try{
            int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=0)throw std::runtime_error("351 serial copy requires actual currentGPU0");
            auto&pool=stage_copy_rank_pool325();
            static thread_local bool counted=false;
            if(!counted){serial_copy_counts351[2].fetch_add(1);counted=true;}
            stage_copy_pool_instance_v2=&pool;
            serial_copy_counts351[6].store(1); // actual device0 witnessed, not requested ID
            serial_copy_counts351[7].store(1); // unchanged325 rank-pool NUMA/pin validation completed
            serial_copy_counts351[8].store(4);
            serial_copy_counts351[9].store(15); // exact explicit CPU0..3 mask validated by config+325
            return pool;
        }catch(...){serial_copy_counts351[11].fetch_add(1);throw;}
    }
    static NativeStageCopyPoolV2 pool(4);stage_copy_pool_instance_v2=&pool;return pool;
}
static void serial_copy_observed351(uint64_t parallel,size_t bytes){
    if(!serial_rank_enabled351())return;
    serial_copy_counts351[3].fetch_add(1);serial_copy_counts351[4].fetch_add(parallel);serial_copy_counts351[5].fetch_add(bytes);
}
}
extern "C" int ds351_serial_stage_copy_info(uint64_t*out,int n){
    if(!out||n!=12)return -1;
    try{out[0]=351;out[1]=uint64_t(serial_rank_enabled351());for(int i=2;i<12;++i)out[i]=serial_copy_counts351[i].load();return 0;}
    catch(...){return -2;}
}
