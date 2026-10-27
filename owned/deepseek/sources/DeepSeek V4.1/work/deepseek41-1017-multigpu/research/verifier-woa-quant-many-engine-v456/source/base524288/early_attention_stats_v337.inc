// Included once inside strata::ds41; private separate early/full event evidence.
namespace early337 {
static std::array<std::atomic<uint64_t>,24>stats{};
static void add(int i,uint64_t n=1){stats[i].fetch_add(n,std::memory_order_relaxed);}
static void drop(int i){stats[i].fetch_sub(1,std::memory_order_relaxed);}
static void set(int i,uint64_t n){stats[i].store(n,std::memory_order_relaxed);}
static bool enabled(){const char*f=std::getenv("DS1017_PREFILL_EARLY_ATTENTION_V337");if(f&&std::strcmp(f,"0")&&std::strcmp(f,"1"))throw std::invalid_argument("337 literal optin0/1");const bool on=f&&!std::strcmp(f,"1");if(on){if(!wavefront325::enabled())throw std::invalid_argument("337 requires original325wavefront optin");add(1);}return on;}
}
extern "C" int ds337_early_attention_info(uint64_t*out,int n)noexcept{if(!out||n!=24)return 1;out[0]=337;for(int i=1;i<24;++i)out[i]=early337::stats[i].load(std::memory_order_relaxed);return 0;}
