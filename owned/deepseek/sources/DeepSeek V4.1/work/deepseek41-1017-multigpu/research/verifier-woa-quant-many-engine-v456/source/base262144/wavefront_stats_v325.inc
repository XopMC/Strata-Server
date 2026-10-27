// Included once inside strata::ds41 namespace, not a public Engine layout.
namespace wavefront325 {
static std::array<std::atomic<uint64_t>,48>stats{};
static std::atomic<bool>active{false};
static bool enabled(){const char*v=std::getenv("DS1017_PREFILL_WAVEFRONT_V325");if(v&&std::strcmp(v,"0")&&std::strcmp(v,"1"))throw std::invalid_argument("325 literal opt-in flag");return v&&!std::strcmp(v,"1");}
static void add(int i,uint64_t n=1){stats[i].fetch_add(n,std::memory_order_relaxed);}
static void set(int i,uint64_t n){stats[i].store(n,std::memory_order_relaxed);}
static void minimum(int i,uint64_t n){auto old=stats[i].load(std::memory_order_relaxed);while((!old||n<old)&&!stats[i].compare_exchange_weak(old,n,std::memory_order_relaxed)){};}
}
extern "C" int ds325_prefill_wavefront_info(uint64_t*out,int n)noexcept{if(!out||n!=48)return 1;out[0]=325;for(int i=1;i<48;++i)out[i]=wavefront325::stats[i].load(std::memory_order_relaxed);return 0;}
