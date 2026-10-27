// Included once INSIDE strata::ds41 namespace in this private Engine TU.
namespace prefill317_detail {
static std::array<std::atomic<uint64_t>,12>stats{};
static bool enabled(){const char*f=std::getenv("DS1017_PREFILL_PASS_SEAM_V317");if(f&&std::strcmp(f,"0")&&std::strcmp(f,"1"))throw std::invalid_argument("literal317 pass seam flag required");return f&&!std::strcmp(f,"1");}
static void bump(int slot){stats[slot].fetch_add(1,std::memory_order_relaxed);}
static void drop(int slot){stats[slot].fetch_sub(1,std::memory_order_relaxed);}
static void set(int slot,uint64_t value){stats[slot].store(value,std::memory_order_relaxed);}
}
extern "C" int ds317_prefill_seam_info(uint64_t*out,int n)noexcept{
 if(!out||n!=12)return 1;out[0]=317;for(int i=1;i<12;i++)out[i]=prefill317_detail::stats[i].load(std::memory_order_relaxed);return 0;
}
