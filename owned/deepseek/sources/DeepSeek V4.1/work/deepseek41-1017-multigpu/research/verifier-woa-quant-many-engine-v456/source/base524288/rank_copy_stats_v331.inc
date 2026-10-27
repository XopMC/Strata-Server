// Included once within strata::ds41. New DSO-local source-only diagnostic C ABI.
namespace copy331 {
static std::array<std::atomic<uint64_t>,16>stats{};
static thread_local bool observing=false;
static thread_local int actual_rank=-1;
static bool enabled(){
 const char*v=std::getenv("DS1017_PARALLEL_HEADLESS_STAGE_COPY_V331");
 if(v&&std::strcmp(v,"0")&&std::strcmp(v,"1"))throw std::invalid_argument("331 literal opt-in0/1 required");
 const bool on=v&&!std::strcmp(v,"1");
 if(on){
  const char*parallel=std::getenv("DS41_NATIVE_PREFILL_PARALLEL_STAGE_COPY");
  if(!wavefront325::enabled()||!parallel||std::strcmp(parallel,"1"))throw std::invalid_argument("331 requires explicit325 and original parallel-stage mode");
  stats[1].fetch_add(1,std::memory_order_relaxed);
 }
 return on;
}
}
// Read ONLY after synchronous prefill/generation/close on the freshN1 child.
extern "C" int ds331_rank_stage_copy_info(uint64_t*out,int n)noexcept{
 if(!out||n!=16)return 1;out[0]=331;
 for(int i=1;i<16;i++)out[i]=copy331::stats[i].load(std::memory_order_acquire);
 return 0;
}
