// Inside strata::ds41 after original317/325 stats. DefaultOFF no execution.
namespace geometry335 {
static std::array<std::atomic<uint64_t>,32>stats{};
static void set(int i,uint64_t n){stats[i].store(n,std::memory_order_relaxed);}
static void add(int i,uint64_t n=1){stats[i].fetch_add(n,std::memory_order_relaxed);}
static void minimum(int i,uint64_t n){auto v=stats[i].load(std::memory_order_relaxed);while((!v||n<v)&&!stats[i].compare_exchange_weak(v,n,std::memory_order_relaxed)){};}
struct Request {
 bool on=false,done=false;
 Request(const Fixed&g,bool serial_mode,bool wave):on(g.is256()){
  if(!on)return;
  if(serial_mode==wave)throw std::invalid_argument("335 choose exactly serial reference OR wavefront");
  if(stats[9].load()||stats[12].exchange(1))throw std::logic_error("335 fresh N1 child request required");
  set(1,1);set(2,serial_mode?1:2);set(3,g.context);set(4,g.prompt_end);set(5,g.split);set(6,g.cap);set(7,g.B);set(8,g.E);add(9);
 }
 void complete(){if(on){add(10);done=true;}}
 ~Request(){if(on){if(!done)add(11);set(12,0);}}
};
}
extern "C" int ds335_prefill_geometry_info(uint64_t*out,int n)noexcept{
 if(!out||n!=32)return 1;out[0]=335;
 for(int i=1;i<32;++i)out[i]=geometry335::stats[i].load(std::memory_order_relaxed);
 // Read actual original source counters. Not inferred from requested geometry.
 out[13]=prefill317_detail::stats[3].load();out[14]=prefill317_detail::stats[2].load();out[15]=prefill317_detail::stats[1].load();out[16]=prefill317_detail::stats[4].load();out[17]=prefill317_detail::stats[5].load();out[18]=wavefront325::stats[6].load();out[19]=wavefront325::stats[7].load();return 0;
}
