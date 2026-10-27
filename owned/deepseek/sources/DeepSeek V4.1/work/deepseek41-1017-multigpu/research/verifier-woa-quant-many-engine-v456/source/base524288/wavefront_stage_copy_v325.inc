// Isolated host byte-copy callers. The original global pool is NOT multi-caller
// safe; its literal defaultOFF path remains above. These live only on the two
// actual persistent325 issuers and die on those threads after every copy joined.
static thread_local bool stage_copy_rank_mode325=false;
static std::array<std::atomic<uint64_t>,3>stage_copy_counts325{};
static std::vector<int>parse_cpus325(const std::string&text,bool ranges){
    std::vector<int>out;size_t at=0;
    while(at<text.size()){
        const size_t end=text.find(',',at),stop=end==std::string::npos?text.size():end;
        if(stop==at)throw std::invalid_argument("325 empty CPU field");
        const auto part=text.substr(at,stop-at);const auto dash=part.find('-');
        if(dash!=std::string::npos&&!ranges)throw std::invalid_argument("325 explicit copy CPUs require four singleton IDs");
        auto integer=[](const std::string&s){size_t used=0;int n=std::stoi(s,&used);if(used!=s.size()||n<0||n>=CPU_SETSIZE)throw std::invalid_argument("325 CPU token");return n;};
        const int first=integer(part.substr(0,dash)),last=dash==std::string::npos?first:integer(part.substr(dash+1));
        if(last<first)throw std::invalid_argument("325 CPU interval");for(int c=first;c<=last;++c)out.push_back(c);
        at=stop+1;
    }
    return out;
}
static NativeStageCopyPoolV2&stage_copy_rank_pool325(){
    static thread_local std::unique_ptr<NativeStageCopyPoolV2>pool;
    static thread_local int bound=-1;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device<0||device>1)throw std::runtime_error("325 copy pool actual rank");
    if(pool){if(bound!=device)throw std::logic_error("325 copy pool device changed");return *pool;}
    const char*key=device?"DS1017_WAVEFRONT_COPY1_CPUS_V325":"DS1017_WAVEFRONT_COPY0_CPUS_V325";
    const char*raw=std::getenv(key);if(!raw)throw std::invalid_argument(std::string("325 explicit ")+key+" required");
    auto cpus=parse_cpus325(raw,false);if(cpus.size()!=4)throw std::invalid_argument("325 four copy workers per rank required");
    const int node=stage_copy_current_numa_node_v2(device,"325 actual GPU NUMA node unavailable");
    std::string list;{std::ifstream f("/sys/devices/system/node/node"+std::to_string(node)+"/cpulist");if(!(f>>list))throw std::runtime_error("325 NUMA CPU list unavailable");}
    const auto local=parse_cpus325(list,true);
    for(int c:cpus)if(c==sched_getcpu()||std::find(local.begin(),local.end(),c)==local.end())throw std::invalid_argument("325 copy worker not local or aliases issuer");
    pool=std::make_unique<NativeStageCopyPoolV2>(4,std::move(cpus));bound=device;if(copy331::observing){copy331::actual_rank=device;copy331::stats[2+device].fetch_add(1,std::memory_order_relaxed);}return *pool;
}
