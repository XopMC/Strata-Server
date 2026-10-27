// Included ONCE in Engine TU inside strata::ds41. No publicEngine/class changes.
namespace { namespace woa449 {
struct State {
    std::mutex mutex;
    std::array<uint64_t,12> values{};
    bool closed=false;
    const bool quant_mode;
    explicit State(bool mode,bool quant):quant_mode(quant) { values[0]=449;values[1]=uint64_t(mode);values[2]=1; }
};
static std::mutex registry_mutex;
static std::array<std::weak_ptr<State>,2> registry;
inline std::shared_ptr<State> create(bool mode,bool quant) {
    auto p=std::make_shared<State>(mode,quant);
    std::lock_guard<std::mutex> lock(registry_mutex);
    for(auto& slot:registry)if(slot.expired()){slot=p;return p;}
    throw std::logic_error("449 bounded workspace registry already has two live entries");
}
inline void close(const std::shared_ptr<State>&p) noexcept {
    if(!p)return;
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    std::lock_guard<std::mutex> lock(p->mutex);p->closed=true;p->values[2]=0;
    for(auto&slot:registry)if(auto q=slot.lock())if(q.get()==p.get())slot.reset();
}
// OUTSIDE capture. Destruction on anyexception records a failed hostverify,
// never fabricates a successful GPUlaunch or commits a failed verification.
struct Window {
    std::shared_ptr<State> p;
    bool completed=false;
    explicit Window(std::shared_ptr<State> state):p(std::move(state)) {
        std::lock_guard<std::mutex> lock(p->mutex);
        if(p->closed||p->values[10])throw std::logic_error("449 verifier notidle");
        p->values[10]=1;
    }
    Window(const Window&)=delete;Window&operator=(const Window&)=delete;
    ~Window() {
        if(!completed){std::lock_guard<std::mutex>lock(p->mutex);++p->values[8];p->values[10]=0;}
    }
    void success(int m,bool reused) {
        std::lock_guard<std::mutex>lock(p->mutex);
        ++p->values[3];p->values[4]+=uint64_t(m);
        ++p->values[(p->values[1]&&m>=2&&m<=6)?5:6];
        p->values[7]+=uint64_t(reused);p->values[9]=uint64_t(m);
        p->values[10]=0;completed=true;
    }
};
}} //namespace woa449, anonymous (all helpers TU-local)
extern "C" int ds449_verify_woa_info(uint64_t*out,int count) {
    if(!out||count!=12)return 1;
    try {
        std::shared_ptr<woa449::State> p;
        {std::lock_guard<std::mutex>lock(woa449::registry_mutex);
         for(auto&slot:woa449::registry)if(auto q=slot.lock()){
             if(p)return 3;p=std::move(q);
         }}
        if(!p)return 2;
        std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);
        if(!lock.owns_lock())return 4;
        if(p->closed)return 2;
        if(p->values[10])return 4;
        std::copy(p->values.begin(),p->values.end(),out);return 0;
    }catch(...){return 5;}
}

// Same idle host-window snapshot; quant_mode is latched, NOT a hardware launch count.
extern "C" int ds456_verify_woa_quant_info(uint64_t*out,int count) {
    if(!out||count!=12)return 1;
    try {
        std::shared_ptr<woa449::State> p;
        {std::lock_guard<std::mutex>lock(woa449::registry_mutex);
         for(auto&slot:woa449::registry)if(auto q=slot.lock()){
             if(p)return 3;p=std::move(q);
         }}
        if(!p)return 2;
        std::unique_lock<std::mutex>lock(p->mutex,std::try_to_lock);
        if(!lock.owns_lock())return 4;
        if(p->closed)return 2;
        if(p->values[10])return 4;
        std::copy(p->values.begin(),p->values.end(),out);
        out[0]=456;out[1]=uint64_t(p->quant_mode);return 0;
    }catch(...){return 5;}
}
