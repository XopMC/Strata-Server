#pragma once
#include "adaptive_geometry_v528.hpp"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace strata::ds41::serving516 {
inline bool enabled() {
    const char* v = std::getenv("DS1017_PAIR_SERVING_V516");
    if (v && std::strcmp(v, "0") && std::strcmp(v, "1"))
        throw std::invalid_argument("516 serving flag must be literal0/1");
    return v && !std::strcmp(v, "1");
}
inline adaptive528::Plan prepared528;
inline bool ordinary(bool on, int context, int rows, int pos, bool fresh,
                     bool images, bool nll, bool wavefront) {
    if (!on) {if(prepared528.armed)throw std::invalid_argument("528 prepared capability requires serving flag1");return false;}
    if(prepared528.armed){
        const auto&p=prepared528;
        if(p.g.context!=context||p.N!=rows||p.pos!=pos||bool(p.flags&adaptive528::Images)!=images||bool(p.flags&adaptive528::Nll)!=nll||((p.mode==2||p.mode==3)&&!fresh))
            throw std::invalid_argument("528 actual request differs from idle capability");
        if(p.mode==2&&!wavefront)throw std::invalid_argument("528 prepared wavefront capability requires enabled scheduler");
        if(p.mode!=2&&p.mode!=3&&p.mode!=4)throw std::invalid_argument("534 mandatory strategy has no ordinary mode1");
        return p.mode==4;
    }
    throw std::invalid_argument("534 mandatory strategy requires prepared528 capability");
}
inline std::mutex registry;
inline adaptive528::Plan snapshot528(int context){
    std::lock_guard<std::mutex> lock(registry);
    if(prepared528.armed&&prepared528.g.context!=context)throw std::invalid_argument("528 Engine context differs from prepared capability");
    return prepared528;
}
inline std::atomic<uint64_t> engines{0}, active{0}, epoch{0};
inline std::atomic<uint64_t> mode{0}, rows{0}, pos{0}, context{0}, cap{0}, B{0}, E{0};
// First Impl member: a failed constructor also drops its registration. The
// headless helper is registered too. Reset cannot race model construction.
struct Lifetime {
    Lifetime() { std::lock_guard<std::mutex> lock(registry); ++engines; }
    ~Lifetime() { std::lock_guard<std::mutex> lock(registry); --engines; }
    Lifetime(const Lifetime&) = delete;
    Lifetime& operator=(const Lifetime&) = delete;
};
struct Scope {
    bool on, fallback, old_wave, old_seam;
    bool& wave;
    bool& seam;
    Scope(bool& w, bool& s, int c, int n, int p, bool fresh, bool images,
          bool nll, int chunk, int batch, int expert_batch)
        : on(enabled()), fallback(ordinary(on,c,n,p,fresh,images,nll,w)),
          old_wave(w), old_seam(s), wave(w), seam(s) {
        if (!on) return;
        if(prepared528.armed&&(prepared528.mode==2||prepared528.mode==3)&&!s)
            throw std::invalid_argument("528 original pass seam must be enabled");
        if(prepared528.armed&&
            (chunk!=prepared528.g.cap||batch!=prepared528.g.B||expert_batch!=prepared528.g.E))
            throw std::invalid_argument("528 configured cap/B/E differ from immutable plan");
        uint64_t expected = 0;
        if (!active.compare_exchange_strong(expected,1))
            throw std::logic_error("516 N1 prefill already active");
        mode = fallback ? 4 : 2; rows=n; pos=p; context=c;
        cap=chunk; B=batch; E=expert_batch;
        if(prepared528.armed&&(prepared528.mode==2||prepared528.mode==3)){
            if(prepared528.mode==3){wave=false;seam=true;}
        }
        if (fallback) { wave=false; seam=false; }
    }
    ~Scope() {
        if (on) { wave=old_wave; seam=old_seam; active=0; }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
struct Disable {
    bool& value;
    const bool saved;
    Disable(bool& v,bool off):value(v),saved(v) { if(off)value=false; }
    ~Disable() { value=saved; }
    Disable(const Disable&)=delete;
    Disable& operator=(const Disable&)=delete;
};
}
