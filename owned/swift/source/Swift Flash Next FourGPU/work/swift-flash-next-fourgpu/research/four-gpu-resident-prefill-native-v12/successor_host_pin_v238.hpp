#pragma once
// Private Linux host-only experiment. No public class, CUDA, or model ABI.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
namespace strata::prefill::successor_host_pin_v238 {
// This is thread-local evidence of the actual RAII pin, never a process-wide admission hint.
inline thread_local bool owning_scope_active_v238 = false;
inline bool scope_active() noexcept { return owning_scope_active_v238; }
enum class Admission { Original, Parent, Child, Reject };
inline Admission admission(bool enabled, bool scoped, bool geometry, bool first, bool last) noexcept {
    if (scoped) return enabled && geometry && last ? Admission::Child : Admission::Reject;
    return enabled && geometry && first ? Admission::Parent : Admission::Original;
}
struct Policy { bool on=false, valid=true; };
inline const Policy& policy() {
    static const Policy p=[] { Policy r; const char* v=std::getenv("STRATA_PP_SUCCESSOR_CPU254_V220");
        if (!v || !std::strcmp(v,"0")) return r;
        if (!std::strcmp(v,"1")) r.on=true; else r.valid=false;
        return r; }();
    return p;
}
inline bool error(std::string& e,const char* what,int code) {
    e=std::string("V220 ")+what+": "+std::strerror(code); return false;
}
#if defined(__linux__)
inline bool exact(const cpu_set_t& m,int cpu) { return CPU_COUNT(&m)==1 && CPU_ISSET(cpu,&m); }
inline bool expect(int cpu,std::string& e,const char* role,bool log=false) {
    cpu_set_t m; CPU_ZERO(&m); int rc=pthread_getaffinity_np(pthread_self(),sizeof(m),&m);
    if(rc) return error(e,"get current affinity",rc);
    const int current=sched_getcpu();
    if(!exact(m,cpu)||current!=cpu) { e="V220 actual "+std::string(role)+" mask/currentCPU differs from required CPU"+std::to_string(cpu);return false; }
    if(log)std::fprintf(stderr,"V220 parent actual pid=%ld tid=%ld pthread=%lu mask=%d current=%d; successor target=254; selectedN1 caller contract\n",
        (long)getpid(),(long)syscall(SYS_gettid),(unsigned long)pthread_self(),cpu,current);
    return true;
}
class Scope {
    pthread_t owner_{}; cpu_set_t saved_{}; bool active_=false, saved_scope_=false;
    long long p0_=0,T_=0;int saved_cpu_=-1,target_cpu_=-1;
public:
    Scope()=default; Scope(const Scope&)=delete; Scope& operator=(const Scope&)=delete;
    bool finish(std::string& e) {
        if(!active_)return true;
        if(!pthread_equal(owner_,pthread_self())) {e="V220 affinity restore on a foreign thread";return false;}
        const int rc=pthread_setaffinity_np(owner_,sizeof(saved_),&saved_);
        // A failed restore is reported to the future; this owning async thread then exits.
        // Do not leave a second unreported destructor retry on an already-reported failure.
        active_=false;
        owning_scope_active_v238=saved_scope_;
        if(rc)return error(e,"restore affinity",rc);
        if(!expect(saved_cpu_,e,"restored successor"))return false;
        std::fprintf(stderr,"V220 successor restored pid=%ld tid=%ld pthread=%lu chunk=%lld+%lld mask=255 current=255\n",
            (long)getpid(),(long)syscall(SYS_gettid),(unsigned long)pthread_self(),p0_,T_);
        return true;
    }
    bool begin(std::string& e,long long p0,long long T,int target_cpu) {
        if(owning_scope_active_v238){e="V238 nested owning successor pin scope";return false;}
        saved_scope_=owning_scope_active_v238;
        saved_cpu_=sched_getcpu();target_cpu_=target_cpu;if((saved_cpu_!=255&&saved_cpu_!=254)||target_cpu_!=saved_cpu_-1||target_cpu_<253){e="threeGPU invalid stage-thread ownership";return false;}if(!expect(saved_cpu_,e,"initial successor"))return false;
        owner_=pthread_self(); p0_=p0;T_=T;
        int rc=pthread_getaffinity_np(owner_,sizeof(saved_),&saved_);
        if(rc)return error(e,"save affinity",rc);
        if(!exact(saved_,saved_cpu_)){e="V220 saved initial mask changed before bind";return false;}
        cpu_set_t target;CPU_ZERO(&target);CPU_SET(target_cpu_,&target);
        rc=pthread_setaffinity_np(owner_,sizeof(target),&target);
        if(rc)return error(e,"bind successor CPU254",rc);
        active_=true;
        owning_scope_active_v238=true;
        if(!expect(target_cpu_,e,"bound successor"))return false; // caller runs checked finish even on failure
        std::fprintf(stderr,"V220 successor admitted pid=%ld tid=%ld pthread=%lu chunk=%lld+%lld initial_mask=255 initial_current=255 target=254 mask=254 current=254\n",
            (long)getpid(),(long)syscall(SYS_gettid),(unsigned long)pthread_self(),p0_,T_);
        return true;
    }
    ~Scope() noexcept {
        if(active_) { // exception safety net; normal/error paths explicitly finish and report to future
            std::string e;
            try { if(finish(e))return; } catch(...) {}
            std::fprintf(stderr,"V220 FATAL: unreported owning-thread affinity restore failure\n");std::abort();
        }
    }
};
#else
inline bool expect(int,std::string& e,const char*,bool=false){e="V220 Linux only";return false;}
#endif
// Invoked only inside the existing successor async lambda, before run_impl/CUDA.
template<class F> bool run(F&& body,std::string& e,long long p0,long long T,int target_cpu) {
#if defined(__linux__)
    Scope scope;
    bool ok=false;
    try {
        if(!scope.begin(e,p0,T,target_cpu)) {
            std::string restored;
            if(!scope.finish(restored)) {e+="; "+restored;return false;}
            return false;
        }
        ok=body();
    } catch(...) {
        const auto original=std::current_exception();std::string restored;
        if(!scope.finish(restored)) {e="V220 exception plus restore failure: "+restored;return false;}
        std::rethrow_exception(original);
    }
    std::string restored;
    if(!scope.finish(restored)) {e="V220 restore failure after successor: "+restored;return false;}
    return ok;
#else
    (void)body;(void)p0;(void)T;e="V220 Linux only";return false;
#endif
}
} // namespace strata::prefill::successor_host_pin_v238
