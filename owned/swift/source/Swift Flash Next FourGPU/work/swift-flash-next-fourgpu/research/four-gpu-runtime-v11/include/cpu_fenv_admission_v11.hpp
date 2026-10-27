#pragma once
#include "strata/core/progress.hpp"
#include <atomic>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
namespace cpu_fenv_admission_v11 {
struct Pool {const void*identity=nullptr;uint64_t generation=0;int expected=0;bool ready=false;std::vector<long>workers;};
inline std::mutex&mutex(){static std::mutex m;return m;}
inline std::map<const void*,std::shared_ptr<Pool>>&pools(){static std::map<const void*,std::shared_ptr<Pool>>p;return p;}
inline long tid(){return long(syscall(SYS_gettid));}
inline bool controls(){return std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&0xe040u)==0;}
inline void require_current(){if(!controls())throw std::runtime_error("V11 CPU semantics requires actual caller RN/FTZ0/DAZ0; no repair");}
inline void check_or_abort(const char*role){if(controls())return;std::fprintf(stderr,"V11_FENV_REFUSE role=%s tid=%ld round=%d mxcsr=0x%x\n",role,tid(),std::fegetround(),_mm_getcsr());strata::core::release_gpu_waits(stderr);std::fflush(stderr);std::abort();}
inline std::shared_ptr<Pool>begin_pool(const void*identity,int expected){require_current();if(!identity||expected<0)throw std::runtime_error("V11 pool registration shape");static uint64_t generation=0;std::lock_guard<std::mutex>lock(mutex());if(pools().count(identity))throw std::runtime_error("V11 stale pool lifetime registration");auto p=std::make_shared<Pool>();p->identity=identity;p->generation=++generation;p->expected=expected;p->workers.assign(size_t(expected),0);pools()[identity]=p;return p;}
inline void worker(const std::shared_ptr<Pool>&p,int id){check_or_abort("actual-pool-worker-start");std::lock_guard<std::mutex>lock(mutex());if(!p||id<0||id>=p->expected||p->workers[size_t(id)])std::abort();p->workers[size_t(id)]=tid();std::fprintf(stderr,"V11_FENV_WORKER generation=%llu worker=%d tid=%ld mxcsr=0x%x\n",(unsigned long long)p->generation,id,tid(),_mm_getcsr());}
inline void ready(const std::shared_ptr<Pool>&p){require_current();std::lock_guard<std::mutex>lock(mutex());for(auto id:p->workers)if(!id)throw std::runtime_error("V11 actual worker admission missing");p->ready=true;std::fprintf(stderr,"V11_FENV_POOL_READY generation=%llu actual_workers=%d host_tid=%ld\n",(unsigned long long)p->generation,p->expected,tid());}
inline void retire(const void*identity){std::lock_guard<std::mutex>lock(mutex());if(pools().erase(identity)!=1)std::abort();}
inline void require_ready(){require_current();std::lock_guard<std::mutex>lock(mutex());if(pools().empty())throw std::runtime_error("V11 requires actual instrumented model pool; old workers are unqualified");for(auto&entry:pools()){const auto&p=*entry.second;if(!p.ready)throw std::runtime_error("V11 model pool not fully admitted");for(auto id:p.workers)if(!id)throw std::runtime_error("V11 missing current model worker");}}
// Actual process default/versioned expf, selected libm and CPU leaf admission.
void require_process();
}
