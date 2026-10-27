#pragma once
#include "strata/kernels/iq_kernels.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
namespace strata::kernels::actstage_v264 {
static_assert(sizeof(NativeExpertLayout)==88,"V264 requires current native layout, never old64B");
inline bool enabled(){static const bool v=[] {const char*p=std::getenv("STRATA_SWIFT_NVFP4_ACTSTAGE_V264");return p&&!std::strcmp(p,"1");}();return v;}
inline bool fits(const NativeExpertLayout&L){return L.gu_type==40&&L.d_type==40&&L.n_embd==2560&&L.n_ff==640&&L.gu_row==1440&&L.d_row==360&&L.up_off==921600&&L.down_off==1843200&&L.tail_off==2764800&&L.bytes==2764816&&L.had_block==0&&!L.d_signs;}
inline void note(){static std::atomic<unsigned long long> n{0};if(!n.fetch_add(1,std::memory_order_relaxed))std::fprintf(stderr,"ACTSTAGE_V264 ADMIT GU40 NC=4 rows=16 shared=11520; host submissions/captures only NOT graph replay count\n");}
}
