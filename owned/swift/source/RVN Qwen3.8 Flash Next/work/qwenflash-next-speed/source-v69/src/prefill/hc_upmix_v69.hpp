#pragma once
#include <cstdint>
// Private v69 symbol; original Prefill class, public headers and scratch allocations stay unchanged.
namespace strata::prefill {
bool gr_upmix_v69(const uint16_t* lo16,const uint16_t* w_up,const float* R,const float* rs,
                  const float* w_norm,float* mixed,uint16_t* mixed16,uint16_t* mixed_h,
                  int64_t T,void* stream);
}
