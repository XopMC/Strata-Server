#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace capacity_v366 {
inline bool valid(int64_t n){return n==16384||n==32768;}
inline bool enabled(){const char*v=std::getenv("STRATA_TWO_SMALL_LOAN_V366");if(!v)return false;if(!std::strcmp(v,"1"))return true;throw std::runtime_error("V366 explicit loan policy must equal1");}
inline int64_t plan(int64_t tokens,int64_t exact){if(exact>0)return exact;return enabled()&&tokens>0&&tokens<=16384?16384:32768;}
}
