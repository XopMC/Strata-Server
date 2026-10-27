#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace control_quota_v12 {
inline std::vector<std::array<uint64_t,3>>state(int64_t context){
 if(context==262144)return {{7460,7460,0xd7da402f13533b39ULL},{6645,6645,0xc9066f18d999b566ULL}};
 if(context==524288)return {{7285,7285,0x07bd3a565b2172ddULL},{6470,6470,0x4ce6be5343dbb82cULL}};
 if(context==1048576)return {{7083,7083,0xa177f6b4af23510bULL},{6267,6267,0x6e8a73f68de9aedaULL}};
 throw std::runtime_error("CONTROL12 unsupported context");
}
inline size_t slots(int64_t context,int d){if(d<0||d>1)throw std::runtime_error("CONTROL12 two homes only");return size_t(state(context)[size_t(d)][0]);}
}
