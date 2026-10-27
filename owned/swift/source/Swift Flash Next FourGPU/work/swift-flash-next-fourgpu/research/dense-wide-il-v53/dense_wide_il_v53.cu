#include "dense_wide_il_v53.hpp"
#include <stdexcept>
#include "q6_il_leaf_v53.cuh"
#include <cstdint>
#include <cstdio>
namespace swift_wide53 {
int effective_rows(int requested){
    if(requested!=1&&requested!=2&&requested!=4)throw std::invalid_argument("WIDE53 requested rows must be1/2/4");
    return requested==4?2:requested;
}
void launch(const void*w,const void*il,float*y,int rows,int T,int requested,cudaStream_t stream){
    (void)effective_rows(requested);
    if(rows<1||rows>Out||T<5||T>8||!stream||!w||!il||!y||
       (reinterpret_cast<uintptr_t>(w)|reinterpret_cast<uintptr_t>(il)|reinterpret_cast<uintptr_t>(y))%4)
        throw std::invalid_argument("WIDE53 only aligned native14/N2560/M1..10240/T5..8 explicit owner stream");
    swift_wide53_leaf::launch_original_wide(w,il,y,In,rows,T,requested,stream);
    auto e=cudaGetLastError();if(e!=cudaSuccess){std::fprintf(stderr,"WIDE53_FIRST_API_ERROR private wide launch: %s\n",cudaGetErrorString(e));std::fflush(stderr);throw std::runtime_error(cudaGetErrorString(e));}
}
} // namespace swift_wide53
