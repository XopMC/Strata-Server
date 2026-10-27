#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels::cpu {
bool nvfp4_512_fits_v221(int);
void nvfp4_512_gu_rows_v221(const uint8_t*,size_t,size_t,int,const void*const*,int,float*const*,int,int,float,float);
void nvfp4_512_rows_v221(const uint8_t*,size_t,int,const void*const*,int,float*const*,int,int,float);
}
