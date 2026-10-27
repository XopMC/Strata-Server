#pragma once
#include "strata/kernels/fused_gr.hpp"
namespace strata::kernels {
void fused_gr_check_v236();
int fused_gr_variant_v236();
void fused_gr_set_fast_v236(int);
bool fused_gr_read_multi_v236(const FusedGrArgs*, int, float*, void*, unsigned long long* = nullptr, int = 0);
}
