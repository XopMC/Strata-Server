#pragma once
#include <cstdint>
namespace strata::prefill::mmq {
// Pure dispatch admission, shared by integration and standalone component.
inline bool r56_v70_eligible(bool native, int gu_type, int dn_type, int64_t n, int64_t ff,
                            bool has_tail, bool had_down, bool wmma, bool fp4_gu, bool fp4_dn) {
    return native && gu_type == 40 && dn_type == 40 && n == 2560 && ff == 640 && has_tail &&
           !had_down && !wmma && !fp4_gu && !fp4_dn;
}
// ld is exactly1280 (ordinary GU) or2560 (R20 Dm alias).
// row_sd, when nonnull, is a compact nr-element output; caller passes row_sd+r0.
// diagnostic_h, when nonnull, is a compact nr*640-element output. No allocations.
void r56_v84_swiglu_d4(const float* gu, int64_t ld, int64_t nr, const int32_t* bounds,
                      int ng, const float* tails, int64_t row0, void* hq, void* stream,
                      float* diagnostic_h = nullptr, float* row_sd = nullptr);
}
