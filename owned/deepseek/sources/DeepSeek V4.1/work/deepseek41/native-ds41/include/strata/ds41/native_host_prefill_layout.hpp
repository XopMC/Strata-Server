#pragma once
#include <cstddef>
namespace strata::ds41 {
struct NativeHostPrefillLayout {
    size_t hc_bytes,routed_bytes,topk_bytes,candidate_bytes,core_bytes,admitted_bytes;
};
// Two GiB cover route/mixing metadata and bounded pinned transfer buffers.
// Check budget and current available RAM BEFORE any large allocation.
NativeHostPrefillLayout native_host_prefill_layout(int rows,size_t budget,size_t available,size_t reserve);
}
