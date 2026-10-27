// Transient visibility for three historical diagnostic exports and the byte-only
// inverse's existing device guard. Production startup still requires native
// all-pairs P2P admission before allocation/capture; this is not that admission.
#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstdlib>
#include <cstring>

namespace strata::ds41::publication {
using DiagnosticSnapshot = std::array<std::array<unsigned char, 16>, 2>;

inline int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Fresh launch metadata, never a saved identity allowlist. The selected launcher
// owns current-system-index resolution before CUDA initialization. This follows
// the selected DS launcher's transient UUID visibility convention, not PCI order.
inline bool diagnostic_snapshot(int expected, DiagnosticSnapshot& snapshot) {
    if (expected != 1 && expected != 2) return false;
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    const char* fresh = std::getenv("STRATA_RUN_GPU_UUIDS");
    if (!visible || !fresh || !*visible || std::strcmp(visible, fresh)) return false;
    if (std::strlen(fresh) != static_cast<std::size_t>(expected * 40 + expected - 1)) return false;
    for (int d = 0; d < expected; ++d) {
        const char* uuid = fresh + d * 41;
        if (std::memcmp(uuid, "GPU-", 4)) return false;
        int byte = 0, high = -1;
        for (int i = 4; i < 40; ++i) {
            if (i == 12 || i == 17 || i == 22 || i == 27) {
                if (uuid[i] != '-') return false;
                continue;
            }
            const int digit = hex_digit(uuid[i]);
            if (digit < 0) return false;
            if (high < 0) high = digit;
            else { snapshot[d][byte++] = static_cast<unsigned char>((high << 4) | digit); high = -1; }
        }
        if (byte != 16 || high != -1 || (d + 1 < expected && uuid[40] != ',')) return false;
    }
    return expected != 2 || snapshot[0] != snapshot[1];
}

inline bool metadata_snapshot(int expected) {
    DiagnosticSnapshot snapshot{};
    return diagnostic_snapshot(expected, snapshot);
}

inline bool runtime_snapshot(int expected) {
    DiagnosticSnapshot snapshot{};
    if (!diagnostic_snapshot(expected, snapshot)) return false;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count != expected) return false;
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp property{};
        if (cudaGetDeviceProperties(&property, d) != cudaSuccess ||
            std::memcmp(property.uuid.bytes, snapshot[d].data(), 16)) return false;
    }
    return true;
}
}  // namespace strata::ds41::publication
