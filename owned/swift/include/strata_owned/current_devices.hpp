#pragma once
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata_owned {
// Identity admission only. The launcher resolves CURRENT system indices into a
// fresh ordered UUID mask before CUDA starts. This helper neither selects extra
// devices nor enables peers/changes device state; startup P2P admission is owned
// by the common entrypoint. Call outside capture while binding the native path.
inline bool require_current_devices(int expected, std::string& error) noexcept {
    try {
        const char* mask = std::getenv("CUDA_VISIBLE_DEVICES");
        const char* snapshot = std::getenv("STRATA_RUN_GPU_UUIDS");
        if (expected < 1 || expected > 7 || !mask || !*mask || !snapshot ||
            std::strcmp(mask, snapshot)) {
            error = "A fresh restricted launch device snapshot is required";
            return false;
        }
        const std::string text(snapshot);
        std::vector<std::string> ordered;
        size_t begin = 0;
        for (;;) {
            const auto end = text.find(',', begin);
            const auto token = text.substr(begin, end == std::string::npos ? end : end - begin);
            if (token.empty()) { error = "Empty selected device identity"; return false; }
            for (const auto& prior : ordered) {
                if (prior == token) { error = "Duplicate selected device identity"; return false; }
            }
            ordered.push_back(token);
            if (ordered.size() > static_cast<size_t>(expected)) {
                error = "Launch device count differs from the admitted topology";
                return false;
            }
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        int count = 0;
        if (ordered.size() != static_cast<size_t>(expected) ||
            cudaGetDeviceCount(&count) != cudaSuccess || count != expected) {
            error = "CUDA-visible device count differs from the launch snapshot";
            return false;
        }
        for (int device = 0; device < count; ++device) {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
                error = "Unable to query selected CUDA device identity";
                return false;
            }
            const auto* u = reinterpret_cast<const unsigned char*>(properties.uuid.bytes);
            char actual[64];
            std::snprintf(actual, sizeof actual,
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
            if (ordered[device] != actual) {
                error = "Actual CUDA ordinal identity differs from the fresh launch snapshot";
                return false;
            }
        }
        return true;
    } catch (...) {
        error = "Selected device snapshot validation failed";
        return false;
    }
}
} // namespace strata_owned
