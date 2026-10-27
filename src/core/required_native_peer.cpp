#include "strata/core/required_native_peer.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>

// CPU policy tests compile this same transaction without linking/initializing CUDA.
// Production CUDA targets must NOT set this macro. Other backends do not gain a
// native-CUDA peer fallback by linking this file with the macro.
#if !defined(STRATA_NATIVE_PEER_CPU_ONLY)
#include <cuda_runtime.h>
#endif

namespace strata::core {
namespace {

std::string canonical_uuid(const std::string& uuid) {
    if (uuid.size() != 40 || uuid.compare(0, 4, "GPU-") != 0) return {};
    std::string out = uuid;
    for (std::size_t i = 4; i < out.size(); ++i) {
        if (i == 12 || i == 17 || i == 22 || i == 27) {
            if (out[i] != '-') return {};
        } else if (out[i] >= 'A' && out[i] <= 'F') {
            out[i] = char(out[i] - 'A' + 'a');
        } else if (!((out[i] >= '0' && out[i] <= '9') || (out[i] >= 'a' && out[i] <= 'f'))) {
            return {};
        }
    }
    return out;
}

void validate_options(const RequiredNativePeerOptions& o) {
    if (o.admitted_devices < 1 || o.admitted_devices > kNativePeerMaxDevices)
        throw std::runtime_error("admitted device count must be 1..7");
    if (!o.before_model_allocation_and_capture)
        throw std::runtime_error("admission requires serialized GPU-child startup before model allocation/capture");
    if (o.binding != NativePeerBinding::CallerValidatedVisibility &&
        o.binding != NativePeerBinding::FreshUuidSnapshot)
        throw std::runtime_error("fresh launch selection binding is required");
    const auto n = static_cast<std::size_t>(o.admitted_devices);
    if ((!o.current_system_indices.empty() && o.current_system_indices.size() != n) ||
        (o.binding == NativePeerBinding::FreshUuidSnapshot && o.current_system_indices.size() != n))
        throw std::runtime_error("current system-index snapshot count mismatch");
    for (std::size_t i = 0; i < o.current_system_indices.size(); ++i) {
        if (o.current_system_indices[i] < 0) throw std::runtime_error("negative current system index");
        for (std::size_t j = 0; j < i; ++j)
            if (o.current_system_indices[i] == o.current_system_indices[j])
                throw std::runtime_error("duplicate current system index");
    }
    if (o.binding == NativePeerBinding::CallerValidatedVisibility) {
        if (!o.transient_cuda_uuids.empty())
            throw std::runtime_error("UUID snapshot supplied without FreshUuidSnapshot binding");
        return;
    }
    if (o.transient_cuda_uuids.size() != n) throw std::runtime_error("transient UUID snapshot count mismatch");
    for (std::size_t i = 0; i < n; ++i) {
        const auto uuid = canonical_uuid(o.transient_cuda_uuids[i]);
        if (uuid.empty()) throw std::runtime_error("malformed full transient GPU UUID");
        for (std::size_t j = 0; j < i; ++j)
            if (uuid == canonical_uuid(o.transient_cuda_uuids[j]))
                throw std::runtime_error("duplicate transient GPU UUID");
    }
}

void checked(bool success, const std::string& operation, const std::string& detail) {
    if (!success) throw std::runtime_error(operation + ": " + detail);
}

#if !defined(STRATA_NATIVE_PEER_CPU_ONLY)
class CudaRuntime final : public native_peer_detail::Runtime {
    static bool check(cudaError_t status, std::string& error) {
        if (status == cudaSuccess) return true;
        error = cudaGetErrorString(status);
        return false;
    }
public:
    bool device_count(int& n, std::string& e) override { return check(cudaGetDeviceCount(&n), e); }
    bool current_device(int& d, std::string& e) override { return check(cudaGetDevice(&d), e); }
    bool set_device(int d, std::string& e) override { return check(cudaSetDevice(d), e); }
    bool identity(int d, std::string& uuid, bool& uva, std::string& error) override {
        cudaDeviceProp p{};
        if (!check(cudaGetDeviceProperties(&p, d), error)) return false;
        const auto* b = reinterpret_cast<const unsigned char*>(p.uuid.bytes);
        char text[41];
        std::snprintf(text, sizeof(text),
                      "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                      b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
        uuid = text;
        uva = p.unifiedAddressing != 0;
        return true;
    }
    bool default_stream_not_capturing(bool& inactive, std::string& error) override {
        cudaStreamCaptureStatus status{};
        if (!check(cudaStreamIsCapturing(nullptr, &status), error)) return false;
        inactive = status == cudaStreamCaptureStatusNone;
        return true;
    }
    bool can_access(int from, int to, bool& capable, std::string& error) override {
        int can = 0;
        if (!check(cudaDeviceCanAccessPeer(&can, from, to), error)) return false;
        capable = can != 0;
        return true;
    }
    native_peer_detail::EnableResult enable(int to, std::string& error) override {
        using Result = native_peer_detail::EnableResult;
        const auto status = cudaDeviceEnablePeerAccess(to, 0);
        if (status == cudaSuccess) return Result::Enabled;
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            // Clear ONLY this expected status; an unrelated pending error refuses
            // startup. Never disable an edge inherited from another owner.
            const auto pending = cudaGetLastError();
            if (pending == cudaSuccess || pending == cudaErrorPeerAccessAlreadyEnabled)
                return Result::AlreadyEnabled;
            check(pending, error);
        } else {
            check(status, error);
        }
        return Result::Failed;
    }
    bool disable(int to, std::string& error) override { return check(cudaDeviceDisablePeerAccess(to), error); }
};
#endif

}  // namespace

bool native_peer_detail::require_with_runtime(const RequiredNativePeerOptions& o, Runtime& runtime,
                                              RequiredNativePeerReceipt& receipt, std::string& error) {
    receipt = RequiredNativePeerReceipt{};
    error.clear();
    int previous = -1;
    std::array<std::pair<int, int>, kNativePeerMaxDevices * (kNativePeerMaxDevices - 1)> created{};
    int created_count = 0;
    int existing_count = 0;
    try {
        validate_options(o);  // Invalid policy makes ZERO runtime calls.
        std::string why;
        int visible = 0;
        checked(runtime.device_count(visible, why), "visible device count", why);
        if (visible != o.admitted_devices) throw std::runtime_error("selected visible device count mismatch");
        int entry = -1;
        checked(runtime.current_device(entry, why), "save current device", why);
        if (entry < 0 || entry >= visible) throw std::runtime_error("entry device outside selected visibility");
        previous = entry;  // Never select an unvalidated ordinal during rollback.
        // Preflight the complete selected cohort before creating any mapping.
        for (int d = 0; d < visible; ++d) {
            checked(runtime.set_device(d, why), "preflight device " + std::to_string(d), why);
            bool inactive = false;
            checked(runtime.default_stream_not_capturing(inactive, why), "pre-capture admission", why);
            if (!inactive) throw std::runtime_error("native peer admission must precede capture");
            std::string uuid;
            bool uva = false;
            checked(runtime.identity(d, uuid, uva, why), "selected device identity", why);
            if (!uva) throw std::runtime_error("selected device lacks unified addressing");
            if (o.binding == NativePeerBinding::FreshUuidSnapshot &&
                canonical_uuid(uuid) != canonical_uuid(o.transient_cuda_uuids[static_cast<std::size_t>(d)]))
                throw std::runtime_error("fresh CUDA UUID/order mismatch at ordinal " + std::to_string(d));
        }
        for (int from = 0; from < visible; ++from) {
            for (int to = 0; to < visible; ++to) if (from != to) {
                bool capable = false;
                const auto pair = std::to_string(from) + "->" + std::to_string(to);
                checked(runtime.can_access(from, to, capable, why), "native capability " + pair, why);
                if (!capable) throw std::runtime_error("required native peer direction unavailable: " + pair);
            }
        }
        for (int from = 0; from < visible; ++from) {
            if (visible > 1) checked(runtime.set_device(from, why), "enable owner", why);
            for (int to = 0; to < visible; ++to) if (from != to) {
                const auto status = runtime.enable(to, why);
                if (status == EnableResult::Failed)
                    throw std::runtime_error("native enable " + std::to_string(from) + "->" +
                                             std::to_string(to) + ": " + why);
                if (status == EnableResult::Enabled) created[created_count++] = {from, to};
                else ++existing_count;
            }
        }
        checked(runtime.set_device(previous, why), "restore current device", why);
        receipt.devices = visible;
        receipt.directed_pairs = visible * (visible - 1);
        receipt.newly_enabled_pairs = created_count;
        receipt.already_enabled_pairs = existing_count;
        for (std::size_t i = 0; i < o.current_system_indices.size(); ++i)
            receipt.current_system_indices[i] = o.current_system_indices[i];
        receipt.runtime_order_verified = o.binding == NativePeerBinding::FreshUuidSnapshot;
        receipt.admitted = true;
        return true;
    } catch (const std::exception& e) {
        error = std::string("required native P2P: ") + e.what();
    } catch (...) {
        error = "required native P2P: startup metadata/runtime exception";
    }
    // Continue cleanup after a failed edge; never disable on the wrong device.
    // Only startup mappings in `created` belong to this failed transaction.
    for (int i = created_count - 1; i >= 0; --i) {
        try {
            std::string why;
            if (!runtime.set_device(created[i].first, why) || !runtime.disable(created[i].second, why))
                receipt.cleanup_complete = false;
        } catch (...) { receipt.cleanup_complete = false; }
    }
    if (previous >= 0) {
        try {
            std::string why;
            if (!runtime.set_device(previous, why)) receipt.cleanup_complete = false;
        } catch (...) { receipt.cleanup_complete = false; }
    }
    if (!receipt.cleanup_complete) error += "; admission cleanup failed, terminate GPU child";
    return false;
}

bool require_native_peer(const RequiredNativePeerOptions& o,
                         RequiredNativePeerReceipt& receipt, std::string& error) {
#if defined(STRATA_NATIVE_PEER_CPU_ONLY)
    (void)o;
    receipt = RequiredNativePeerReceipt{};
    error = "required native P2P: CUDA adapter is absent in this CPU-only policy-test build";
    return false;
#else
    CudaRuntime runtime;
    if (!native_peer_detail::require_with_runtime(o, runtime, receipt, error)) return false;
    std::fprintf(stderr, "STRATA_P2P_DEFAULT devices=%d directed_pairs=%d native_required=1 host_fallback=0 order_verified=%d\n",
                 receipt.devices, receipt.directed_pairs, receipt.runtime_order_verified ? 1 : 0);
    return true;
#endif
}

}  // namespace strata::core
