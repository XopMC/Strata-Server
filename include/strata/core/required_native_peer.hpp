// Required native CUDA peer admission. No allocations, copies or fallback transport.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace strata::core {

constexpr int kNativePeerMaxDevices = 7;

enum class NativePeerBinding {
    Unspecified,
    // The launcher has already restricted AND verified the current selection/order.
    // Useful when it owns the transient NVML-index -> CUDA-visibility binding.
    CallerValidatedVisibility,
    // Compare each CUDA ordinal with this launch's freshly resolved GPU UUID.
    FreshUuidSnapshot
};

struct RequiredNativePeerOptions {
    int admitted_devices = 0;  // Exactly the visible count, not the whole system count.
    NativePeerBinding binding = NativePeerBinding::Unspecified;
    // Diagnostic current SYSTEM indices. Required for FreshUuidSnapshot; otherwise
    // optional. These are not CUDA ordinals, PCI-order indices or persistent IDs.
    std::vector<int> current_system_indices;
    // Full GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx identifiers, in CUDA ordinal
    // order, resolved afresh for THIS launch. Never a saved UUID allowlist.
    std::vector<std::string> transient_cuda_uuids;
    // Caller assertion: GPU child after fork, serialized startup, no model device
    // allocation and no capture on ANY stream yet. CUDA cannot prove this history.
    bool before_model_allocation_and_capture = false;
};

struct RequiredNativePeerReceipt {
    bool admitted = false;
    bool runtime_order_verified = false;
    bool cleanup_complete = true;  // False on rollback/restore failure: terminate child.
    int devices = 0;
    int directed_pairs = 0;
    int newly_enabled_pairs = 0;
    int already_enabled_pairs = 0;
    // Optional launch diagnostics, copied from options; -1 when caller omitted them.
    std::array<int, kNativePeerMaxDevices> current_system_indices{{-1, -1, -1, -1, -1, -1, -1}};
};

// Call once in the selected CUDA child BEFORE model allocation/capture. Checks all
// selected devices and all N*(N-1) directions, including idle selected helpers.
// Restores the entry device. On failure, rolls back ONLY mappings it just enabled.
// Success leaves mappings enabled for context lifetime; receipt destruction does
// not disable them. On false, refuse startup; never retry with host/legacy transport.
// No environment variables are changed, no excluded devices are selected, and no
// driver/power/reset actions are performed. See docs/XOPMC_P2P.md for caller duties.
bool require_native_peer(const RequiredNativePeerOptions& options,
                         RequiredNativePeerReceipt& receipt, std::string& error);

namespace native_peer_detail {

// Narrow dependency seam for CPU-only policy tests. It is NOT a transport backend.
// The production entry above always uses CUDA; profiles must not inject a backend.
enum class EnableResult { Enabled, AlreadyEnabled, Failed };
class Runtime {
public:
    virtual ~Runtime() = default;
    virtual bool device_count(int& count, std::string& error) = 0;
    virtual bool current_device(int& ordinal, std::string& error) = 0;
    virtual bool set_device(int ordinal, std::string& error) = 0;
    virtual bool identity(int ordinal, std::string& uuid, bool& unified_addressing,
                          std::string& error) = 0;
    virtual bool default_stream_not_capturing(bool& inactive, std::string& error) = 0;
    virtual bool can_access(int from, int to, bool& capable, std::string& error) = 0;
    virtual EnableResult enable(int to, std::string& error) = 0;
    virtual bool disable(int to, std::string& error) = 0;
};

bool require_with_runtime(const RequiredNativePeerOptions& options, Runtime& runtime,
                          RequiredNativePeerReceipt& receipt, std::string& error);

}  // namespace native_peer_detail
}  // namespace strata::core
