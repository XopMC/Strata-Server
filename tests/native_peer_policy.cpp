// CPU-only tests of the actual admission transaction. No CUDA headers or runtime.
// Compile both this file and required_native_peer.cpp with STRATA_NATIVE_PEER_CPU_ONLY.
#include "strata/core/required_native_peer.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#if !defined(STRATA_NATIVE_PEER_CPU_ONLY)
#error "native_peer_policy is a CPU-isolated test; define STRATA_NATIVE_PEER_CPU_ONLY"
#endif

using namespace strata::core;
using namespace strata::core::native_peer_detail;

static int failures = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static std::string uuid(int d) {
    char text[41];
    std::snprintf(text, sizeof(text), "GPU-12345678-1234-ABCD-1234-%012x", d + 1);
    return text;
}

static RequiredNativePeerOptions options(int n) {
    RequiredNativePeerOptions o;
    o.admitted_devices = n;
    o.binding = NativePeerBinding::FreshUuidSnapshot;
    o.before_model_allocation_and_capture = true;
    for (int d = 0; d < n; ++d) {
        // Deliberately not ordinal order or contiguous indices: a transient UUID
        // binding must not silently reinterpret system indices as PCI/CUDA order.
        o.current_system_indices.push_back(100 - d * 3);
        o.transient_cuda_uuids.push_back(uuid(d));
    }
    return o;
}

class Fake final : public Runtime {
public:
    using Matrix = std::array<std::array<bool, 7>, 7>;
    int n, current, calls = 0, fail_at = 0, throw_at = 0;
    int captures_on = -1, no_uva_on = -1, wrong_uuid_on = -1;
    bool fail_disable = false, complete_preflight_at_enable = true;
    Matrix enabled{}, capable{};
    std::vector<std::pair<int, int>> queries, enables, disables;
    std::vector<std::string> log;

    explicit Fake(int count) : n(count), current(count - 1) {
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) capable[a][b] = a != b;
    }
    bool probe(const char* name, std::string& error) {
        ++calls;
        log.emplace_back(name);
        if (calls == throw_at) throw std::runtime_error("injected runtime exception");
        if (calls == fail_at) { error = "injected failure"; return false; }
        return true;
    }
    bool device_count(int& count, std::string& error) override {
        if (!probe("count", error)) return false;
        count = n; return true;
    }
    bool current_device(int& ordinal, std::string& error) override {
        if (!probe("current", error)) return false;
        ordinal = current; return true;
    }
    bool set_device(int ordinal, std::string& error) override {
        CHECK(ordinal >= 0 && ordinal < n);
        if (!probe("set", error)) return false;
        current = ordinal; return true;
    }
    bool identity(int ordinal, std::string& id, bool& uva, std::string& error) override {
        CHECK(ordinal >= 0 && ordinal < n && current == ordinal);
        if (!probe("identity", error)) return false;
        id = uuid(ordinal == wrong_uuid_on ? (ordinal + 1) % n : ordinal);
        uva = ordinal != no_uva_on;
        return true;
    }
    bool default_stream_not_capturing(bool& inactive, std::string& error) override {
        if (!probe("capture", error)) return false;
        inactive = current != captures_on; return true;
    }
    bool can_access(int from, int to, bool& can, std::string& error) override {
        CHECK(from >= 0 && from < n && to >= 0 && to < n && from != to);
        if (!probe("capability", error)) return false;
        queries.emplace_back(from, to);
        can = capable[from][to]; return true;
    }
    EnableResult enable(int to, std::string& error) override {
        CHECK(current >= 0 && current < n && to >= 0 && to < n && current != to);
        if (!probe("enable", error)) return EnableResult::Failed;
        complete_preflight_at_enable &= queries.size() == static_cast<std::size_t>(n * (n - 1));
        enables.emplace_back(current, to);
        if (enabled[current][to]) return EnableResult::AlreadyEnabled;
        enabled[current][to] = true;
        return EnableResult::Enabled;
    }
    bool disable(int to, std::string& error) override {
        CHECK(current >= 0 && current < n && to >= 0 && to < n && current != to);
        if (!probe("disable", error) || fail_disable) { error = "injected cleanup failure"; return false; }
        disables.emplace_back(current, to);
        CHECK(enabled[current][to]);
        enabled[current][to] = false;
        return true;
    }
};

static bool run(const RequiredNativePeerOptions& o, Fake& f,
                RequiredNativePeerReceipt& r, std::string& error) {
    return require_with_runtime(o, f, r, error);
}

static void bad_options(const RequiredNativePeerOptions& o) {
    Fake f(3);
    RequiredNativePeerReceipt r;
    r.admitted = true; r.devices = 7;  // A failed call must not retain old credit.
    std::string error = "old";
    CHECK(!run(o, f, r, error));
    CHECK(!r.admitted && r.devices == 0 && r.directed_pairs == 0 && r.cleanup_complete);
    CHECK(!error.empty() && f.calls == 0);
}

int main() {
    // All selected counts; no unrelated ordinal or self-direction is touched.
    for (int n = 1; n <= 7; ++n) {
        Fake f(n);
        RequiredNativePeerReceipt r;
        std::string error = "stale";
        CHECK(run(options(n), f, r, error));
        CHECK(error.empty() && r.admitted && r.runtime_order_verified && r.cleanup_complete);
        CHECK(r.devices == n && r.directed_pairs == n * (n - 1));
        CHECK(r.newly_enabled_pairs == n * (n - 1) && r.already_enabled_pairs == 0);
        CHECK(f.current == n - 1 && f.complete_preflight_at_enable);
        CHECK(f.queries.size() == static_cast<std::size_t>(n * (n - 1)));
        CHECK(f.enables.size() == f.queries.size() && f.disables.empty());
        for (int d = 0; d < n; ++d) CHECK(r.current_system_indices[d] == 100 - d * 3);
        const auto enabled = f.enabled;
        CHECK(run(options(n), f, r, error));
        CHECK(r.newly_enabled_pairs == 0 && r.already_enabled_pairs == n * (n - 1));
        CHECK(f.enabled == enabled && f.disables.empty());
    }
    { auto o = options(2); o.binding = NativePeerBinding::CallerValidatedVisibility;
      o.transient_cuda_uuids.clear(); o.current_system_indices.clear();
      Fake f(2); RequiredNativePeerReceipt r; std::string error;
      CHECK(run(o, f, r, error)); CHECK(r.admitted && !r.runtime_order_verified);
      CHECK(r.current_system_indices[0] == -1); }
    { auto o = options(3); o.admitted_devices = 0; bad_options(o); }
    { auto o = options(3); o.admitted_devices = 8; bad_options(o); }
    { auto o = options(3); o.before_model_allocation_and_capture = false; bad_options(o); }
    { auto o = options(3); o.binding = NativePeerBinding::Unspecified; bad_options(o); }
    { auto o = options(3); o.binding = static_cast<NativePeerBinding>(99); bad_options(o); }
    { auto o = options(3); o.current_system_indices.clear(); bad_options(o); }
    { auto o = options(3); o.current_system_indices[1] = -1; bad_options(o); }
    { auto o = options(3); o.current_system_indices[1] = o.current_system_indices[0]; bad_options(o); }
    { auto o = options(3); o.transient_cuda_uuids.pop_back(); bad_options(o); }
    { auto o = options(3); o.transient_cuda_uuids[0] = "GPU-partial"; bad_options(o); }
    { auto o = options(3); o.transient_cuda_uuids[0][12] = 'x'; bad_options(o); }
    { auto o = options(3); o.transient_cuda_uuids[0][39] = 'z'; bad_options(o); }
    { auto o = options(3); o.transient_cuda_uuids[1] = o.transient_cuda_uuids[0]; bad_options(o); }
    { auto o = options(3); o.binding = NativePeerBinding::CallerValidatedVisibility; bad_options(o); }
    { auto o = options(3); std::swap(o.transient_cuda_uuids[0], o.transient_cuda_uuids[1]);
      Fake f(3); RequiredNativePeerReceipt r; std::string error;
      CHECK(!run(o, f, r, error)); CHECK(error.find("order mismatch") != std::string::npos);
      CHECK(f.enables.empty() && f.current == 2 && !r.admitted); }
    for (int kind = 0; kind < 4; ++kind) {
        Fake f(kind == 0 ? 2 : 3);
        if (kind == 1) f.captures_on = 1;
        if (kind == 2) f.no_uva_on = 1;
        if (kind == 3) f.wrong_uuid_on = 1;
        RequiredNativePeerReceipt r; std::string error;
        CHECK(!run(options(3), f, r, error));
        CHECK(!r.admitted && r.cleanup_complete && f.enables.empty());
        CHECK(f.current == f.n - 1 && !error.empty());
    }
    { // A corrupt/out-of-scope runtime entry ordinal is not a restore target.
      Fake f(3); f.current = 7; RequiredNativePeerReceipt receipt; std::string why;
      CHECK(!run(options(3), f, receipt, why));
      CHECK(f.calls == 2 && f.current == 7 && f.enables.empty()); }
    { // Even single-device admission checks capture state, despite zero pair edges.
      Fake f(1); f.captures_on = 0; RequiredNativePeerReceipt r; std::string error;
      CHECK(!run(options(1), f, r, error)); CHECK(f.enables.empty()); }
    for (int from = 0; from < 3; ++from) for (int to = 0; to < 3; ++to) if (from != to) {
        Fake f(3); f.capable[from][to] = false;
        RequiredNativePeerReceipt r; std::string error;
        CHECK(!run(options(3), f, r, error));
        CHECK(!r.admitted && r.cleanup_complete && f.enables.empty());
        CHECK(error.find(std::to_string(from) + "->" + std::to_string(to)) != std::string::npos);
    }
    // Inject a failure/exception at EVERY successful-path runtime call. Existing
    // mappings survive; new ones are undone in reverse enable order.
    Fake baseline(3); baseline.enabled[0][1] = baseline.enabled[2][0] = true;
    const auto initial = baseline.enabled;
    RequiredNativePeerReceipt r; std::string error;
    CHECK(run(options(3), baseline, r, error));
    for (int mode = 0; mode < 2; ++mode) for (int at = 1; at <= baseline.calls; ++at) {
        Fake f(3); f.enabled = initial;
        if (mode == 0) f.fail_at = at; else f.throw_at = at;
        CHECK(!run(options(3), f, r, error));
        CHECK(!r.admitted && r.cleanup_complete && !error.empty());
        CHECK(f.enabled == initial && f.current == 2);
        CHECK(!f.enabled[0][0] && !f.enabled[1][1] && !f.enabled[2][2]);
        std::vector<std::pair<int, int>> newly_enabled;
        for (const auto& edge : f.enables) if (!initial[edge.first][edge.second]) newly_enabled.push_back(edge);
        CHECK(f.disables.size() == newly_enabled.size());
        for (std::size_t i = 0; i < f.disables.size(); ++i)
            CHECK(f.disables[i] == newly_enabled[newly_enabled.size() - 1 - i]);
    }
    { // Cleanup failure remains a refusal, with no valid startup receipt.
      Fake f(3); f.fail_disable = true;
      for (std::size_t i = 0; i < baseline.log.size(); ++i)
          if (baseline.log[i] == "enable") { f.fail_at = static_cast<int>(i) + 3; break; }
      CHECK(!run(options(3), f, r, error));
      CHECK(!r.admitted && !r.cleanup_complete);
      CHECK(error.find("terminate GPU child") != std::string::npos); }
    // The production API cannot silently succeed in a CPU-only policy-test build.
    CHECK(!require_native_peer(options(2), r, error));
    CHECK(!r.admitted && error.find("CUDA adapter is absent") != std::string::npos);
    if (failures == 0) std::printf("native_peer_policy: PASS (%d CPU checks; no CUDA calls)\n", checks);
    return failures == 0 ? 0 : 1;
}
