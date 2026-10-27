#pragma once

// Auxiliary scheduling only. All emitted tokens still pass the unchanged target verifier.
// Keep the stock first three proposals; extend a strong chain only when its estimated
// marginal survival pays for an additional draft and a wider target verification.
// Independent implementation informed by SpecDec++ and NInfer's measured-cost controller.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace swift78 {
class PrefixPolicy {
public:
    static constexpr int kDepth = 7;
    static constexpr int kBins = 8;
    void reset() { *this = PrefixPolicy{}; }

    double probability(int depth, double confidence) const {
        if (depth < 0 || depth >= kDepth || !std::isfinite(confidence)) return 0.;
        const int b = bin(confidence);
        // Native softmax confidence is a prior, never an asserted acceptance probability.
        return (4. * std::clamp(confidence, 0., 1.) + yes_[depth][b]) /
               (4. + seen_[depth][b]);
    }

    void observe(int t, int accepted, const float* confidence, bool suffix, double verify_ms) {
        if (t < 1 || t > 8 || accepted < 0 || accepted >= t) return;
        if (std::isfinite(verify_ms) && verify_ms > 0. && verify_ms < 100.) {
            // Discard the first invocation at each width: it may include graph capture.
            if (cost_seen_[t]++) {
                const double alpha = cost_seen_[t] < 9 ? 1. / (cost_seen_[t] - 1.) : .125;
                cost_[t] += alpha * (verify_ms - cost_[t]);
            }
        }
        if (suffix || !confidence) return;
        // A proposal after the first rejection has an invalid conditioning prefix.
        const int valid = std::min(t - 1, accepted + 1);
        for (int j = 0; j < valid; ++j) {
            if (!std::isfinite(confidence[j]) || confidence[j] < 0.f || confidence[j] > 1.f) continue;
            const int b = bin(confidence[j]);
            if (seen_[j][b] >= 32.) { seen_[j][b] *= .5; yes_[j][b] *= .5; }
            seen_[j][b] += 1.;
            if (j < accepted) yes_[j][b] += 1.;
        }
        ++observed;
    }

    void observe_draft(int n, double ms) {
        if (n > 0 && std::isfinite(ms) && ms > 0. && ms < 10.) {
            const double sample = ms / n;
            draft_ms_ += .125 * (sample - draft_ms_);
        }
    }

    double verify_cost(int t) const {
        if (t < 1 || t > 8) return 1e9;
        if (cost_seen_[t] >= 5) return cost_[t];
        // Prior shape from the measured native CPU+GPU0 cohort, rescaled online.
        int reference = 0;
        for (int k = 1; k <= 8; ++k)
            if (cost_seen_[k] >= 5 && cost_seen_[k] > cost_seen_[reference]) reference = k;
        const double prior = 7.5 + 2.4 * t;
        return reference ? prior * cost_[reference] / (7.5 + 2.4 * reference) : prior;
    }

    bool extend(int drafted, const float* confidence) {
        if (drafted < 3) return true;  // Preserve the stock first-three-proposal policy.
        if (drafted >= kDepth || !confidence) return false;
        double survival = 1., expected = 1.;
        for (int j = 0; j < drafted; ++j) {
            survival *= probability(j, confidence[j]);
            expected += survival;
        }
        // The next confidence is unknown until that step runs. Use the last observed
        // confidence with the next depth's independently learned acceptance statistics.
        const double next_survival = survival * probability(drafted, confidence[drafted - 1]);
        const double current_cost = verify_cost(1 + drafted) + drafted * draft_ms_;
        const double next_cost = verify_cost(2 + drafted) + (drafted + 1) * draft_ms_;
        const bool good = (expected + next_survival) / next_cost >= 1.01 * expected / current_cost;
        ++decisions;
        if (good) ++extensions; else ++stops;
        return good;
    }

    double seen(int depth) const {
        double n = 0.;
        for (double x : seen_[depth]) n += x;
        return n;
    }
    double draft_ms() const { return draft_ms_; }
    std::uint64_t observed = 0, decisions = 0, extensions = 0, stops = 0;

private:
    static int bin(double x) { return std::clamp(static_cast<int>(x * kBins), 0, kBins - 1); }
    std::array<std::array<double, kBins>, kDepth> seen_{}, yes_{};
    std::array<double, 9> cost_{};
    std::array<std::uint32_t, 9> cost_seen_{};
    double draft_ms_ = .55;
};
}  // namespace swift78
