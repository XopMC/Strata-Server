#pragma once
#include "tail_api.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace swift_tail_r027 {
// One guard may cover MANY same-stream groups; it need not synchronize pergroup.
// Declare AFTER all borrowed allocation/staging owners, so it drains BEFORE them.
// It owns completion, not allocations. Host upload source credits are separate.
class SubmissionGuard {
    int owner_;
    cudaStream_t stream_;
    uint64_t epoch_;
    bool pending_ = false;
    void drain_or_exit() noexcept {
        if (!pending_) return;
        int previous = -1;
        cudaError_t e = cudaGetDevice(&previous);
        if (e == cudaSuccess) e = cudaSetDevice(owner_);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream_);
        if (e == cudaSuccess && previous != owner_) e = cudaSetDevice(previous);
        if (e != cudaSuccess) {
            std::fprintf(stderr,"R027 owner%d epoch%llu cannot prove stream completion: %s\n",
                         owner_,(unsigned long long)epoch_,cudaGetErrorString(e));
            std::fflush(stderr);
            std::_Exit(2); // Do not unwind borrowed storage under uncertain GPU writes.
        }
        pending_ = false;
    }
public:
    SubmissionGuard(int owner, cudaStream_t stream, uint64_t epoch)
        : owner_(owner), stream_(stream), epoch_(epoch) {
        if (owner < 0 || owner >= 3 || !stream || !epoch)
            throw std::runtime_error("R027 invalid explicit stream/owner/epoch");
    }
    SubmissionGuard(const SubmissionGuard&) = delete;
    SubmissionGuard& operator=(const SubmissionGuard&) = delete;
    ~SubmissionGuard() { drain_or_exit(); }
    // Call BEFORE the first async upload/producer launch, including uncertain submit errors.
    void before_enqueue(const Binding& b) {
        int current = -1;
        if (b.owner != owner_ || b.stream != stream_ || b.epoch != epoch_ ||
            cudaGetDevice(&current) != cudaSuccess || current != owner_)
            throw std::runtime_error("R027 issuing device/stream/epoch mismatch before enqueue");
        pending_ = true;
    }
    void finish() noexcept { drain_or_exit(); }
};
} // namespace swift_tail_r027
