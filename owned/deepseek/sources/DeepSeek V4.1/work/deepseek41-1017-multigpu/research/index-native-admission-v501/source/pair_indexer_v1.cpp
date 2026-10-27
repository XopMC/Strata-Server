#include "index_pair_admission_v501.hpp"
#include "../../pair-indexer-engine-v1-r2/pair_indexer_v1.hpp"
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace strata::ds41::pair_indexer_v1 {
namespace {
namespace k14 = strata::ds41::kernels;
using bf16 = __nv_bfloat16;
constexpr size_t MiB = size_t(1024) * 1024;
constexpr int Columns = 128, Heads = 32, Top = 512;
std::array<std::atomic<uint64_t>, 20> counters{};
std::atomic<uint64_t> next_epoch{0};
void need(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(std::string("pair indexer v1: ") + message);
}
void ck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("pair indexer v1: ") + cudaGetErrorString(error));
}
void maximum(int field, uint64_t value) {
    uint64_t previous = counters[field].load();
    while (previous < value && !counters[field].compare_exchange_weak(previous, value)) {}
}
struct Device {
    int previous = 0;
    explicit Device(int current) { ck(cudaGetDevice(&previous)); ck(cudaSetDevice(current)); }
    ~Device() noexcept { (void) cudaSetDevice(previous); }
};
int source_slot(int layer) { return layer == 2 ? 0 : layer == 8 ? 1 : layer == 14 ? 2 : -1; }
int split_cut(int rows) { return (rows / 512) * 256; }
bool flag(const char* name) { const char* value = std::getenv(name); return value && !std::strcmp(value, "1"); }
}  // namespace

struct Owner::Impl {
    struct Source {
        const bf16* pointer = nullptr;
        bf16* mirror = nullptr;
        uint64_t epoch = 0;
        int produced = 0, mirrored = 0;
    };
    std::array<Source, 3> sources{};
    uint64_t epoch = 0;
    int max_sequence = 0, batch = 0, key_capacity = 0, helper_capacity = 0;
    cudaStream_t producer{}, copy0{}, compute1{}, transfer1{};
    cudaEvent_t source_ready{}, forward_ready{}, helper_done{}, joined{};
    bf16 *q1 = nullptr, *w1 = nullptr;
    int32_t* top1 = nullptr;
    void* workspace1 = nullptr;
    size_t workspace_bytes = 0, helper_bytes = 0;
    void *host_q = nullptr, *host_w = nullptr, *host_top = nullptr, *host_keys = nullptr;
    bool resources_started = false, initialized = false, poisoned = false, finished = false;

    // Always attempt ALL four streams, even after an earlier device or stream
    // error. producer is borrowed; its lifetime ends only after our drain.
    bool drain_all_noexcept() noexcept {
        if (!resources_started) return true;
        int previous = 0;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        bool ok = have_previous;
        if (cudaSetDevice(0) == cudaSuccess) {
            if (cudaStreamSynchronize(producer) != cudaSuccess) ok = false;
            if (copy0 && cudaStreamSynchronize(copy0) != cudaSuccess) ok = false;
        } else ok = false;
        if (cudaSetDevice(1) == cudaSuccess) {
            if (compute1 && cudaStreamSynchronize(compute1) != cudaSuccess) ok = false;
            if (transfer1 && cudaStreamSynchronize(transfer1) != cudaSuccess) ok = false;
        } else ok = false;
        if (have_previous && cudaSetDevice(previous) != cudaSuccess) ok = false;
        if (ok) ++counters[15];
        return ok;
    }

    ~Impl() noexcept {
        if (!resources_started) return;
        // A failed drain cannot release buffers still borrowed by DMA/kernels.
        if (!drain_all_noexcept()) {
            std::fputs("PAIR_INDEXER_V1_FATAL all-stream drain failed\n", stderr);
            std::terminate();
        }
        int previous = 0;
        (void) cudaGetDevice(&previous);
        if (cudaSetDevice(1) == cudaSuccess) {
            for (auto& source : sources) if (source.mirror) (void) cudaFree(source.mirror);
            if (q1) (void) cudaFree(q1);
            if (w1) (void) cudaFree(w1);
            if (top1) (void) cudaFree(top1);
            if (workspace1) (void) cudaFree(workspace1);
            if (helper_done) (void) cudaEventDestroy(helper_done);
            if (compute1) (void) cudaStreamDestroy(compute1);
            if (transfer1) (void) cudaStreamDestroy(transfer1);
        }
        if (cudaSetDevice(0) == cudaSuccess) {
            if (source_ready) (void) cudaEventDestroy(source_ready);
            if (forward_ready) (void) cudaEventDestroy(forward_ready);
            if (joined) (void) cudaEventDestroy(joined);
            if (copy0) (void) cudaStreamDestroy(copy0);
        }
        for (void* pointer : {host_q, host_w, host_top, host_keys}) if (pointer) (void) cudaFreeHost(pointer);
        (void) cudaSetDevice(previous);
        if (initialized) --counters[17];
    }

    void initialize(int sequence, int rows, cudaStream_t stream) {
        max_sequence = sequence; batch = rows; producer = stream;
        key_capacity = max_sequence / 2 + 1;
        for (int count = 512; count <= batch; ++count)
            helper_capacity = std::max(helper_capacity, count - split_cut(count));
        ds501::require_environment();
        need(flag("DS41_INDEXER_S8_R1"), "selected S8 environment required");
        int count = 0; ck(cudaGetDeviceCount(&count)); need(count == 2, "exactly two visible devices required");
        for (int device = 0; device < 2; ++device) {
            Device scope(device);
            ds501::validate_device(device);int peer=0;
            ck(cudaDeviceCanAccessPeer(&peer, device, 1 - device)); need(!peer, "qualified noP2P topology required");
        }
        { Device scope(0); cudaStreamCaptureStatus status{}; ck(cudaStreamIsCapturing(producer, &status));
          need(status == cudaStreamCaptureStatusNone, "ordinary uncaptured prefill required"); }
        workspace_bytes = k14::indexer_topk_prefill_workspace_bytes(helper_capacity, max_sequence / 2);
        const size_t key_bytes = size_t(key_capacity) * Columns * sizeof(bf16);
        const size_t q_bytes = size_t(helper_capacity) * Heads * Columns * sizeof(bf16);
        const size_t w_bytes = size_t(helper_capacity) * Heads * sizeof(bf16);
        const size_t top_bytes = size_t(helper_capacity) * Top * sizeof(int32_t);
        helper_bytes = 3 * key_bytes + q_bytes + w_bytes + top_bytes + workspace_bytes;
        need(helper_bytes < 512 * MiB, "three mirrors plus all helper scratch exceed 512 MiB");
        resources_started = true;
        { Device scope(1); size_t free = 0, total = 0; ck(cudaMemGetInfo(&free, &total));
          need(free >= helper_bytes + 64 * MiB, "insufficient helper free memory before allocation");
          ck(cudaStreamCreateWithFlags(&compute1, cudaStreamNonBlocking));
          ck(cudaStreamCreateWithFlags(&transfer1, cudaStreamNonBlocking));
          ck(cudaEventCreateWithFlags(&helper_done, cudaEventDisableTiming));
          for (auto& source : sources) ck(cudaMalloc(reinterpret_cast<void**>(&source.mirror), key_bytes));
          ck(cudaMalloc(reinterpret_cast<void**>(&q1), q_bytes)); ck(cudaMalloc(reinterpret_cast<void**>(&w1), w_bytes));
          ck(cudaMalloc(reinterpret_cast<void**>(&top1), top_bytes)); ck(cudaMalloc(&workspace1, workspace_bytes)); }
        { Device scope(0);
          ck(cudaStreamCreateWithFlags(&copy0, cudaStreamNonBlocking));
          ck(cudaEventCreateWithFlags(&source_ready, cudaEventDisableTiming));
          ck(cudaEventCreateWithFlags(&forward_ready, cudaEventDisableTiming));
          ck(cudaEventCreateWithFlags(&joined, cudaEventDisableTiming));
          ck(cudaHostAlloc(&host_q, q_bytes, cudaHostAllocPortable)); ck(cudaHostAlloc(&host_w, w_bytes, cudaHostAllocPortable));
          ck(cudaHostAlloc(&host_top, top_bytes, cudaHostAllocPortable)); ck(cudaHostAlloc(&host_keys, key_bytes, cudaHostAllocPortable)); }
        epoch = ++next_epoch;
        initialized = true;
        ++counters[2]; ++counters[14]; ++counters[17]; counters[19] = epoch;
        maximum(18, helper_bytes);
    }

    void publish(int layer, const bf16* pointer, int first, int count) {
        const int slot = source_slot(layer);
        if (slot < 0) return;
        need(!poisoned && !finished, "publication on retired owner");
        auto& source = sources[slot];
        need(pointer && first >= 0 && count >= 0 && first <= key_capacity - count,
             "key publication bounds");
        need(first == source.produced, "noncontiguous key publication");
        if (source.pointer) need(source.pointer == pointer && source.epoch == epoch, "source pointer/epoch changed");
        else { source.pointer = pointer; source.epoch = epoch; }
        source.produced += count;
        ++counters[5]; maximum(13, source.produced);
    }

    bool run(int layer, int ratio, const bf16* q, const bf16* pointer, const bf16* w,
             int rows, int position, int32_t* output, void* local_workspace, size_t local_bytes) {
        const int slot = source_slot(layer), cut = split_cut(rows);
        // FULL shape admission is decided before events, DMA or either K14.
        if (slot < 0 || ratio != 2 || rows < 512 || rows > batch || cut < 256 || rows - cut < 256) {
            ++counters[7]; return false;
        }
        need(!poisoned && !finished, "run on retired owner");
        need(position >= 0 && position <= max_sequence - rows && q && w && pointer && output && local_workspace,
             "full call bounds/pointers");
        const int keys = (position + rows) / 2, local_keys = (position + cut) / 2;
        const size_t original_required = k14::indexer_topk_prefill_workspace_bytes(rows, keys);
        const size_t local_required = k14::indexer_topk_prefill_workspace_bytes(cut, local_keys);
        const size_t helper_required = k14::indexer_topk_prefill_workspace_bytes(rows - cut, keys);
        need(local_bytes >= original_required && local_bytes >= local_required && workspace_bytes >= helper_required,
             "full/local/helper public workspace admission");
        auto& source = sources[slot];
        need(source.pointer == pointer && source.epoch == epoch && source.produced == keys &&
             source.mirrored >= 0 && source.mirrored <= source.produced, "missing/stale key publication");
        const int append_first = source.mirrored, append_rows = source.produced - source.mirrored;
        const size_t append_bytes = size_t(append_rows) * Columns * sizeof(bf16);
        const size_t q_bytes = size_t(rows - cut) * Heads * Columns * sizeof(bf16);
        const size_t w_bytes = size_t(rows - cut) * Heads * sizeof(bf16);
        const size_t top_bytes = size_t(rows - cut) * Top * sizeof(int32_t);
        try {
            { Device scope(0);
              // The original D2D key publication AND original Q/W projections
              // precede this event. Never wait for all local K14 compute here.
              ck(cudaEventRecord(source_ready, producer)); ck(cudaStreamWaitEvent(copy0, source_ready, 0));
              if (append_bytes) ck(cudaMemcpyAsync(host_keys, pointer + size_t(append_first) * Columns,
                                                   append_bytes, cudaMemcpyDeviceToHost, copy0));
              ck(cudaMemcpyAsync(host_q, q + size_t(cut) * Heads * Columns, q_bytes, cudaMemcpyDeviceToHost, copy0));
              ck(cudaMemcpyAsync(host_w, w + size_t(cut) * Heads, w_bytes, cudaMemcpyDeviceToHost, copy0));
              ck(cudaEventRecord(forward_ready, copy0));
              k14::indexer_topk_prefill(q, pointer, w, cut, position, 2, nullptr, nullptr, 0, Top, 0, 2048, 8,
                                        output, local_workspace, local_required, producer);
              ck(cudaEventSynchronize(forward_ready)); }
            { Device scope(1);
              if (append_bytes) ck(cudaMemcpyAsync(source.mirror + size_t(append_first) * Columns, host_keys,
                                                   append_bytes, cudaMemcpyHostToDevice, compute1));
              ck(cudaMemcpyAsync(q1, host_q, q_bytes, cudaMemcpyHostToDevice, compute1));
              ck(cudaMemcpyAsync(w1, host_w, w_bytes, cudaMemcpyHostToDevice, compute1));
              // BASE ZERO keys; causal row identity is position + cut + i.
              k14::indexer_topk_prefill(q1, source.mirror, w1, rows - cut, position + cut, 2,
                                        nullptr, nullptr, 0, Top, 0, 2048, 8, top1, workspace1, helper_required, compute1);
              ck(cudaMemcpyAsync(host_top, top1, top_bytes, cudaMemcpyDeviceToHost, compute1));
              ck(cudaEventRecord(helper_done, compute1)); }
            { Device scope(0);
              ck(cudaStreamWaitEvent(producer, helper_done, 0));
              ck(cudaMemcpyAsync(output + size_t(cut) * Top, host_top, top_bytes, cudaMemcpyHostToDevice, producer));
              ck(cudaEventRecord(joined, producer)); ck(cudaEventSynchronize(joined)); }
            // All earlier event consumers/input readers/result writers joined;
            // these exact event slots can now be rerecorded for the next call.
            source.mirrored = source.produced;
            ++counters[6]; counters[8] += cut; counters[9] += rows - cut;
            counters[10] += append_bytes; counters[11] += q_bytes + w_bytes; counters[12] += top_bytes;
            return true;
        } catch (...) {
            poisoned = true; ++counters[16];
            const bool drained = drain_all_noexcept();
            if (!drained) throw std::runtime_error("pair indexer v1: submit failed; all-stream drain also failed");
            throw; // Never replay the original full math after partial submission.
        }
    }
};

Owner::Owner(int max_sequence, int batch_capacity, int initial_position, cudaStream_t producer) {
    ++counters[1];
    if (!flag("DS1017_PAIR_INDEXER_PREFILL_V1")) return;
    if (initial_position != 0) { ++counters[3]; return; }
    if ((max_sequence != 262144 && max_sequence != 524288) || batch_capacity < 512 || batch_capacity > 2048) {
        ++counters[4]; return;
    }
    impl_ = std::make_unique<Impl>();
    impl_->initialize(max_sequence, batch_capacity, producer);
}
Owner::~Owner() noexcept = default;
void Owner::publish(int layer, const bf16* source, int first, int count) {
    if (impl_) impl_->publish(layer, source, first, count);
}
bool Owner::run(int layer, int ratio, const bf16* q, const bf16* source, const bf16* w,
                int rows, int position, int32_t* output, void* workspace, size_t bytes) {
    return impl_ && impl_->run(layer, ratio, q, source, w, rows, position, output, workspace, bytes);
}
void Owner::finish() {
    if (!impl_ || impl_->finished) return;
    const bool drained = impl_->drain_all_noexcept();
    if (!drained) { impl_->poisoned = true; ++counters[16]; }
    need(drained, "finish all-stream drain failed");
    impl_->finished = true;
}
}  // namespace strata::ds41::pair_indexer_v1

extern "C" int ds1017_pair_indexer_info_v1(uint64_t* output, int n) {
    if (!output || n != 20) return -1;
    output[0] = 1;
    for (int i = 1; i < n; ++i) output[i] = strata::ds41::pair_indexer_v1::counters[i].load();
    return 0;
}
