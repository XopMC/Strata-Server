#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cstring>
#include <fstream>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds41 {

// Resolve the caller's current CUDA ordinal at pool creation. No device or
// NUMA identity is retained across launches, and a missing topology refuses
// the accelerated copy pool rather than guessing a CPU node.
inline int stage_copy_current_numa_node_v2(int expected_device, const char* error) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
        (expected_device >= 0 && device != expected_device))
        throw std::runtime_error(error);
    char pci[32] = {};
    if (cudaDeviceGetPCIBusId(pci, sizeof(pci), device) != cudaSuccess)
        throw std::runtime_error(error);
    unsigned domain = 0, bus = 0, slot = 0, function = 0;
    char trailing = 0;
    if (std::sscanf(pci, "%x:%x:%x.%x%c", &domain, &bus, &slot, &function, &trailing) != 4 ||
        bus > 255 || slot > 31 || function > 7)
        throw std::runtime_error(error);
    char canonical[32] = {};
    const int length = std::snprintf(canonical, sizeof(canonical), "%04x:%02x:%02x.%x",
                                     domain, bus, slot, function);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(canonical))
        throw std::runtime_error(error);
    int node = -1;
    std::string extra;
    std::ifstream in(std::string("/sys/bus/pci/devices/") + canonical + "/numa_node");
    if (!(in >> node) || node < 0 || (in >> extra))
        throw std::runtime_error(error);
    return node;
}

// Fixed, reusable host staging copy pool. It only copies the existing packed
// projection bytes; it does not change their order, contents or CUDA stream.
class NativeStageCopyPoolV2 {
public:
    explicit NativeStageCopyPoolV2(int workers) : NativeStageCopyPoolV2(workers,{}) {}
    NativeStageCopyPoolV2(int workers,std::vector<int> explicit_cpus) : count_(workers), affinity_(workers, -1) {
        if (count_ < 1 || count_ > 8) throw std::invalid_argument("stage copy worker count");
        cpus_ = explicit_cpus.empty()?local_cpus(count_):std::move(explicit_cpus);
        {auto sorted=cpus_;std::sort(sorted.begin(),sorted.end());if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()||std::any_of(sorted.begin(),sorted.end(),[](int cpu){return cpu<0||cpu>=CPU_SETSIZE;}))throw std::invalid_argument("325 explicit copy CPU list");}
        if (static_cast<int>(cpus_.size()) != count_)
            throw std::runtime_error("insufficient CPU0-local stage-copy workers");
        threads_.reserve(static_cast<size_t>(count_));
        try {for (int i = 0; i < count_; ++i) threads_.emplace_back([this, i] { worker(i); });}
        catch (...) {stop_and_join();throw;}
        std::unique_lock<std::mutex> lock(mutex_);
        ready_cv_.wait(lock, [&] { return ready_ == count_; });
        for (int rc : affinity_) if (rc != 0) {
            lock.unlock();
            stop_and_join();
            throw std::runtime_error("failed to pin a stage-copy worker to GPU0 NUMA node");
        }
    }

    ~NativeStageCopyPoolV2() { stop_and_join(); }
    NativeStageCopyPoolV2(const NativeStageCopyPoolV2&) = delete;
    NativeStageCopyPoolV2& operator=(const NativeStageCopyPoolV2&) = delete;

    void copy(void* destination, const void* source, size_t bytes) {
        if (!destination || !source) throw std::invalid_argument("null stage-copy address");
        if (bytes < (4u << 20)) {
            std::memcpy(destination, source, bytes);
            ++calls_;
            bytes_ += bytes;
            return;
        }
        const int active = std::min<int>(count_, static_cast<int>((bytes + (1u << 20) - 1) >> 20));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            source_ = static_cast<const uint8_t*>(source);
            destination_ = static_cast<uint8_t*>(destination);
            bytes_work_ = bytes;
            active_ = active;
            pending_ = active;
            ++generation_;
        }
        start_cv_.notify_all();
        const size_t first_end = bytes / static_cast<size_t>(active + 1);
        std::memcpy(destination_, source_, first_end);
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return pending_ == 0; });
        ++calls_;
        ++parallel_calls_;
        bytes_ += bytes;
    }

    uint64_t calls() const { return calls_; }
    uint64_t parallel_calls() const { return parallel_calls_; }
    uint64_t bytes() const { return bytes_; }

private:
    static std::vector<int> local_cpus(int count) {
        const int node = stage_copy_current_numa_node_v2(-1, "GPU NUMA node unavailable for stage-copy pool");
        const int main_cpu = sched_getcpu();
        std::vector<int> candidates;
        if (node >= 0) {
            std::ifstream in("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
            std::string cpulist;
            if (!in || !(in >> cpulist)) throw std::runtime_error("GPU0 NUMA cpulist unavailable");
            size_t at = 0;
            while (at < cpulist.size()) {
                size_t end = cpulist.find(',', at);
                if (end == std::string::npos) end = cpulist.size();
                size_t dash = cpulist.find('-', at);
                int first = std::stoi(cpulist.substr(at, dash < end ? dash - at : end - at));
                int last = dash < end ? std::stoi(cpulist.substr(dash + 1, end - dash - 1)) : first;
                // The R6 owner cgroup explicitly admits physical CPU IDs0–255.
                // The model runtime pins the calling thread to CPU255, so its
                // per-thread affinity mask cannot be used as the pool's set.
                for (int cpu = first; cpu <= last && cpu < 256; ++cpu)
                    if (cpu != main_cpu) candidates.push_back(cpu);
                at = end + 1;
            }
        } else throw std::runtime_error("GPU0 NUMA node unavailable for stage-copy pool");
        if (static_cast<int>(candidates.size()) < count)
            throw std::runtime_error("GPU0-local CPU set too small for stage-copy pool");
        candidates.resize(static_cast<size_t>(count));
        return candidates;
    }

    void worker(int id) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus_[id], &set);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            affinity_[id] = rc;
            ++ready_;
        }
        ready_cv_.notify_one();
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            start_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            const int active = active_;
            const uint8_t* src = source_;
            uint8_t* dst = destination_;
            const size_t bytes = bytes_work_;
            if (id >= active) continue;
            lock.unlock();
            const size_t stripes = static_cast<size_t>(active + 1);
            const size_t begin = bytes * static_cast<size_t>(id + 1) / stripes;
            const size_t end = bytes * static_cast<size_t>(id + 2) / stripes;
            std::memcpy(dst + begin, src + begin, end - begin);
            lock.lock();
            if (--pending_ == 0) done_cv_.notify_one();
        }
    }

    void stop_and_join() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& thread : threads_) if (thread.joinable()) thread.join();
    }

    int count_;
    std::vector<int> cpus_, affinity_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_cv_, done_cv_, ready_cv_;
    bool stopping_ = false;
    int ready_ = 0, active_ = 0, pending_ = 0;
    uint64_t generation_ = 0, calls_ = 0, parallel_calls_ = 0, bytes_ = 0;
    const uint8_t* source_ = nullptr;
    uint8_t* destination_ = nullptr;
    size_t bytes_work_ = 0;
};

} // namespace strata::ds41
