// src/ds41/pack.cpp - read the ds41 pack written by tools/ds41/pack.py.
#include "strata/ds41/pack_stream_dense_v190.hpp"

#include "strata/ds41/config.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace strata::ds41 {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("ds41 pack: " + what); }

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

DType parse_dtype(const std::string& s) {
    if (s == "f8e4m3") return DType::F8E4M3;
    if (s == "e8m0") return DType::E8M0;
    if (s == "bf16") return DType::BF16;
    if (s == "f32") return DType::F32;
    if (s == "f16") return DType::F16;
    if (s == "i32") return DType::I32;
    if (s == "i16") return DType::I16;
    if (s == "i8") return DType::I8;
    if (s == "u8") return DType::U8;
    fail("unknown dtype " + s);
}

std::ifstream open_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);
    return f;
}

bool contains(uint64_t limit, uint64_t offset, uint64_t bytes) {
    return offset <= limit && bytes <= limit - offset;
}

uint64_t component_number(const std::string& text) {
    if (text.empty() || text[0] == '-') fail("bad component size");
    size_t used = 0;
    const auto n = std::stoull(text, &used);
    if (used != text.size()) fail("bad component size");
    return n;
}

bool skip(const std::string& line) { return line.empty() || line[0] == '#'; }

const char* kComp[12] = {"w1.trellis", "w1.suh", "w1.svh", "w1.mul1", "w3.trellis", "w3.suh",
                         "w3.svh",     "w3.mul1", "w2.trellis", "w2.suh", "w2.svh", "w2.mul1"};

}  // namespace

size_t dtype_size(DType t) {
    switch (t) {
        case DType::F8E4M3: case DType::E8M0: case DType::I8: case DType::U8: return 1;
        case DType::BF16: case DType::F16: case DType::I16: return 2;
        case DType::F32: case DType::I32: return 4;
    }
    return 0;
}


// V190 placement experiment. Dense bytes stay borrowed from the admitted RAM
// owner; no arithmetic, weight format, scales or quantizer is changed here.
struct DenseStreamStateV190 {
    static constexpr size_t kCopyChunk = size_t(64) << 20;
    static constexpr size_t kBankLimit = size_t(344) << 20;
    struct Item { DenseTensor* tensor; size_t full_offset; size_t bank_offset; };
    struct Bank {
        uint8_t* host = nullptr;
        uint8_t* device = nullptr;
        cudaEvent_t copied = nullptr, used = nullptr;
        int layer = -1;
        bool submitted = false, activated = false, retired = true;
    };
    std::shared_ptr<const BorrowedWeights> weights;
    std::vector<Item> base_items, all_layer_items;
    std::array<std::vector<Item>, kLayers> layer_items;
    std::array<size_t, kLayers> layer_span{}, layer_payload{};
    std::array<Bank, 2> banks{};
    uint8_t* base = nullptr;
    uint8_t* layers = nullptr;
    uint8_t* mapped_embedding_host_r1 = nullptr;
    DenseTensor* mapped_embedding_tensor_r1 = nullptr;
    cudaStream_t upload = nullptr;
    size_t base_bytes = 0, layer_bytes = 0, bank_capacity = 0;
    bool active = false, failed = false;
    bool headless320=false;uint64_t omitted_head_bytes320=0;
    uint64_t begins = 0, restores = 0, reclaimed = 0;
    uint64_t layer_upload_bytes = 0, layer_uploads = 0;
    uint64_t prefetches = 0, activations = 0, retires = 0;
    uint64_t restore_bytes = 0, restore_tensor_copies = 0;
    uint64_t free_before_begin = 0, free_after_begin = 0, free_after_restore = 0;
    size_t scratch_peak = 0;

    static size_t align256(size_t value) {
        if (value > std::numeric_limits<size_t>::max() - 255) fail("v190 offset overflow");
        return (value + 255) & ~size_t(255);
    }
    static int ordinary_layer(const std::string& name) {
        if (name.compare(0, 7, "layers.") != 0 || name.find(".engram.") != std::string::npos) return -1;
        const size_t dot = name.find('.', 7);
        if (dot == std::string::npos || dot == 7) fail("v190 malformed layer name");
        int layer = 0;
        for (size_t i = 7; i < dot; ++i) {
            if (name[i] < '0' || name[i] > '9') fail("v190 malformed layer number");
            layer = layer * 10 + (name[i] - '0');
            if (layer >= kLayers) fail("v190 layer number outside backbone");
        }
        return layer;
    }
    DenseStreamStateV190(std::shared_ptr<const BorrowedWeights> w, std::map<std::string, DenseTensor>& tensors,bool headless=false)
        : weights(std::move(w)),headless320(headless) {
        const char* mapped=std::getenv("DS41_MAPPED_EMBEDDING_R1");
        const bool mapped_enabled=mapped&&std::strcmp(mapped,"1")==0;
        for (auto& [name, tensor] : tensors) {
            if(headless320&&name=="head.weight"){
                if(tensor.dtype!=DType::BF16||tensor.shape!=std::vector<int64_t>{kVocab,kDim}||tensor.bytes!=size_t(kVocab)*kDim*2)
                    fail("320 requires exact BF16 vocabulary head before omission");
                omitted_head_bytes320=tensor.bytes;tensor.device=nullptr;continue;
            }
            if(mapped_enabled&&name=="embed.weight"){
                if(tensor.dtype!=DType::BF16||tensor.shape!=std::vector<int64_t>{kVocab,kDim}||tensor.bytes!=size_t(kVocab)*kDim*2)
                    fail("Mapped embedding requires the exact native BF16 source tensor");
                mapped_embedding_tensor_r1=&tensor;
                continue;
            }
            const int layer = ordinary_layer(name);
            if (layer < 0) {
                const size_t offset = align256(base_bytes);
                if (tensor.bytes > (size_t(24) << 30) - offset) fail("v190 base layout exceeds24GiB");
                base_items.push_back({&tensor, offset, offset});
                base_bytes = offset + size_t(tensor.bytes);
            } else {
                const size_t full = align256(layer_bytes);
                const size_t local = align256(layer_span[layer]);
                if (tensor.bytes > (size_t(24) << 30) - full || tensor.bytes > kBankLimit - std::min(local, kBankLimit))
                    fail("v190 layer layout exceeds bounded bank");
                Item item{&tensor, full, local};
                all_layer_items.push_back(item);
                layer_items[layer].push_back(item);
                layer_bytes = full + size_t(tensor.bytes);
                layer_span[layer] = local + size_t(tensor.bytes);
                layer_payload[layer] += size_t(tensor.bytes);
            }
        }
        if(headless320&&(!mapped_enabled||!mapped_embedding_tensor_r1||omitted_head_bytes320!=size_t(kVocab)*kDim*2))
            fail("320 requires explicit mapped embedding and exact omitted head");
        base_bytes = align256(base_bytes);
        layer_bytes = align256(layer_bytes);
        for (int layer = 0; layer < kLayers; ++layer) {
            if (layer_items[layer].empty()) fail("v190 missing admitted layer dense weights");
            layer_span[layer] = align256(layer_span[layer]);
            bank_capacity = std::max(bank_capacity, layer_span[layer]);
        }
        if (!base_bytes || !layer_bytes || bank_capacity > kBankLimit || layer_bytes <= 2 * bank_capacity)
            fail("v190 streamed dense layout has no bounded VRAM saving");
    }
    void upload_mapped_embedding_r1(){
        if(!mapped_embedding_tensor_r1)return;
        if(mapped_embedding_host_r1)fail("Mapped embedding already allocated");
        auto&t=*mapped_embedding_tensor_r1;
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&mapped_embedding_host_r1),size_t(t.bytes),headless320?(cudaHostAllocMapped|cudaHostAllocPortable):cudaHostAllocMapped),"mapped BF16 embedding allocation");
        void*device=nullptr;
        check_cuda(cudaHostGetDevicePointer(&device,mapped_embedding_host_r1,0),"mapped embedding GPU pointer");
        std::vector<uint8_t>scratch;
        for(size_t offset=0;offset<t.bytes;offset+=kCopyChunk){
            const size_t bytes=std::min(kCopyChunk,size_t(t.bytes)-offset);
            const auto*source=weights->read_range(t.name,offset,bytes,scratch);
            std::memcpy(mapped_embedding_host_r1+offset,source,bytes);
            if(std::memcmp(mapped_embedding_host_r1+offset,source,bytes))fail("Mapped embedding BF16 byte copy differs");
        }
        t.device=device;
        if(headless320)std::fprintf(stderr,"headless320: %zu unchanged BF16 embedding bytes in portable pinned host RAM; vocabulary head omitted\n",size_t(t.bytes));
        else std::fprintf(stderr,"mapped embedding R1: %zu unchanged BF16 bytes in pinned host RAM; head stays in VRAM\n",size_t(t.bytes));
    }
    void usable() const {
        if (failed) fail("v190 dense placement failed; destroy this engine");
    }
    void clear_layer_views(int layer) noexcept {
        if (layer < 0) return;
        for (auto& item : layer_items[layer]) item.tensor->device = nullptr;
    }
    void copy_group(const std::vector<Item>& items, uint8_t* target, bool restoring) {
        if (!target) fail("v190 missing dense allocation");
        std::vector<uint8_t> scratch;
        for (const auto& item : items) {
            auto& tensor = *item.tensor;
            for (size_t offset = 0; offset < tensor.bytes; offset += kCopyChunk) {
                const size_t bytes = std::min(kCopyChunk, size_t(tensor.bytes) - offset);
                const auto* source = weights->read_range(tensor.name, offset, bytes, scratch);
                scratch_peak = std::max(scratch_peak, scratch.size());
                check_cuda(cudaMemcpy(target + item.full_offset + offset, source, bytes, cudaMemcpyHostToDevice),
                           "v190 restore borrowed dense bytes");
                if (restoring) restore_bytes += bytes;
            }
            tensor.device = target + item.full_offset;
            if (restoring) ++restore_tensor_copies;
        }
    }
    void release_banks() {
        if (upload) check_cuda(cudaStreamSynchronize(upload), "v190 retire upload stream");
        check_cuda(cudaStreamSynchronize(nullptr), "v190 retire dense consumers");
        for (auto& bank : banks) {
            if (bank.device) { check_cuda(cudaFree(bank.device), "v190 free layer bank"); bank.device = nullptr; }
            if (bank.host) { check_cuda(cudaFreeHost(bank.host), "v190 free pinned layer bank"); bank.host = nullptr; }
            if (bank.copied) { check_cuda(cudaEventDestroy(bank.copied), "v190 destroy copied event"); bank.copied = nullptr; }
            if (bank.used) { check_cuda(cudaEventDestroy(bank.used), "v190 destroy used event"); bank.used = nullptr; }
            bank.layer = -1; bank.submitted = false; bank.activated = false; bank.retired = true;
        }
        if (upload) { check_cuda(cudaStreamDestroy(upload), "v190 destroy upload stream"); upload = nullptr; }
    }
    ~DenseStreamStateV190() {
        // A failed placement is not restored: Engine has been poisoned. Only
        // synchronize and free our own allocations; never upload in destruction.
        if (upload) cudaStreamSynchronize(upload);
        cudaStreamSynchronize(nullptr);
        for (auto& bank : banks) {
            if (bank.device) cudaFree(bank.device);
            if (bank.host) cudaFreeHost(bank.host);
            if (bank.copied) cudaEventDestroy(bank.copied);
            if (bank.used) cudaEventDestroy(bank.used);
        }
        if (upload) cudaStreamDestroy(upload);
        if (layers) cudaFree(layers);
        if (base) cudaFree(base);
        if(mapped_embedding_host_r1)cudaFreeHost(mapped_embedding_host_r1);
    }
};

void Pack::configure_dense_stream_v190(bool enabled) {
    if (arena_ || (dense_stream_v190_ && dense_stream_v190_->base))
        fail("v190 configuration must precede first dense upload");
    if (!enabled) { dense_stream_v190_.reset(); return; }
    if (!native_) fail("v190 dense streaming requires admitted borrowed native checkpoint");
    if (dense_stream_v190_) fail("v190 dense streaming already configured");
    dense_stream_v190_ = std::make_unique<DenseStreamStateV190>(native_->weights(), dense_);
}
void Pack::configure_headless_dense_v320(){
    if(arena_||dense_stream_v190_||!native_)fail("320 headless configuration requires fresh native Pack");
    dense_stream_v190_=std::make_unique<DenseStreamStateV190>(native_->weights(),dense_,true);
}
std::array<uint64_t,8> Pack::headless_dense_info_v320()const noexcept{
    if(!dense_stream_v190_)return {320,0,0,0,0,0,0,0};
    const auto&s=*dense_stream_v190_;const auto legacy=dense_stream_info_v190();
    return {320,uint64_t(s.headless320),s.omitted_head_bytes320,uint64_t(s.base_bytes),uint64_t(s.layer_bytes),uint64_t(s.bank_capacity),s.mapped_embedding_host_r1?uint64_t(s.mapped_embedding_tensor_r1->bytes):0,legacy[10]};
}
bool Pack::dense_stream_enabled_v190() const noexcept { return bool(dense_stream_v190_); }
bool Pack::dense_stream_active_v190() const noexcept { return dense_stream_v190_ && dense_stream_v190_->active; }

void Pack::begin_dense_stream_v190() {
    if (!dense_stream_v190_) fail("v190 dense streaming disabled");
    auto& state = *dense_stream_v190_;
    state.usable();
    try {
        if (state.active || !state.base || !state.layers) fail("v190 dense begin requires restored dense weights");
        check_cuda(cudaStreamSynchronize(nullptr), "v190 begin consumer boundary");
        size_t free = 0, total = 0;
        check_cuda(cudaMemGetInfo(&free, &total), "v190 memory before dense release");
        state.free_before_begin = free;
        check_cuda(cudaFree(state.layers), "v190 release resident ordinary layer dense");
        state.layers = nullptr;
        for (auto& item : state.all_layer_items) item.tensor->device = nullptr;
        state.active = true;
        check_cuda(cudaStreamCreateWithFlags(&state.upload, cudaStreamNonBlocking), "v190 layer upload stream");
        for (auto& bank : state.banks) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&bank.device), state.bank_capacity), "v190 bounded GPU layer bank");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&bank.host), state.bank_capacity, cudaHostAllocDefault),
                       "v190 bounded pinned layer bank");
            check_cuda(cudaEventCreateWithFlags(&bank.copied, cudaEventDisableTiming), "v190 layer copied event");
            check_cuda(cudaEventCreateWithFlags(&bank.used, cudaEventDisableTiming), "v190 layer consumed event");
        }
        state.reclaimed = state.layer_bytes - 2 * state.bank_capacity;
        ++state.begins;
        check_cuda(cudaMemGetInfo(&free, &total), "v190 memory after dense banks");
        state.free_after_begin = free;
    } catch (...) { state.failed = true; throw; }
}

void Pack::prefetch_dense_layer_v190(int layer) {
    if (!dense_stream_v190_) fail("v190 dense streaming disabled");
    auto& state = *dense_stream_v190_;
    state.usable();
    try {
        if (!state.active || layer < 0 || layer >= kLayers) fail("v190 invalid dense prefetch boundary");
        ++state.prefetches;
        auto& bank = state.banks[layer % 2];
        if (bank.submitted && bank.layer == layer) return;
        if (!bank.retired || bank.activated) fail("v190 cannot overwrite unretired layer bank");
        if (bank.submitted) {
            // The host slot can be reused as soon as its old H2D completed.
            check_cuda(cudaEventSynchronize(bank.copied), "v190 protect pinned source lifetime");
            // The device slot may be overwritten only after all consumers.
            check_cuda(cudaStreamWaitEvent(state.upload, bank.used, 0), "v190 protect layer device lifetime");
        }
        state.clear_layer_views(bank.layer);
        std::vector<uint8_t> scratch;
        for (const auto& item : state.layer_items[layer]) {
            const auto& tensor = *item.tensor;
            for (size_t offset = 0; offset < tensor.bytes; offset += DenseStreamStateV190::kCopyChunk) {
                const size_t bytes = std::min(DenseStreamStateV190::kCopyChunk, size_t(tensor.bytes) - offset);
                const auto* source = state.weights->read_range(tensor.name, offset, bytes, scratch);
                state.scratch_peak = std::max(state.scratch_peak, scratch.size());
                std::memcpy(bank.host + item.bank_offset + offset, source, bytes);
            }
        }
        // Include alignment gaps so each original tensor and scale is at its
        // checked256-byte bank offset. They are not exposed as tensor payload.
        check_cuda(cudaMemcpyAsync(bank.device, bank.host, state.layer_span[layer], cudaMemcpyHostToDevice, state.upload),
                   "v190 upload exact layer dense bank");
        check_cuda(cudaEventRecord(bank.copied, state.upload), "v190 record layer ready");
        bank.layer = layer; bank.submitted = true; bank.activated = false; bank.retired = false;
        state.layer_upload_bytes += state.layer_span[layer];
        ++state.layer_uploads;
    } catch (...) { state.failed = true; throw; }
}

void Pack::activate_dense_layer_v190(int layer, cudaStream_t consumer) {
    if (!dense_stream_v190_) fail("v190 dense streaming disabled");
    auto& state = *dense_stream_v190_;
    state.usable();
    try {
        if (!state.active || layer < 0 || layer >= kLayers) fail("v190 invalid dense activation boundary");
        auto& bank = state.banks[layer % 2];
        if (!bank.submitted || bank.layer != layer) prefetch_dense_layer_v190(layer);
        if (bank.activated || bank.retired) fail("v190 layer activation must be consumed exactly once");
        check_cuda(cudaStreamWaitEvent(consumer, bank.copied, 0), "v190 current layer waits for dense upload");
        for (auto& item : state.layer_items[layer]) item.tensor->device = bank.device + item.bank_offset;
        bank.activated = true;
        ++state.activations;
    } catch (...) { state.failed = true; throw; }
}

void Pack::retire_dense_layer_v190(int layer, cudaStream_t consumer) {
    if (!dense_stream_v190_) fail("v190 dense streaming disabled");
    auto& state = *dense_stream_v190_;
    state.usable();
    try {
        if (!state.active || layer < 0 || layer >= kLayers) fail("v190 invalid dense retire boundary");
        auto& bank = state.banks[layer % 2];
        if (!bank.submitted || bank.layer != layer || !bank.activated || bank.retired)
            fail("v190 retire requires activated current layer");
        check_cuda(cudaEventRecord(bank.used, consumer), "v190 record layer consumed");
        bank.activated = false; bank.retired = true;
        ++state.retires;
    } catch (...) { state.failed = true; throw; }
}

void Pack::end_dense_stream_v190() {
    if (!dense_stream_v190_) fail("v190 dense streaming disabled");
    auto& state = *dense_stream_v190_;
    state.usable();
    try {
        if (!state.active || state.layers) fail("v190 restore requires active streamed dense state");
        for (const auto& bank : state.banks)
            if (bank.activated || (bank.submitted && !bank.retired)) fail("v190 restore has unretired layer bank");
        // Caller frees its large PFscratch first. Free bounded layer banks
        // before allocating the restored full layer arena to avoid a VRAM peak.
        state.release_banks();
        for (auto& item : state.all_layer_items) item.tensor->device = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&state.layers), state.layer_bytes), "v190 restore full layer arena");
        state.copy_group(state.all_layer_items, state.layers, true);
        state.active = false;
        ++state.restores;
        size_t free = 0, total = 0;
        check_cuda(cudaMemGetInfo(&free, &total), "v190 memory after dense restore");
        state.free_after_restore = free;
    } catch (...) { state.failed = true; throw; }
}

std::array<uint64_t, 24> Pack::dense_stream_info_v190() const noexcept {
    std::array<uint64_t, 24> info{};
    info[0] = 190;
    if (!dense_stream_v190_) {
        info[10] = arena_ && native_ ? native_->dense_arena_bytes() : 0;
        return info;
    }
    const auto& s = *dense_stream_v190_;
    size_t gpu_banks = 0, pinned_banks = 0;
    for (const auto& b : s.banks) {
        if (b.device) gpu_banks += s.bank_capacity;
        if (b.host) pinned_banks += s.bank_capacity;
    }
    info = {190, 1, uint64_t(s.active), s.begins, s.restores, s.base_bytes, s.layer_bytes, s.bank_capacity,
            gpu_banks, pinned_banks, (s.base ? s.base_bytes : 0) + (s.layers ? s.layer_bytes : 0) + gpu_banks,
            s.reclaimed, s.layer_upload_bytes, s.layer_uploads, s.prefetches, s.activations, s.retires,
            s.restore_bytes, s.restore_tensor_copies, s.free_before_begin, s.free_after_begin, s.free_after_restore,
            s.scratch_peak, uint64_t(s.failed)};
    return info;
}

Pack::Pack(const std::string& dir) : dir_(dir) {
    // pack_info.txt is written last: without "finished 1" the pack is incomplete
    {
        auto f = open_text(dir + "/pack_info.txt");
        std::string key, line;
        bool finished = false;
        while (std::getline(f, line)) {
            std::istringstream s(line);
            long long v = 0;
            s >> key >> v;
            if (key == "finished") finished = v == 1;
            if (key == "layers") n_layers_ = (int) v;
            if (key == "experts") n_experts_ = (int) v;
        }
        if (!finished) fail(dir + " is not a finished pack (pack_info.txt has no 'finished 1')");
        if (n_layers_ != kLayers || n_experts_ != kExperts) fail("layer or expert count does not match config.hpp");
    }
    {
        auto f = open_text(dir + "/index.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            DenseTensor t;
            std::string dt;
            int ndim = 0;
            s >> t.name >> dt >> ndim;
            if (!s || ndim < 0 || ndim > 16) fail("bad tensor rank: " + line);
            t.dtype = parse_dtype(dt);
            t.shape.resize(ndim);
            for (auto& d : t.shape) s >> d;
            s >> t.file_offset >> t.bytes;
            if (!s) fail("bad index.txt line: " + line);
            uint64_t bytes = dtype_size(t.dtype);
            for (auto d : t.shape) {
                if (d <= 0 || (uint64_t) d > std::numeric_limits<uint64_t>::max() / bytes)
                    fail("invalid or overflowing shape for " + t.name);
                bytes *= (uint64_t) d;
            }
            if (bytes != t.bytes) fail("size mismatch for " + t.name);
            dense_[t.name] = t;
        }
    }
    {
        experts_.resize((size_t) n_layers_ * n_experts_);
        std::vector<bool> seen(experts_.size(), false);
        auto f = open_text(dir + "/experts.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            int l = 0, e = 0;
            ExpertSlot x;
            s >> l >> e >> x.offset >> x.bytes >> x.bits[0] >> x.bits[1] >> x.bits[2];
            for (int c = 0; c < 12; ++c) {
                std::string item;
                s >> item;
                const auto a = item.find(':'), b = item.rfind(':');
                if (a == std::string::npos || b <= a || item.substr(0, a) != kComp[c])
                    fail("bad component in experts.txt: " + item);
                x.comp_off[c] = component_number(item.substr(a + 1, b - a - 1));
                x.comp_bytes[c] = component_number(item.substr(b + 1));
                if (!contains(x.bytes, x.comp_off[c], x.comp_bytes[c])) fail("component outside its slot: " + line);
            }
            if (!s || l < 0 || l >= n_layers_ || e < 0 || e >= n_experts_) fail("bad experts.txt line: " + line);
            // The tiers round slot sizes up to 4 KiB.
            if (x.bytes > std::numeric_limits<size_t>::max() - 4095)
                fail("slot size cannot be aligned");
            experts_[(size_t) l * n_experts_ + e] = x;
            seen[(size_t) l * n_experts_ + e] = true;
        }
        for (bool b : seen)
            if (!b) fail("experts.txt does not list every expert");
    }
    {
        auto f = open_text(dir + "/engram.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            EngramTable t;
            s >> t.layer >> t.rows >> t.dim >> t.weight_offset >> t.scale_offset;
            std::getline(s >> std::ws, t.path);
            if (!s || t.path.empty() || t.rows <= 0 || t.dim <= 0) fail("bad engram.txt line: " + line);
            engram_.push_back(t);
        }
    }
    {
        auto f = open_text(dir + "/engram_hash.txt");
        const size_t tables = engram_.size();
        hash_.multipliers.resize(tables);
        hash_.primes.resize(tables);
        hash_.offsets.resize(tables);
        std::string line;
        while (std::getline(f, line)) {
            if (skip(line)) continue;
            std::istringstream s(line);
            std::string key;
            s >> key;
            if (key == "max_ngram") s >> hash_.max_ngram;
            else if (key == "n_heads") s >> hash_.n_heads;
            else if (key == "pad") s >> hash_.pad;
            else if (key == "vocab") { long long v; s >> v; }
            else if (key == "layers") {
                int l;
                while (s >> l) hash_.layers.push_back(l);
                if (!s.eof()) fail("bad engram hash layers");
                continue;
            } else {
                int li = 0;
                if (!(s >> li) || li < 0 || (size_t) li >= tables) fail("bad engram hash table index");
                std::vector<int64_t> v;
                long long x;
                while (s >> x) v.push_back(x);
                auto& dst = key == "multipliers" ? hash_.multipliers : key == "primes" ? hash_.primes : hash_.offsets;
                if (key != "multipliers" && key != "primes" && key != "offsets") fail("bad engram_hash.txt key " + key);
                if (!s.eof() || !dst[li].empty()) fail("bad or repeated engram hash array");
                dst[li] = v;
                continue;
            }
            if (!s) fail("bad engram hash scalar: " + line);
        }
        if (hash_.layers.size() != tables) fail("engram hash layer count mismatch");
        if (tables) {
            // The engine has room for 24 row IDs per table and token.
            if (hash_.max_ngram < 2 || hash_.max_ngram > 25 || hash_.n_heads < 1 ||
                hash_.n_heads > 24 / (hash_.max_ngram - 1)) fail("bad engram hash dimensions");
            const size_t cols = (size_t) (hash_.max_ngram - 1) * hash_.n_heads;
            for (size_t i = 0; i < tables; ++i) {
                const int layer = hash_.layers[i];
                if (layer < 0 || layer >= n_layers_ || layer != engram_[i].layer ||
                    (i && layer <= hash_.layers[i - 1])) fail("engram hash layer mismatch");
                if (hash_.multipliers[i].size() != (size_t) hash_.max_ngram ||
                    hash_.primes[i].size() != cols || hash_.offsets[i].size() != cols)
                    fail("missing or incomplete engram hash array");
                for (size_t c = 0; c < cols; ++c) {
                    const int64_t prime = hash_.primes[i][c], off = hash_.offsets[i][c];
                    if (prime <= 0 || off < 0 || !contains((uint64_t) engram_[i].rows, off, prime))
                        fail("bad engram hash prime or offset");
                }
            }
        }
        std::ifstream tm(dir + "/engram_tokenmap.bin", std::ios::binary | std::ios::ate);
        if (!tm) fail("cannot open engram_tokenmap.bin");
        const auto n = (size_t) tm.tellg() / 4;
        if (n != (size_t) kVocab) fail("engram token map has the wrong size");
        hash_.token_map.resize(n);
        tm.seekg(0);
        tm.read(reinterpret_cast<char*>(hash_.token_map.data()), (std::streamsize) (n * 4));
    }
    // shape spot checks: a different model fails here, not in the logits
    auto expect = [&](const char* n, std::vector<int64_t> shape) {
        if (dense(n).shape != shape) fail(std::string("unexpected shape for ") + n);
    };
    expect("embed.weight", {kVocab, kDim});
    expect("head.weight", {kVocab, kDim});
    expect("layers.0.attn.wq_b.weight", {kHeads * kHeadDim, kQLora});
    expect("layers.0.attn.wo_a.weight", {kOGroups * kOLora, kHeads * kHeadDim / kOGroups});
    expect("layers.0.hc_attn_fn", {kHcMix, kHc * kDim});
}

Pack::~Pack() {
    if (arena_) cudaFree(arena_);
    if (experts_map_) munmap(const_cast<uint8_t*>(experts_map_), experts_map_bytes_);
}

Pack::Pack(std::shared_ptr<NativeCheckpoint> checkpoint,EngramHash hash)
    :dir_("<borrowed-verified-checkpoint>"),n_layers_(kLayers),n_experts_(kExperts),hash_(std::move(hash)),native_(std::move(checkpoint)){
    if(!native_)fail("missing native checkpoint owner");
    if(hash_.max_ngram!=4||hash_.n_heads!=8||hash_.layers!=std::vector<int>{1,14}||hash_.token_map.size()!=size_t(kVocab)||
       hash_.multipliers.size()!=2||hash_.primes.size()!=2||hash_.offsets.size()!=2)fail("native Engram hash geometry differs");
    if(hash_.pad!=hash_.token_map[2])fail("native Engram padding differs");
    for(int32_t token:hash_.token_map)if(token<0||token>=99092)fail("invalid compressed tokenizer ID");
    for(int i=0;i<2;i++){
        if(hash_.multipliers[i].size()!=4||hash_.primes[i].size()!=24||hash_.offsets[i].size()!=24)fail("incomplete Engram hash metadata");
        int64_t total=0;
        for(int j=0;j<24;j++){
            if(hash_.primes[i][j]<1||hash_.primes[i][j]>400000000||hash_.offsets[i][j]!=total)fail("invalid Engram hash offsets/primes");
            total+=hash_.primes[i][j];
        }
        const auto&table=native_->weights()->at("layers."+std::to_string(hash_.layers[i])+".engram.embed.weight");
        if(total!=table.shape[0])fail("Engram hash modulus coverage differs from source rows");
        EngramTable t;t.layer=hash_.layers[i];t.rows=total;t.dim=256;engram_.push_back(t);
    }
    auto dtype=[](TensorEncoding e){switch(e){
        case TensorEncoding::Bf16:return DType::BF16;case TensorEncoding::F32:return DType::F32;
        case TensorEncoding::F8e4m3:return DType::F8E4M3;case TensorEncoding::E8m0:return DType::E8M0;
        default:fail("unsupported dense native encoding");
    }};
    uint64_t off=0;
    for(const auto&name:native_->dense_names()){
        const auto&t=native_->weights()->at(name);off=(off+255)&~uint64_t(255);
        dense_.emplace(name,DenseTensor{name,dtype(t.encoding),t.shape,off,t.bytes,nullptr});off+=t.bytes;
    }
    if(off!=native_->dense_arena_bytes())fail("native dense arena size differs");
}

const DenseTensor& Pack::dense(const std::string& name) const {
    auto it = dense_.find(name);
    if (it == dense_.end()) fail("no dense tensor " + name);
    if (dense_stream_v190_) {
        dense_stream_v190_->usable();
        if (dense_stream_v190_->active && DenseStreamStateV190::ordinary_layer(name) >= 0 && !it->second.device)
            fail("v190 ordinary layer dense tensor is not activated: " + name);
    }
    return it->second;
}

uint64_t Pack::upload_dense() {
    if(arena_)fail("dense arena already uploaded");
    if (dense_stream_v190_) {
        auto& state = *dense_stream_v190_;
        state.usable();
        try {
            if (state.base || state.layers || state.active) fail("v190 split dense already uploaded");
            state.upload_mapped_embedding_r1();
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&state.base), state.base_bytes), "v190 stable base dense arena");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&state.layers), state.layer_bytes), "v190 ordinary layer dense arena");
            state.copy_group(state.base_items, state.base, false);
            state.copy_group(state.all_layer_items, state.layers, false);
            return state.base_bytes + state.layer_bytes;
        } catch (...) { state.failed = true; throw; }
    }
    if(native_){
        const auto total=native_->dense_arena_bytes();
        check_cuda(cudaMalloc(&arena_,total),"cudaMalloc native dense arena");
        std::vector<uint8_t>scratch;
        for(auto&[name,t]:dense_){
            for(size_t off=0;off<t.bytes;off+=64ull<<20){
                const size_t length=std::min<size_t>(64ull<<20,t.bytes-off);
                const auto*source=native_->weights()->read_range(name,off,length,scratch);
                check_cuda(cudaMemcpy(static_cast<uint8_t*>(arena_)+t.file_offset+off,source,length,cudaMemcpyHostToDevice),"upload borrowed dense tensor");
            }
            t.device=static_cast<uint8_t*>(arena_)+t.file_offset;
        }
        return total;
    }
    const std::string path = dir_ + "/dense.bin";
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        fail("cannot size dense.bin");
    }
    const uint64_t total = (uint64_t) st.st_size;
    for (const auto& kv : dense_) {
        if (!contains(total, kv.second.file_offset, kv.second.bytes)) {
            close(fd);
            fail("tensor past end of dense.bin: " + kv.first);
        }
    }
    check_cuda(cudaMalloc(&arena_, total), "cudaMalloc dense arena");
    constexpr uint64_t kChunk = 64ull << 20;
    void* host = nullptr;
    check_cuda(cudaMallocHost(&host, kChunk), "cudaMallocHost staging");
    for (uint64_t off = 0; off < total; off += kChunk) {
        const uint64_t n = std::min(kChunk, total - off);
        uint64_t got = 0;
        while (got < n) {
            const ssize_t r = pread(fd, (char*) host + got, n - got, (off_t) (off + got));
            if (r <= 0) fail("short read of dense.bin");
            got += (uint64_t) r;
        }
        check_cuda(cudaMemcpy((char*) arena_ + off, host, n, cudaMemcpyHostToDevice), "upload dense");
    }
    cudaFreeHost(host);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // its 11 GB stay out of the file cache the experts need
    close(fd);
    for (auto& kv : dense_) {
        kv.second.device = (char*) arena_ + kv.second.file_offset;
    }
    return total;
}

void Pack::map_experts() {
    if(native_)return; // Native experts retain borrowed packed RAM through their registry.
    const std::string path = dir_ + "/experts.bin";
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        fail("cannot size experts.bin");
    }
    const uint64_t total = (uint64_t) st.st_size;
    for (const auto& expert : experts_) {
        if (!contains(total, expert.offset, expert.bytes)) {
            close(fd);
            fail("expert past end of experts.bin");
        }
    }
    void* p = mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) fail("mmap experts.bin failed");
    if (experts_map_) munmap(const_cast<uint8_t*>(experts_map_), experts_map_bytes_);
    experts_map_ = (const uint8_t*) p;
    experts_map_bytes_ = total;
}

}  // namespace strata::ds41

#include "headless_resource_v320.inc"
