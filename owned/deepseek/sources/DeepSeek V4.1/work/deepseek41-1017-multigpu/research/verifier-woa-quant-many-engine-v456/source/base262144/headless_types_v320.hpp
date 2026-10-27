#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::ds41 {
enum class EngineRoleV320 : uint32_t { Headful = 0, HeadlessPrefillRank1 = 320 };
struct HeadlessOptionsV320 {
    // All persistent requested device payload + transient PP reserve must fit.
    // Actual free memory must additionally retain reserve_bytes. No auto shrink.
    uint64_t max_device_bytes = uint64_t(20) << 30;
    uint64_t reserve_bytes = uint64_t(512) << 20;
};
struct HeadlessBoundaryV320 {
    uint64_t version=320, epoch=0, event=0;
    int32_t layer=-1,next_layer=0,c0=0,rows=0,position=0,absolute_end=0;
    int32_t ratio=0,source_layer=-1,compressed_rows=0,carry=0;
    const uint16_t *window=nullptr,*compressed=nullptr,*index_keys=nullptr;
    const float *kv_state=nullptr,*score_state=nullptr;
    uint64_t window_bytes=0,compressed_bytes=0,index_bytes=0,carry_bytes=0;
    // Borrowed GPU1 pointers/event. Caller must complete its D2H consumer then
    // ACK this exact epoch/layer before next advance/re-record/finish/close.
};
class HeadlessPinnedPrefixV320 {
public:
    static std::shared_ptr<HeadlessPinnedPrefixV320> allocate(uint64_t bytes);
    ~HeadlessPinnedPrefixV320()noexcept;
    HeadlessPinnedPrefixV320(const HeadlessPinnedPrefixV320&)=delete;
    HeadlessPinnedPrefixV320&operator=(const HeadlessPinnedPrefixV320&)=delete;
    void*data()noexcept{return data_;}
    const void*data()const noexcept{return data_;}
    uint64_t capacity()const noexcept{return bytes_;}
private:
    explicit HeadlessPinnedPrefixV320(uint64_t bytes);
    void*data_=nullptr;uint64_t bytes_=0;
};
struct HeadlessPrefixV320 {
    uint64_t version=320, producer_epoch=0;
    int32_t layer=-1, prefix_end=0, source_layer=-1;
    // Canonical disjoint frame: window@0, compressed@window_bytes,
    // index@window_bytes+compressed_bytes; nonsources use only window@0.
    uint64_t window_offset=0,compressed_offset=0,index_offset=0;
    uint64_t window_bytes=0,compressed_bytes=0,index_bytes=0;
    // Typed owning cudaHostAllocPortable buffer, <=512MiB. The caller completes
    // producer D2H before import and stops modifying these spans until receiver
    // H2D finishes. Receiver checks every offset/extent and retains this object.
    std::shared_ptr<const HeadlessPinnedPrefixV320> storage;
};
struct HeadlessFrequencyV320 {
    uint64_t version=320,owner_generation=0,format_epoch=0,pass_epoch=0,tail_counted_uses=0;
    int32_t observed_begin=0,observed_end=0,tail_rows=0;
    std::vector<int> full_prompt; // bounded<=524288 token IDs, not model payload
    std::array<std::array<uint64_t,384>,40> counts{};
};
// headless_info_v320() words: version,role,device,native_math,CPUowner,
// worker_joinable,doorbell,head_pointer,head_saved_bytes,planned_startup_device,
// own_device_payload,current_pack_payload,max_device_budget,reserve,
// active_pass,pending_layer_ACK. Pack embedding pinned copy is disclosed in
// pack_headless_info_v320(): version,headless,head_saved,base,full_layers,
// bank_capacity,pinned_embedding_bytes,current_dense_GPU.
} // namespace strata::ds41
