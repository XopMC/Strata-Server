#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <fcntl.h>
#include <unistd.h>

namespace native_plan_v399 {
// Ordinary HOST metadata only. base is compared as an integer, never dereferenced.
// slot_off is the original HOST offset table; offset_count comes from full_slots().
struct Home {
    int64_t slots = 0, offset_count = 0, arena_bytes = 0, cache_blob = 0;
    const uint8_t* base = nullptr;
    const uint64_t* slot_off = nullptr;
};
struct Record {
    uint64_t request, window, callback;
    int64_t position;
    int32_t layer, home, T, k, plan_cap, cpu_jobs;
    int32_t counts[4], ids[80], host_res[80], start[81], dst[80], tok[80];
    uint64_t ptr_offset_bytes[80], expected_offset_bytes[80];
};
static_assert(sizeof(uintptr_t) == 8, "V399 proposal is bound to the original 64-bit host ABI");
static_assert(sizeof(Record) == 2960);

class Observer {
public:
    static constexpr size_t Capacity = 65536; // 1365 complete 48-layer target windows.
    static constexpr int64_t ExpectedSlots[2] = {7460, 6645}; // Private actual-filled256 binding.
    static constexpr size_t RingBytes = Capacity * sizeof(Record); // 185 MiB ordinary HOST memory.
private:
    std::unique_ptr<Record[]> ring_;
    Home homes_[2];
    std::unique_ptr<uint64_t[]> offsets_[2]; // Owned complete admitted HOST layout snapshot, before READY.
    std::FILE* file_ = nullptr;
    const char* error_ = nullptr;
    size_t used_ = 0;
    uint64_t request_ = 0, window_ = 0, callback_ = 0;
    int64_t position_ = 0;
    int T_ = 0, next_layer_ = 0;
    bool request_open_ = false, window_open_ = false;
    bool refuse(const char* why) noexcept { if (!error_) error_ = why; return false; }
    template<class Dispatch> bool supported(const Dispatch& d) noexcept {
        if (d.failed) return refuse("V399 original expert dispatch failed");
        if (d.remote_count || d.peer || d.fs || d.pcie_num || d.pcie_model.on)
            return refuse("V399 remote/peer/foresight/PCIe path unsupported");
        if (d.n_expert != 512 || !d.host_res)
            return refuse("V399 requires original 48x512 host residency");
        return true;
    }
    // No native payload stride is inferred. Nonuniform offsets are authoritative.
    bool expected_offset(const Home& h, int32_t slot, uint64_t& offset) noexcept {
        if (slot < 0 || int64_t(slot) >= h.slots)
            return refuse("V399 logical slot outside explicitly admitted home capacity");
        if (h.slot_off) {
            if (int64_t(slot) >= h.offset_count)
                return refuse("V399 logical slot outside admitted HOST offset-table length");
            offset = h.slot_off[slot];
        } else {
            if (h.cache_blob <= 0 || uint64_t(slot) > UINT64_MAX / uint64_t(h.cache_blob))
                return refuse("V399 uniform slot byte offset overflow");
            offset = uint64_t(slot) * uint64_t(h.cache_blob);
        }
        if (offset >= uint64_t(h.arena_bytes))
            return refuse("V399 expected byte offset outside admitted original cache arena");
        return true;
    }
    void i32_array(const char* name, const int32_t* p, int n) noexcept {
        std::fprintf(file_, ",\"%s\":[", name);
        for (int i=0;i<n;++i) std::fprintf(file_, "%s%d", i ? "," : "", p[i]);
        std::fputc(']', file_);
    }
    void u64_array(const char* name, const uint64_t* p, int n) noexcept {
        std::fprintf(file_, ",\"%s\":[", name);
        for (int i=0;i<n;++i) std::fprintf(file_, "%s%llu", i ? "," : "", (unsigned long long)p[i]);
        std::fputc(']', file_);
    }
public:
    Observer() = default;
    Observer(const Observer&) = delete;
    Observer& operator=(const Observer&) = delete;
    ~Observer() { if (file_) std::fclose(file_); } // Never dump an incomplete request.
    bool enabled() const noexcept { return file_ != nullptr; }
    const char* error() const noexcept { return error_ ? error_ : "V399 unspecified observer refusal"; }

    // Before READY only: actual per-home counts/bytes are explicit caller inputs.
    // No fallback to d.cache: SplitDrive changes base/offset table by layer home.
    bool initialize(const char* path, bool admitted, const Home (&homes)[2]) noexcept {
        if (!path) return true;
        if (!admitted || path[0] != '/') return refuse("V399 unsupported startup or nonabsolute output path");
        if (file_ || ring_) return refuse("V399 observer initialized twice");
        for (int h=0;h<2;++h) {
            const Home& v=homes[h];
            if (v.slots != ExpectedSlots[h] || v.offset_count < v.slots || v.offset_count > 1048576 ||
                v.arena_bytes <= 0 || v.cache_blob <= 0 || !v.base)
                return refuse("V399 actual home capacity/layout differs from explicit filled256 binding");
            const uintptr_t base=reinterpret_cast<uintptr_t>(v.base);
            if (uint64_t(v.arena_bytes) > std::numeric_limits<uintptr_t>::max()-base)
                return refuse("V399 admitted cache address range overflow");
            homes_[h]=v;
            offsets_[h].reset(new (std::nothrow) uint64_t[size_t(v.slots)]);
            if (!offsets_[h]) return refuse("V399 admitted HOST offset snapshot unavailable");
            uint64_t previous=0;
            for (int64_t slot=0;slot<v.slots;++slot) {
                uint64_t off=0;
                if (!expected_offset(homes_[h],int32_t(slot),off)) return false;
                if ((slot==0 && off!=0) || (slot>0 && off<=previous))
                    return refuse("V399 original slot byte offsets are not a strict zero-based layout");
                offsets_[h][size_t(slot)]=off; previous=off;
            }
        }
        ring_.reset(new (std::nothrow) Record[Capacity]);
        if (!ring_) return refuse("V399 bounded preallocated HOST record ring unavailable");
        volatile unsigned char* pages=reinterpret_cast<volatile unsigned char*>(ring_.get());
        for (size_t i=0;i<RingBytes;i+=4096) pages[i]=0;
        pages[RingBytes-1]=0;
        const int fd=::open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if (fd<0) return refuse("V399 cannot create fresh output file");
        file_=::fdopen(fd,"w");
        if (!file_) { ::close(fd); return refuse("V399 cannot open output stream"); }
        return true;
    }

    template<class Dispatch> bool begin_request(const Dispatch& d, bool admitted) noexcept {
        if (!enabled()) return true;
        if (error_ || request_open_ || window_open_ || request_==UINT64_MAX)
            return refuse("V399 invalid request lifecycle");
        if (!admitted || !supported(d)) return refuse("V399 unsupported request routing mode");
        ++request_; used_=0; window_=0; callback_=0; request_open_=true;
        return true;
    }
    bool begin_window(int T, int64_t position) noexcept {
        if (!enabled()) return true;
        if (error_ || !request_open_ || window_open_ || T<1 || T>8 || position<0)
            return refuse("V399 invalid explicit target-window boundary");
        if (Capacity-used_<48) return refuse("V399 bounded HOST ring full before target window");
        ++window_; position_=position; T_=T; next_layer_=0; window_open_=true;
        return true;
    }

    // AFTER original expert_pool_dispatch_multi, on that same original HOST thread.
    // No allocation, I/O, timing, atomics, GPU API/readback/dereference or numerical write.
    // Only original HOST plan/residency reads and owned bounded metadata stores.
    template<class Dispatch>
    void observe(const Dispatch& d, const int32_t* ids, int64_t n_tok, int64_t k, int64_t layer) noexcept {
        if (!enabled() || !request_open_ || error_) return;
        if (!window_open_) { refuse("V399 untagged callback during generation"); return; }
        if (!supported(d)) return;
        if (layer!=next_layer_ || layer<0 || layer>=48 || n_tok!=T_ || k!=10 || !ids || d.layers!=layer+1) {
            refuse("V399 callback layer/T/k/dispatch order differs from explicit G1 window"); return;
        }
        const int home=layer<24 ? 0 : 1, n=T_*10;
        const Home& h=homes_[home];
        if (d.cache_base!=h.base || d.cache_slot_off!=h.slot_off || d.cache_blob!=h.cache_blob) {
            refuse("V399 actual callback cache differs from explicitly admitted logical home"); return;
        }
        const auto* p=d.plan;
        if (!p || p->cap<n || p->cap>80 || !p->counts || !p->start || !p->dst || !p->tok || !p->ptr || !p->ptr2 || !p->start2) {
            refuse("V399 original published HOST plan unavailable or unsupported routed capacity"); return;
        }
        // Frozen V355 mapped() zeroes the entire allocation (verify.cpp421-437).
        // Original producer writes counts[0..2] only (expert_source.cpp4954-4956),
        // so reserved counts[3]==0 is this frozen ABI guarantee, not a generic sink rule.
        // Establish that four-word layout BEFORE reading word3 or start2[0].
        const uint64_t cap=uint64_t(p->cap), ptr_i32=(4+(cap+1)+cap+cap+1)&~uint64_t(1);
        const uint64_t words=ptr_i32+4*cap+(cap+1)+1;
        const uintptr_t base=reinterpret_cast<uintptr_t>(p->counts);
        if ((base&7u) || words*4>std::numeric_limits<uintptr_t>::max()-base ||
            reinterpret_cast<uintptr_t>(p->start)!=base+16 ||
            reinterpret_cast<uintptr_t>(p->dst)!=base+4*(4+cap+1) ||
            reinterpret_cast<uintptr_t>(p->tok)!=base+4*(4+cap+1+cap) ||
            reinterpret_cast<uintptr_t>(p->ptr)!=base+4*ptr_i32 ||
            reinterpret_cast<uintptr_t>(p->ptr2)!=base+4*(ptr_i32+2*cap) ||
            reinterpret_cast<uintptr_t>(p->start2)!=base+4*(ptr_i32+4*cap)) {
            refuse("V399 original mapped plan ABI is not the admitted four-count layout"); return;
        }
        const int groups=p->counts[0], routes=p->counts[1];
        if (groups<0 || groups>n || routes<groups || routes>n || p->counts[2]!=0 || p->counts[3]!=0 ||
            (groups==0 && routes!=0) || p->start[0]!=0 || p->start[groups]!=routes ||
            p->start2[0]!=routes || p->cpu_jobs<0 || p->cpu_jobs>n-routes || used_>=Capacity) {
            refuse("V399 invalid original native/CPU counts, reserved header or no-PCIe partition"); return;
        }
        Record& r=ring_[used_];
        r.request=request_; r.window=window_; r.callback=callback_; r.position=position_;
        r.layer=int32_t(layer); r.home=home; r.T=T_; r.k=10; r.plan_cap=int32_t(p->cap); r.cpu_jobs=p->cpu_jobs;
        for (int j=0;j<4;++j) r.counts[j]=p->counts[j];
        for (int row=0;row<n;++row) {
            const int e=ids[row];
            if (e<0 || e>=512) { refuse("V399 routed expert ID outside original 512"); return; }
            const int slot=d.host_res[size_t(layer)*512+size_t(e)];
            if (slot<-1 || int64_t(slot)>=h.slots) { refuse("V399 routed residency outside original CPU/native slot domain"); return; }
            r.ids[row]=e; r.host_res[row]=slot;
            for (int j=0;j<row;++j) {
                if ((j/10==row/10 && r.ids[j]==e) || (r.ids[j]==e && r.host_res[j]!=slot) ||
                    (slot>=0 && r.host_res[j]==slot && r.ids[j]!=e)) {
                    refuse("V399 duplicate per-token expert or inconsistent/aliased logical residency"); return;
                }
            }
        }
        // Validate ACTUAL published start/dst/tok against every original native row.
        // Never use an unvalidated p->dst or p->start value as a HOST array index.
        int group=0, entry=0, cpu_jobs=0;
        for (int first=0;first<n;++first) {
            bool seen=false;
            for (int j=0;j<first;++j) if (r.ids[j]==r.ids[first]) { seen=true; break; }
            if (seen) continue;
            if (r.host_res[first]<0) { ++cpu_jobs; continue; }
            if (group>=groups || p->start[group]!=entry) { refuse("V399 native first-occurrence group order differs"); return; }
            r.start[group]=p->start[group];
            uint64_t expected=0;
            if (!expected_offset(h,r.host_res[first],expected)) return;
            if (expected!=offsets_[home][size_t(r.host_res[first])]) {
                refuse("V399 original slot offset layout changed after READY"); return;
            }
            const uint64_t native_ptr=uint64_t(p->ptr[group]);
            const uint64_t cache_base=uint64_t(reinterpret_cast<uintptr_t>(d.cache_base));
            if (native_ptr<cache_base || native_ptr-cache_base!=expected) {
                refuse("V399 original native pointer BYTE OFFSET differs from actual logical slot layout"); return;
            }
            r.ptr_offset_bytes[group]=native_ptr-cache_base;
            r.expected_offset_bytes[group]=expected;
            for (int row=first;row<n;++row) if (r.ids[row]==r.ids[first]) {
                if (entry>=routes || p->dst[entry]!=row || p->tok[entry]!=row/10) {
                    refuse("V399 original native row partition/token incidence differs from full routed IDs"); return;
                }
                r.dst[entry]=p->dst[entry]; r.tok[entry]=p->tok[entry]; ++entry;
            }
            if (p->start[group+1]!=entry) { refuse("V399 native contiguous group end differs"); return; }
            ++group;
        }
        if (group!=groups || entry!=routes || cpu_jobs!=p->cpu_jobs) {
            refuse("V399 incomplete native partition or original distinct CPU job classification differs"); return;
        }
        r.start[groups]=p->start[groups];
        // Partially filled records are never committed or dumped after a refusal.
        ++used_; ++callback_; ++next_layer_;
    }
    bool end_window() noexcept {
        if (!enabled()) return true;
        if (error_ || !request_open_ || !window_open_ || next_layer_!=48)
            return refuse("V399 incomplete/refused target window");
        window_open_=false; return true;
    }

    // Only after original decode_ms snapshot and successful original wait_commit.
    bool dump_request(const char* finish) noexcept {
        if (!enabled()) return true;
        if (error_ || !request_open_ || window_open_ || !window_ || used_!=window_*48)
            return refuse("V399 incomplete/refused request cannot be published");
        request_open_=false;
        uint64_t histogram[2][9][3][81]={};
        std::fprintf(file_,"{\"kind\":\"request_begin\",\"schema\":399,\"request\":%llu,\"windows\":%llu,\"callbacks\":%zu,\"scope\":\"serial_target_G1_N1\",\"home_slot_capacities\":[%lld,%lld],\"home_arena_bytes\":[%lld,%lld],\"pointer_encoding\":\"BYTE_OFFSET_FROM_ORIGINAL_HOME_CACHE_BASE\",\"finish\":\"",
            (unsigned long long)request_,(unsigned long long)window_,used_,
            (long long)homes_[0].slots,(long long)homes_[1].slots,(long long)homes_[0].arena_bytes,(long long)homes_[1].arena_bytes);
        for (const unsigned char* s=reinterpret_cast<const unsigned char*>(finish?finish:"unknown");*s;++s) {
            if (*s=='"' || *s=='\\') { std::fputc('\\',file_); std::fputc(*s,file_); }
            else if (*s<32) std::fprintf(file_,"\\u%04x",unsigned(*s));
            else std::fputc(*s,file_);
        }
        std::fputs("\"}\n",file_);
        for (int home=0;home<2;++home) {
            const Home& h=homes_[home];
            std::fprintf(file_,"{\"kind\":\"home_layout\",\"schema\":399,\"request\":%llu,\"logical_home\":%d,\"slots\":%lld,\"offset_count\":%lld,\"arena_bytes\":%lld,\"cache_blob\":%lld,\"offset_mode\":\"%s\"",
                (unsigned long long)request_,home,(long long)h.slots,(long long)h.offset_count,
                (long long)h.arena_bytes,(long long)h.cache_blob,h.slot_off?"original_host_slot_off":"original_uniform_cache_blob");
            u64_array("slot_offsets_bytes",offsets_[home].get(),int(h.slots));
            std::fputs("}\n",file_);
        }
        for (size_t i=0;i<used_;++i) {
            const Record& r=ring_[i]; const int groups=r.counts[0], routes=r.counts[1], n=r.T*r.k;
            std::fprintf(file_,"{\"kind\":\"callback\",\"schema\":399,\"request\":%llu,\"window\":%llu,\"position\":%lld,\"callback\":%llu,\"layer\":%d,\"logical_home\":%d,\"T\":%d,\"k\":%d,\"plan_cap\":%d,\"native_groups\":%d,\"native_routes\":%d,\"cpu_jobs\":%d,\"cpu_routes\":%d",
                (unsigned long long)r.request,(unsigned long long)r.window,(long long)r.position,(unsigned long long)r.callback,
                r.layer,r.home,r.T,r.k,r.plan_cap,groups,routes,r.cpu_jobs,n-routes);
            i32_array("counts",r.counts,4); i32_array("ids",r.ids,n); i32_array("host_res",r.host_res,n);
            i32_array("start",r.start,groups+1); i32_array("dst",r.dst,routes); i32_array("tok",r.tok,routes);
            u64_array("ptr_offset_bytes",r.ptr_offset_bytes,groups); u64_array("expected_offset_bytes",r.expected_offset_bytes,groups);
            std::fputs(",\"groups\":[",file_);
            for (int g=0;g<groups;++g) {
                const int first=r.dst[r.start[g]];
                std::fprintf(file_,"%s{\"expert\":%d,\"logical_slot\":%d,\"routes\":%d}",g?",":"",r.ids[first],r.host_res[first],r.start[g+1]-r.start[g]);
            }
            std::fputs("]}\n",file_);
            ++histogram[r.home][r.T][0][groups]; ++histogram[r.home][r.T][1][routes]; ++histogram[r.home][r.T][2][r.cpu_jobs];
        }
        const char* names[3]={"native_groups","native_routes","cpu_jobs"};
        for (int home=0;home<2;++home) for (int T=1;T<=8;++T) for (int kind=0;kind<3;++kind) for (int value=0;value<=80;++value)
            if (histogram[home][T][kind][value]) std::fprintf(file_,"{\"kind\":\"histogram\",\"request\":%llu,\"logical_home\":%d,\"T\":%d,\"metric\":\"%s\",\"value\":%d,\"callbacks\":%llu}\n",
                (unsigned long long)request_,home,T,names[kind],value,(unsigned long long)histogram[home][T][kind][value]);
        if (std::ferror(file_)) return refuse("V399 output write failed; incomplete request");
        std::fprintf(file_,"{\"kind\":\"request_end\",\"schema\":399,\"request\":%llu,\"windows\":%llu,\"callbacks\":%zu,\"overflow\":false,\"complete\":true}\n",
            (unsigned long long)request_,(unsigned long long)window_,used_);
        if (std::fflush(file_) || std::ferror(file_)) return refuse("V399 output flush failed; incomplete request");
        return true;
    }
};
}
