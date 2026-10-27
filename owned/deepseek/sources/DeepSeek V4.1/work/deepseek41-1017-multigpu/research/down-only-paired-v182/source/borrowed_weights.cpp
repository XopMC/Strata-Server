#include "strata/ds41/borrowed_weights.hpp"
#include <limits>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include "pair_registry_v145.hpp"

namespace strata::ds41 {
namespace {
[[noreturn]] void bad(const char* reason) {
    throw std::invalid_argument(std::string("borrowed weights: ") + reason);
}
size_t multiply(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) bad("shape/byte size overflow");
    return a * b;
}
size_t add_size(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) bad("row size overflow");
    return a + b;
}
bool valid_bits(int bits) { return bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8; }
uintptr_t end_of(const uint8_t* data, size_t bytes) {
    const auto start = reinterpret_cast<uintptr_t>(data);
    if (!data || !bytes || bytes > std::numeric_limits<uintptr_t>::max() - start) bad("invalid memory interval");
    return start + bytes;
}
size_t own_stride(size_t columns, int bits) {
    if (!valid_bits(bits) || columns == 0 || columns % 32) bad("invalid Uniform precision/group32 shape");
    return add_size(multiply(columns / 8, size_t(bits)), multiply(columns / 32, size_t(2)));
}
size_t expected_bytes(const BorrowedTensor& t) {
    if (t.shape.size() > 8) bad("tensor rank exceeds8");
    size_t elements = 1;
    for (int64_t dim : t.shape) {
        if (dim <= 0) bad("nonpositive tensor dimension");
        elements = multiply(elements, static_cast<size_t>(dim));
    }
    if(t.encoding==pair145::Encoding){
        if(!pair145::enabled()||t.shape.size()!=2||t.bits||t.protected_native_rows||
           !((t.shape[0]==2304&&t.shape[1]==5120)||(t.shape[0]==5120&&t.shape[1]==2304)))bad("invalid explicit paired139 geometry");
        return pair145::MatrixBytes;
    }
    if (t.encoding == TensorEncoding::Output16Compact) {
        if(t.shape.size()!=2 || t.shape[0]<16 || t.shape[0]>65536 || t.shape[0]%16 ||
           t.shape[1]<32 || t.shape[1]>65536 || t.shape[1]%32 ||
           (t.bits!=0&&t.bits!=3) || t.protected_native_rows)
            bad("invalid output16-compact-v3 geometry/precision");
        const size_t bytes=multiply(multiply(size_t(t.shape[0]/16),size_t(t.shape[1]/32)),size_t(64*(t.bits?t.bits:4)+32));
        if(bytes>64ull<<20)bad("compact expert exceeds bounded matrix reader");
        return bytes;
    }
    if (t.encoding == TensorEncoding::Uniform) {
        if (t.shape.size() != 2) bad("Uniform requires row-major matrices");
        return multiply(static_cast<size_t>(t.shape[0]), own_stride(static_cast<size_t>(t.shape[1]), t.bits));
    }
    if (t.bits != 0 || t.protected_native_rows) bad("Uniform metadata on a native tensor");
    switch (t.encoding) {
        case TensorEncoding::Mxfp4:
            if (t.shape.size() != 2 || t.shape[1] % 16) bad("MXFP4 requires packed complete32-value groups");
            return elements;
        case TensorEncoding::F8e4m3: case TensorEncoding::E8m0:
        case TensorEncoding::I8: case TensorEncoding::U8: return elements;
        case TensorEncoding::Bf16: case TensorEncoding::F16: case TensorEncoding::I16: return multiply(elements, 2);
        case TensorEncoding::F32: case TensorEncoding::I32: return multiply(elements, 4);
        case TensorEncoding::I64: return multiply(elements, 8);
        default: bad("unknown tensor encoding");
    }
}
bool valid_name(const std::string& name) {
    if (name.empty() || name.size() > 512) return false;
    for (unsigned char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '.')) return false;
    return true;
}
} // namespace

BorrowedWeights::BorrowedWeights(std::shared_ptr<const void> owner, std::vector<MemoryRegion> regions)
    : owner_(std::move(owner)), regions_(std::move(regions)) {
    if (!owner_ || regions_.empty()) bad("an explicit allocation owner and memory regions are required");
    for (const auto& region : regions_) end_of(region.data, region.bytes);
    // Selected owners have46080 real compact mappings plus retained raw spans.
    // Preserve each real allocation; reject overlap and never bridge holes.
    std::sort(regions_.begin(),regions_.end(),[](const MemoryRegion&a,const MemoryRegion&b){
        return reinterpret_cast<uintptr_t>(a.data)<reinterpret_cast<uintptr_t>(b.data);
    });
    uintptr_t previous_end=0;
    for(const auto&region:regions_){
        if(reinterpret_cast<uintptr_t>(region.data)<previous_end)bad("registered RAM allocations overlap");
        previous_end=end_of(region.data,region.bytes);
    }
}

void BorrowedWeights::add(const std::string& name, BorrowedTensor tensor) {
    if (!valid_name(name) || tensors_.count(name)) bad("invalid or duplicate tensor name");
    if (tensor.bytes != expected_bytes(tensor)) bad("payload byte count differs from its encoding/shape");
    if(tensor.encoding==TensorEncoding::Output16Compact||tensor.encoding==pair145::Encoding){
        int layer=-1,expert=-1,count=0;char projection=0;
        if(std::sscanf(name.c_str(),"layers.%d.ffn.experts.%d.w%c.weight%n",&layer,&expert,&projection,&count)!=3 ||
           count!=int(name.size()) || layer<0 || layer>=40 || expert<0 || expert>=384 ||
           (projection!='1'&&projection!='2'&&projection!='3'))bad("compact descriptor requires a canonical target-expert weight name");
        if((reinterpret_cast<uintptr_t>(tensor.data)&1)!=0)bad("compact scale footer requires uint16 alignment");
        if(tensor.encoding==pair145::Encoding&&(reinterpret_cast<uintptr_t>(tensor.data)&((2u<<20)-1)))bad("paired139 requires retained 2MiB-aligned allocation");
    }
    check_region({tensor.data,tensor.bytes});
    tensors_.emplace(name, std::move(tensor));
}
void BorrowedWeights::check_region(MemoryRegion region_to_check) const {
    const auto end = end_of(region_to_check.data, region_to_check.bytes);
    const auto start = reinterpret_cast<uintptr_t>(region_to_check.data);
    const auto following=std::upper_bound(regions_.begin(),regions_.end(),start,
        [](uintptr_t address,const MemoryRegion&region){return address<reinterpret_cast<uintptr_t>(region.data);});
    if(following==regions_.begin())bad("tensor is outside the registered RAM allocations");
    const auto&containing=*(following-1);
    if(start<reinterpret_cast<uintptr_t>(containing.data)||end>end_of(containing.data,containing.bytes))
        bad("tensor is outside the registered RAM allocations");
}
void BorrowedWeights::add_segmented(const std::string& name,BorrowedTensor tensor,std::vector<MemoryRegion> segments){
    if(!valid_name(name)||tensors_.count(name)||tensor.data||segments.empty()||segments.size()>100000)bad("invalid segmented tensor");
    if(tensor.encoding==TensorEncoding::Output16Compact||tensor.encoding==pair145::Encoding)bad("compact descriptors require one real contiguous owner allocation");
    if(tensor.bytes!=expected_bytes(tensor))bad("segmented payload byte count differs");
    size_t total=0;
    for(const auto& s:segments){check_region(s);total=add_size(total,s.bytes);}
    if(total!=tensor.bytes)bad("segmented payload length differs");
    // A single span can remain directly usable by the expert kernels.
    if(segments.size()==1)tensor.data=segments[0].data;
    else segments_.emplace(name,std::move(segments));
    try{tensors_.emplace(name,std::move(tensor));}catch(...){segments_.erase(name);throw;}
}
const uint8_t* BorrowedWeights::read_range(const std::string& name,size_t offset,size_t bytes,std::vector<uint8_t>& scratch)const{
    const auto&t=at(name);
    if(!bytes||bytes>64ull<<20||offset>t.bytes||bytes>t.bytes-offset)bad("invalid or oversized packed range");
    scratch.clear();if(t.data)return t.data+offset;
    const auto& parts=segments_.at(name);size_t begin=0;
    for(const auto&s:parts){
        if(offset>=begin&&offset-begin<s.bytes&&bytes<=s.bytes-(offset-begin))return s.data+(offset-begin);
        begin+=s.bytes;
    }
    scratch.resize(bytes);size_t copied=0;begin=0;
    for(const auto&s:parts){
        const size_t end=begin+s.bytes;
        if(offset<end&&offset+bytes>begin){
            const size_t lo=std::max(begin,offset),hi=std::min(end,offset+bytes);
            std::memcpy(scratch.data()+lo-offset,s.data+lo-begin,hi-lo);copied+=hi-lo;
        }
        begin=end;
    }
    if(copied!=bytes)bad("incomplete segmented range");return scratch.data();
}

const BorrowedTensor& BorrowedWeights::metadata(const std::string& name) const {
    const auto found=tensors_.find(name);
    if(found==tensors_.end())bad("missing tensor");
    return found->second;
}
const BorrowedTensor& BorrowedWeights::at(const std::string& name) const {
    const auto&value=metadata(name);
    if(value.encoding==TensorEncoding::Output16Compact||value.encoding==pair145::Encoding)bad("compact expert is not a raw tensor; explicit compact reader required");
    return value;
}
bool BorrowedWeights::is_output16_compact(const std::string& prefix)const {
    return metadata(prefix+".weight").encoding==TensorEncoding::Output16Compact;
}
CompactProjectionV248 BorrowedWeights::compact_projection(const std::string& prefix,int rows,int columns)const {
    const auto&w=metadata(prefix+".weight");
    if(w.encoding!=TensorEncoding::Output16Compact || w.shape!=std::vector<int64_t>{rows,columns})
        bad("requested compact projection differs from its logical geometry");
    return {w.data,w.bytes,rows,columns,w.bits};
}
std::vector<uint64_t> BorrowedWeights::output16_info_v248()const {
    uint64_t count=0,bytes=0;
    for(const auto&entry:tensors_)if(entry.second.encoding==TensorEncoding::Output16Compact){++count;bytes+=entry.second.bytes;}
    return {1,count,bytes};
}

PackedProjection BorrowedWeights::inspect_projection(const std::string& prefix, int rows, int columns) const {
    if (rows < 1 || rows > 65536 || columns < 32 || columns > 65536 || columns % 32) bad("invalid expert projection geometry");
    const auto& w = metadata(prefix + ".weight");
    if (w.shape.size() != 2 || w.shape[0] != rows) bad("expert weight row count differs");
    if(w.encoding==pair145::Encoding){
        const auto d=pair145::projection(*this,prefix,rows,columns);
        return {nullptr,nullptr,size_t(columns/2),size_t(columns/32),rows,columns,0};
    }
    if(w.encoding==TensorEncoding::Output16Compact){
        if(w.shape[1]!=columns)bad("compact expert logical column count differs");
        // Admission/ranking only: no pointer here can masquerade as raw rows.
        return {nullptr,nullptr,w.bits?own_stride(size_t(columns),w.bits):size_t(columns/2),
                w.bits?0:size_t(columns/32),rows,columns,w.bits};
    }
    if (w.encoding == TensorEncoding::Uniform) {
        if (w.shape[1] != columns || w.protected_native_rows) bad("own projection shape or protected-row overlay requires another reader");
        return {w.data, nullptr, own_stride(size_t(columns), w.bits), 0, rows, columns, w.bits};
    }
    if (w.encoding != TensorEncoding::Mxfp4 || w.shape[1] != columns / 2) bad("expected native MXFP4 or explicit own Uniform weights");
    const auto& scales = at(prefix + ".scale");
    if (scales.encoding != TensorEncoding::E8m0 || scales.shape != std::vector<int64_t>{rows, columns / 32})
        bad("native MXFP4 requires matching separate group32 E8M0 scales");
    return {w.data, scales.data, size_t(columns / 2), size_t(columns / 32), rows, columns, 0};
}
PackedProjection BorrowedWeights::projection(const std::string&prefix,int rows,int columns)const{
    if(is_output16_compact(prefix)||metadata(prefix+".weight").encoding==pair145::Encoding)bad("compact expert has no raw projection; explicit bounded inverse required");
    auto p=inspect_projection(prefix,rows,columns);
    if(!p.weights||(!p.bits&&!p.scales))bad("segmented expert requires explicit bounded contiguous staging");return p;
}
StagedProjection BorrowedWeights::stage_projection(const std::string&prefix,int rows,int columns,size_t budget)const{
    if(!budget||budget>64ull<<20)bad("projection copy budget must be1–64MiB");
    if(is_output16_compact(prefix)||metadata(prefix+".weight").encoding==pair145::Encoding)bad("compact expert requires an explicit bounded inverse staging path");
    StagedProjection result;result.projection=inspect_projection(prefix,rows,columns);
    auto stage=[&](const std::string&name,std::shared_ptr<std::vector<uint8_t>>&storage,const uint8_t*&pointer){
        if(pointer)return;const auto&t=at(name);
        if(t.bytes>budget-result.owned_bytes())bad("segmented projection exceeds copy budget");
        storage=std::make_shared<std::vector<uint8_t>>();pointer=read_range(name,0,t.bytes,*storage);
    };
    stage(prefix+".weight",result.weights,result.projection.weights);
    if(!result.projection.bits)stage(prefix+".scale",result.scales,result.projection.scales);
    return result;
}
} // namespace strata::ds41
