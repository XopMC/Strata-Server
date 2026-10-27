#pragma once
// Frozen role-aligned private successor. Weight E2M1 numeric codes remain FOUR bits.
#include "../runtime-nvfp4-reader-v1/nvfp4_projection_v1.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace ds1017_compact_rolealign_v4 {
inline constexpr char codec_id[]="ds1017-e2m1-scale-delta-rolealign-v4";
inline constexpr uint32_t codec_version=4;
// A distinct PRIVATE encoding proposal, never reinterpret Nvfp4=13 as this layout.
inline constexpr uint32_t encoding_id=14;
struct CompactProjectionView {
 const uint8_t*data=nullptr;             // host canonical owner (CPU view)
 uint64_t bytes=0,row_bytes=0;
 uint32_t rows=0,columns=0;
 uint64_t owner_generation=0;
 uint32_t struct_size=64,version=codec_version;
 uint8_t base_e8=0,global_e8=0,scale_delta_bits=0,max_scale_delta=0;
 uint32_t reserved=0;
 uint64_t reserved2=0;
};
static_assert(std::is_standard_layout_v<CompactProjectionView>);
static_assert(std::is_trivially_copyable_v<CompactProjectionView>);
static_assert(sizeof(CompactProjectionView)==64&&alignof(CompactProjectionView)==8);
static_assert(offsetof(CompactProjectionView,data)==0&&offsetof(CompactProjectionView,bytes)==8);
static_assert(offsetof(CompactProjectionView,rows)==24&&offsetof(CompactProjectionView,owner_generation)==32);
static_assert(offsetof(CompactProjectionView,struct_size)==40&&offsetof(CompactProjectionView,base_e8)==48);
static_assert(offsetof(CompactProjectionView,reserved2)==56);
using OwnerLease=ds1017_nvfp4_v1::OwnerLease;
enum class Status {Ok,Version,Geometry,Extent,Generation,Owner,Encoding};
inline constexpr uint64_t code_bytes(uint32_t k){return uint64_t(k)/2;}
inline constexpr uint64_t scale_bytes(uint32_t k,uint8_t bits){return uint64_t(k/32)*bits/8;}
inline constexpr uint64_t row_bytes(uint32_t k,uint8_t bits){const uint64_t raw=code_bytes(k)+scale_bytes(k,bits); const uint64_t alignment=k==2304?(bits==4?32:16):1; return (raw+alignment-1)&~(alignment-1);}
// Row: [K/2 code bytes][(K/32)*bits/8 scale bytes], then role-dependent row padding: GU/UP no padding, DN2 align16, DN4 align32. No block headers.
// Each 16-value E2M1 subgroup: byte j stores code[j] low nibble and code[j+8]
// high nibble, j=0..7. Two subgroups/K32, four/K64: original NV4 code order.
// Scale g is bits at offset g*bits, least-significant field first in its byte.
inline uint8_t scale_delta(const uint8_t*row,uint32_t k,uint8_t bits,uint32_t g){
 const uint32_t bit=g*bits;return uint8_t((row[code_bytes(k)+bit/8]>>(bit%8))&((1u<<bits)-1));
}
inline Status validate_descriptor(const CompactProjectionView&v){
 if(v.struct_size!=sizeof(v)||v.version!=codec_version||v.reserved||v.reserved2)return Status::Version;
 if(!((v.rows==2304&&v.columns==5120)||(v.rows==5120&&v.columns==2304)))return Status::Geometry;
 if(v.scale_delta_bits!=2&&v.scale_delta_bits!=4)return Status::Encoding;
 if(v.max_scale_delta>9||v.max_scale_delta>=(1u<<v.scale_delta_bits)||v.base_e8<112||
    unsigned(v.base_e8)+v.max_scale_delta>127||v.global_e8<113||v.global_e8>119||
    int(v.base_e8)-int(v.global_e8)<-9||int(v.base_e8)+v.max_scale_delta-int(v.global_e8)>8)return Status::Encoding;
 if(v.row_bytes!=row_bytes(v.columns,v.scale_delta_bits)||v.bytes!=uint64_t(v.rows)*v.row_bytes||!v.data)return Status::Extent;
 return Status::Ok;
}
inline Status validate(const CompactProjectionView&v,const OwnerLease&o,uint64_t expected_generation){
 const auto s=validate_descriptor(v);if(s!=Status::Ok)return s;
 if(!expected_generation||v.owner_generation!=expected_generation||o.generation!=expected_generation)return Status::Generation;
 if(!o.allocation||!o.region_base)return Status::Owner;
 const auto p=reinterpret_cast<uintptr_t>(v.data),b=reinterpret_cast<uintptr_t>(o.region_base);
 if(p+v.bytes<p||b+o.region_bytes<b||p<b||p+v.bytes>b+o.region_bytes)return Status::Owner;
 return Status::Ok;
}
// Caller must retain the validated owner and immutable generation. Every packed
// field must be <= max_scale_delta before ANY output is accepted. Numeric code
// nibbles have no invalid encodings. Source E8=base+delta, exact NV4 local power
// =(base+delta)-global_e8. No requantization, rounding or value change.
inline Status validate_row(const CompactProjectionView&v,uint32_t row){
 if(row>=v.rows)return Status::Geometry;
 const auto*p=v.data+uint64_t(row)*v.row_bytes;
 for(uint32_t g=0;g<v.columns/32;++g)if(scale_delta(p,v.columns,v.scale_delta_bits,g)>v.max_scale_delta)return Status::Encoding;
 return Status::Ok;
}
} // namespace ds1017_compact_rolealign_v4
