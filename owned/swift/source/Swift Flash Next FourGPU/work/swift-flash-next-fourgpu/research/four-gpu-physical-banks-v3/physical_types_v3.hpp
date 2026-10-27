#pragma once
#include <cstdint>
namespace four_banks_v3 {
inline constexpr int Layers=48,ExpertsPerLayer=512,Keys=Layers*ExpertsPerLayer,Devices=4;
inline constexpr uint32_t BlobBytes=2764816;
inline constexpr uint64_t SlotStride=2765056;
// Immutable flat table indexed ONLY by global key=layer*512+original expert ID.
// This slot/offset is PHYSICAL. It is never a logical host_res slot value.
struct alignas(8) PhysicalRef {
 uint64_t byte_offset=0;
 uint64_t physical_epoch=0;
 int32_t device=-1;
 int32_t physical_slot=-1;
 uint32_t blob_bytes=0;
 uint32_t key=0;
};
static_assert(sizeof(PhysicalRef)==32&&alignof(PhysicalRef)==8);
// Optional POD for a device planner. All pointers must name already-published
// resources held by the host physical lease through its final GPU consumer.
// refs has exactly Keys elements. The epoch is independent of logical R epoch.
struct PhysicalDeviceView {
 const PhysicalRef*refs=nullptr;
 const unsigned char*base[Devices]{};
 uint64_t bank_bytes[Devices]{};
 uint64_t physical_epoch=0;
};
// Preserve original signed virtual slot values: >=0 selects native math;
// negative selects original CPU semantics. Physical presence changes neither.
struct LogicalDeviceView {
 const int32_t*virtual_slots=nullptr; // Keys entries, original logical R image
 uint64_t logical_epoch=0;
};
}
