#pragma once

// Private DS1017 source-only reader ABI. This is deliberately separate from
// the production DS41 ABI1 headers: never pass Nvfp4=13 to an older DSO.
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ds1017_nvfp4_v1 {

inline constexpr int kEncodingNvfp4 = 13;
inline constexpr std::uint32_t kAdapterAbiVersion = 2;
inline constexpr std::uint32_t kCapabilitySchemaVersion = 2;
inline constexpr std::uint64_t kNvfp4CapabilityBit = std::uint64_t{1} << kEncodingNvfp4;
inline constexpr char kCodecId[] = "ds1017-nvfp4-g64-e4m3x4-v1";
inline constexpr char kCapabilitySchema[] = "ds41-native-runtime-capabilities/2";

// Preserve the exact R6 wire-number ordering and append Nvfp4 after
// Output16Compact=12. ABI1 consumers still refuse the new value via handshake.
enum class TensorEncoding : int {
    Bf16 = 0, F32 = 1, F16 = 2, F8e4m3 = 3, E8m0 = 4,
    I8 = 5, U8 = 6, I16 = 7, I32 = 8, I64 = 9,
    Mxfp4 = 10, Uniform = 11, Output16Compact = 12, Nvfp4 = 13,
};

// Fixed-width DSO handshake record. `struct_size` lets a future consumer
// refuse a shorter/older record instead of guessing what an ABI1 library does.
struct RuntimeCapabilitiesV2 {
    std::uint32_t struct_size;
    std::uint32_t abi_version;
    std::uint64_t supported_encoding_mask;
    char nvfp4_codec_id[48];
};

// Canonical CPU-leaf view agreed with the DS1017 direct-NVFP4 leaf owner.
// `data` names one contiguous expert matrix in the verified bundle mapping.
// Owner retention is a separate RAII object, never encoded as a raw pointer.
struct ProjectionView {
    const std::uint8_t* data;
    std::size_t bytes;
    std::size_t row_bytes;
    int rows;
    int columns;
    std::uint8_t global_scale_code;
    std::int16_t global_power2_exponent;
    std::uint64_t owner_generation;
};

struct OwnerLease {
    std::shared_ptr<const void> allocation;
    const std::uint8_t* region_base;
    std::size_t region_bytes;
    std::uint64_t generation;
};

enum class Status : int {
    Ok = 0,
    NullOutput,
    OutputTooSmall,
    CapabilityAbiMismatch,
    CapabilityMissingNvfp4,
    CapabilityCodecMismatch,
    MissingOwner,
    OwnerGenerationMismatch,
    OwnerRangeMismatch,
    NullData,
    InvalidGeometry,
    SizeOverflow,
    ByteCountMismatch,
    InvalidGlobalScale,
    AddressOverflow,
};

bool supports_nvfp4(const RuntimeCapabilitiesV2& capabilities) noexcept;
Status validate_projection(const ProjectionView& view, const OwnerLease& owner) noexcept;

}  // namespace ds1017_nvfp4_v1

extern "C" {

// A legacy DSO without this exact symbol is incapable until separately
// proven otherwise. `output_bytes` must fit the complete v2 structure.
int ds1017_native_runtime_capabilities_v2(
    ds1017_nvfp4_v1::RuntimeCapabilitiesV2* output,
    std::size_t output_bytes) noexcept;

}
