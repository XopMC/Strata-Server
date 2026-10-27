#include "output16_v4_packer_v1.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
using View = ds1017_compact_rolealign_v4::CompactProjectionView;
using Status = ds1017_compact_rolealign_v4::Status;

constexpr std::uint64_t kOutputAlignment = std::uint64_t{2} * 1024 * 1024;
constexpr std::uint64_t kTileBytes = 256 + 32;
constexpr std::uint64_t kCodeBytesPerK32 = 16;

std::uint8_t compact_code(const std::uint8_t* row, std::uint32_t column) noexcept {
    const std::uint32_t byte = (column / 16) * 8 + column % 8;
    const std::uint32_t shift = 4 * ((column % 16) / 8);
    return static_cast<std::uint8_t>((row[byte] >> shift) & 0x0f);
}

enum Error : int {
    kOk = 0,
    kNullArgument = 1,
    kInvalidDescriptor = 2,
    kGenerationMismatch = 3,
    kSizeOverflow = 4,
    kDestinationTooSmall = 5,
    kDestinationAlignment = 6,
    kAddressOverflow = 7,
    kSourceDestinationOverlap = 8,
    kInvalidScaleDelta = 9,
};

thread_local const char* g_error = "ok";

int fail(Error code, const char* message) noexcept {
    g_error = message;
    return static_cast<int>(code);
}

bool output_bytes(const View& view, std::uint64_t& result) noexcept {
    const std::uint64_t tiles = view.rows / 16;
    const std::uint64_t groups = view.columns / 32;
    if (groups != 0 && tiles > std::numeric_limits<std::uint64_t>::max() / groups) {
        return false;
    }
    const std::uint64_t tile_count = tiles * groups;
    if (tile_count > std::numeric_limits<std::uint64_t>::max() / kTileBytes) {
        return false;
    }
    result = tile_count * kTileBytes;
    return true;
}

bool address_end(std::uintptr_t begin, std::uint64_t bytes, std::uintptr_t& end) noexcept {
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    end = begin + static_cast<std::uintptr_t>(bytes);
    return true;
}

int validate_source(const View* source, std::uint64_t expected_generation,
                    std::uint64_t& required) noexcept {
    if (source == nullptr) return fail(kNullArgument, "null source view");
    if (ds1017_compact_rolealign_v4::validate_descriptor(*source) != Status::Ok) {
        return fail(kInvalidDescriptor, "invalid DS1017 role-aligned V4 descriptor");
    }
    if (expected_generation == 0 || source->owner_generation != expected_generation) {
        return fail(kGenerationMismatch, "source generation differs from caller lease generation");
    }
    if (!output_bytes(*source, required)) {
        return fail(kSizeOverflow, "Output16 byte count overflows uint64");
    }
    g_error = "ok";
    return kOk;
}

}  // namespace

extern "C" int ds41_output16_v4_abi_version() noexcept {
    return 1;
}

extern "C" int ds41_output16_v4_required_bytes(const View* source,
                                                std::uint64_t* required_bytes) noexcept {
    if (required_bytes == nullptr) return fail(kNullArgument, "null required_bytes output");
    *required_bytes = 0;
    std::uint64_t required = 0;
    const int status = validate_source(source, source ? source->owner_generation : 0, required);
    if (status != kOk) return status;
    *required_bytes = required;
    g_error = "ok";
    return kOk;
}

extern "C" int ds41_output16_v4_pack(const View* source,
                                      std::uint64_t expected_generation,
                                      std::uint8_t* destination,
                                      std::uint64_t destination_capacity,
                                      std::uint64_t* written_bytes) noexcept {
    if (written_bytes == nullptr) return fail(kNullArgument, "null written_bytes output");
    *written_bytes = 0;
    if (destination == nullptr) return fail(kNullArgument, "null destination");

    std::uint64_t required = 0;
    const int source_status = validate_source(source, expected_generation, required);
    if (source_status != kOk) return source_status;
    if (destination_capacity < required) {
        return fail(kDestinationTooSmall, "destination capacity is smaller than the full matrix");
    }
    const auto destination_address = reinterpret_cast<std::uintptr_t>(destination);
    if (destination_address % kOutputAlignment != 0) {
        return fail(kDestinationAlignment, "destination is not 2 MiB aligned");
    }

    std::uintptr_t source_end = 0;
    std::uintptr_t destination_end = 0;
    const auto source_address = reinterpret_cast<std::uintptr_t>(source->data);
    if (!address_end(source_address, source->bytes, source_end) ||
        !address_end(destination_address, required, destination_end)) {
        return fail(kAddressOverflow, "source or destination address range overflows uintptr_t");
    }
    if (destination_address < source_end && source_address < destination_end) {
        return fail(kSourceDestinationOverlap, "source and destination ranges overlap");
    }

    // Validate every scale field before accepting any output bytes. The caller
    // discards the destination on any failure; this scan also prevents partial
    // mutation for a malformed late-row descriptor.
    const std::uint64_t groups = source->columns / 32;
    for (std::uint32_t row_index = 0; row_index < source->rows; ++row_index) {
        const std::uint8_t* row = source->data + std::uint64_t(row_index) * source->row_bytes;
        for (std::uint32_t group = 0; group < groups; ++group) {
            const std::uint8_t delta = ds1017_compact_rolealign_v4::scale_delta(
                row, source->columns, source->scale_delta_bits, group);
            if (delta > source->max_scale_delta ||
                unsigned(source->base_e8) + unsigned(delta) > 127u) {
                return fail(kInvalidScaleDelta, "scale delta is outside the admitted V4 range");
            }
        }
    }

    const std::uint64_t tiles = source->rows / 16;
    for (std::uint64_t tile = 0; tile < tiles; ++tile) {
        for (std::uint32_t group = 0; group < groups; ++group) {
            std::uint8_t* block = destination + (tile * groups + group) * kTileBytes;
            std::uint8_t* code_block = block;
            std::uint8_t* scale_footer = block + kCodeBytesPerK32 * 16;

            for (std::uint32_t row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
                const std::uint32_t row_index = static_cast<std::uint32_t>(tile * 16 + row_in_tile);
                const std::uint8_t* row = source->data + std::uint64_t(row_index) * source->row_bytes;
                const std::uint8_t delta = ds1017_compact_rolealign_v4::scale_delta(
                    row, source->columns, source->scale_delta_bits, group);
                const std::uint8_t e8m0_code = static_cast<std::uint8_t>(source->base_e8 + delta);
                scale_footer[row_in_tile * 2] = e8m0_code;
                scale_footer[row_in_tile * 2 + 1] = 0;
            }

            // The V4 source stores each K16 subgroup as eight low nibbles
            // followed by eight high nibbles. The old MXFP4/Output16 source
            // stores adjacent logical codes in each byte. Convert that
            // nibble order directly while constructing the Output16 tile.
            // This is lossless E2M1 code movement, not quantization.
            for (std::uint32_t subgroup = 0; subgroup < 8; ++subgroup) {
                for (std::uint32_t row_pair = 0; row_pair < 8; ++row_pair) {
                    const std::uint32_t row0_index = static_cast<std::uint32_t>(tile * 16 + row_pair * 2);
                    const std::uint32_t row1_index = row0_index + 1;
                    const std::uint8_t* row0 = source->data + std::uint64_t(row0_index) * source->row_bytes;
                    const std::uint8_t* row1 = source->data + std::uint64_t(row1_index) * source->row_bytes;
                    const std::uint64_t out_offset = std::uint64_t(subgroup) * 32 + row_pair * 4;
                    const std::uint32_t logical_base = group * 32 + subgroup * 4;
                    const auto pair_bytes = [logical_base](const std::uint8_t* row) noexcept {
                        const std::uint8_t c0 = compact_code(row, logical_base);
                        const std::uint8_t c1 = compact_code(row, logical_base + 1);
                        const std::uint8_t c2 = compact_code(row, logical_base + 2);
                        const std::uint8_t c3 = compact_code(row, logical_base + 3);
                        return std::array<std::uint8_t, 2>{
                            static_cast<std::uint8_t>(c0 | (c1 << 4)),
                            static_cast<std::uint8_t>(c2 | (c3 << 4)),
                        };
                    };
                    const auto old_row0 = pair_bytes(row0);
                    const auto old_row1 = pair_bytes(row1);
                    code_block[out_offset] = old_row0[0];
                    code_block[out_offset + 1] = old_row0[1];
                    code_block[out_offset + 2] = old_row1[0];
                    code_block[out_offset + 3] = old_row1[1];
                }
            }
        }
    }

    *written_bytes = required;
    g_error = "ok";
    return kOk;
}

extern "C" const char* ds41_output16_v4_error() noexcept {
    return g_error;
}
