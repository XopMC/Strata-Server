#pragma once

// Source-only bridge from the DS1017 compact E2M1+scale-delta view to the
// legacy CPU Output16-compact-v3 bit0 representation. The caller must retain
// the source OwnerLease for the entire synchronous call.
#include "../compact-scale-expert-rolealign-v4/compact_projection_v4.hpp"

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define DS41_OUTPUT16_V4_EXPORT __attribute__((visibility("default")))
#else
#define DS41_OUTPUT16_V4_EXPORT
#endif

extern "C" {

// This private DSO surface is versioned independently from the old runtime.
DS41_OUTPUT16_V4_EXPORT int ds41_output16_v4_abi_version() noexcept;

// Computes the exact legacy Output16-compact-v3 bit0 allocation size for a
// valid DS1017 V4 projection descriptor.
DS41_OUTPUT16_V4_EXPORT int ds41_output16_v4_required_bytes(
    const ds1017_compact_rolealign_v4::CompactProjectionView* source,
    std::uint64_t* required_bytes) noexcept;

// Synchronously repacks one complete expert matrix into the legacy Output16
// bit0 tile/footer layout. It allocates no memory, starts no threads and never
// changes the source. `expected_generation` must match the view; caller keeps
// the associated OwnerLease alive until return. `destination` must be 2 MiB
// aligned and have at least the size returned by required_bytes().
DS41_OUTPUT16_V4_EXPORT int ds41_output16_v4_pack(
    const ds1017_compact_rolealign_v4::CompactProjectionView* source,
    std::uint64_t expected_generation,
    std::uint8_t* destination,
    std::uint64_t destination_capacity,
    std::uint64_t* written_bytes) noexcept;

DS41_OUTPUT16_V4_EXPORT const char* ds41_output16_v4_error() noexcept;

}

#undef DS41_OUTPUT16_V4_EXPORT
