#pragma once
// SOURCE PROPOSAL ONLY: no compiler/device execution by the researcher.
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace swift_tail_r027 {
struct TileSpan { uint32_t offset = 0, count = 0; };
static_assert(sizeof(TileSpan) == 8);
struct Geometry {
    int J = 0, groups = 0, ntx = 0, tiles = 0, grid = 0;
    int k_blocks = 40, blocks_per_iter = 4;
    int owner = -1;
    int64_t max_rows = 0, total_rows = 0, ld_dst = 0;
};
struct HostPlan {
    Geometry geometry;
    std::vector<TileSpan> spans;
    std::vector<int32_t> previous_blocks;
};
inline int64_t range_edge(const Geometry& g, int b) {
    const int64_t raw = int64_t(b) * (int64_t(g.tiles) * g.k_blocks) / g.grid;
    return raw - ((raw % g.k_blocks) % g.blocks_per_iter);
}
// Reproduce original fixup's PREVIOUS-BLOCK traversal, not a reassociated sum.
inline HostPlan make_plan(const Geometry& g) {
    if (g.J < 8 || g.J > 128 || g.groups < 1 || g.groups > 16 || g.ntx < 1 ||
        g.tiles != int64_t(10) * g.groups * g.ntx || g.grid < 1 ||
        g.k_blocks != 40 || g.blocks_per_iter != 4 ||
        int64_t(g.tiles) * g.k_blocks >= (int64_t(1) << 30))
        throw std::runtime_error("R027 invalid original native40 StreamK geometry");
    HostPlan p;
    p.geometry = g;
    p.spans.resize(g.tiles);
    std::vector<uint8_t> terminal(g.tiles, 0);
    for (int b0 = 0; b0 < g.grid; ++b0) {
        const int64_t start = range_edge(g, b0), stop = range_edge(g, b0 + 1);
        if (start == stop || start % g.k_blocks == 0 ||
            (start / g.k_blocks == stop / g.k_blocks && stop % g.k_blocks != 0))
            continue;
        const int tile = int(start / g.k_blocks);
        if (tile < 0 || tile >= g.tiles || terminal[tile]++)
            throw std::runtime_error("R027 nonunique terminal fixup owner");
        TileSpan& span = p.spans[tile];
        span.offset = uint32_t(p.previous_blocks.size());
        int b = b0 - 1;
        int64_t end = start;
        for (;;) {
            if (b < 0) throw std::runtime_error("R027 missing original preceding partial");
            const int64_t begin = range_edge(g, b);
            if (begin == end) { --b; end = begin; continue; }
            p.previous_blocks.push_back(b);
            ++span.count;
            if (begin % g.k_blocks == 0 || begin / g.k_blocks < start / g.k_blocks)
                break;
            --b;
            end = begin;
        }
    }
    if (p.previous_blocks.size() > size_t(g.grid))
        throw std::runtime_error("R027 one incomplete tail per CUDAblock violated");
    return p;
}
} // namespace swift_tail_r027
