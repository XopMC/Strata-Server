// Compile with the EXACT original MMQ flags, including -use_fast_math.
// SOURCE PROPOSAL ONLY. No builds or GPU validation performed by researcher.
#include "common.cuh"
#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#include "raw_gu_r027.cuh"
#include "tail_api.hpp"
#include <stdexcept>

namespace swift_tail_r027 {
namespace {
void need(bool x, const char* s) { if (!x) throw std::runtime_error(s); }
void product(const strata::prefill::mmq::Product& p) {
    need(p.type == 40 && p.n >= 1 && p.n <= 16 && p.w_rows == 1280 && p.w_cols == 2560 &&
         p.expert_bytes == 1843200 && (p.ld_dst == 1280 || p.ld_dst == 2560) &&
         p.total_rows > 0 && p.total_rows <= 32768LL * 10 &&
         p.max_rows > 0 && p.max_rows <= p.total_rows &&
         p.w && p.xq && p.ids && p.bounds && p.dst && !p.tab && !p.y_scale,
         "R027 ordinary native40 W4A8 GU product required");
}
template<int J> void prepare_shared(size_t bytes) {
    CUDA_SET_SHARED_MEMORY_LIMIT((raw_tail_r027_mul_mat_q<GGML_TYPE_NVFP4, J, false>), bytes);
}
template<int J> void raw(const strata::prefill::mmq::Product& p, const Shape& s, const Binding& b) {
    constexpr int warp = 32, nwarps = 8;
    raw_tail_r027_mul_mat_q<GGML_TYPE_NVFP4, J, false>
        <<<dim3(s.geometry.grid), dim3(warp, nwarps), s.shared_bytes, b.stream>>>
        ((const char*)p.w, (const int*)p.xq, p.ids, p.bounds, p.dst, b.partial, nullptr,
         init_fastdiv_values(40), 1280, p.total_rows, 40, p.total_rows, p.ld_dst,
         init_fastdiv_values(1), init_fastdiv_values(p.n), p.expert_bytes / 36, 0, 0,
         init_fastdiv_values(1), init_fastdiv_values(1), 0, 0, 0,
         init_fastdiv_values(s.geometry.ntx));
}
#define R027_J_CASES(FN) \
    case 8: FN(8); break; case 16: FN(16); break; case 24: FN(24); break; \
    case 32: FN(32); break; case 40: FN(40); break; case 48: FN(48); break; \
    case 64: FN(64); break; case 80: FN(80); break; case 96: FN(96); break; \
    case 112: FN(112); break; case 128: FN(128); break
} // namespace

Shape prepare_shape(const strata::prefill::mmq::Product& p) {
    product(p);
    const int device = ggml_cuda_get_device();
    need(device >= 0 && device < 3, "R027 only current admitted three-device scope");
    const auto& d = ggml_cuda_info().devices[device];
    need(d.cc == 890 && d.nsm > 0, "R027 original SM89 configuration required");
    need(strata::prefill::mmq::nvfp4_mode() == strata::prefill::mmq::Nvfp4Mode::W4A8 &&
         !strata::prefill::mmq::fp4_activations(40), "R027 no new activation precision");
    int best = 0, tiles = INT_MAX;
    for (int J = 8; J <= 128 && tiles > 1; J += 8) {
        const auto c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, false, d.cc);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, d.cc) > d.smpbo) continue;
        const int n = (p.max_rows + c.J - 1) / c.J;
        if (n < tiles) { best = J; tiles = n; }
    }
    need(best != 0, "R027 original J policy has no admissible specialization");
    const auto c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, best, false, d.cc);
    need(c.I == 128 && c.nthreads == 256 && c.K_vram == 256 && c.stream_k,
         "R027 exact original native40 I/threads/K/StreamK required");
    Shape s;
    auto& g = s.geometry;
    g.J = best; g.groups = p.n; g.ntx = tiles; g.tiles = tiles * 10 * p.n;
    const int waves = (g.tiles + d.nsm - 1) / d.nsm;
    g.grid = 100 * g.tiles / (d.nsm * waves) >= 90 ? g.tiles : d.nsm;
    g.owner = device; g.max_rows = p.max_rows; g.total_rows = p.total_rows; g.ld_dst = p.ld_dst;
    need(int64_t(g.tiles) * 40 < (int64_t(1) << 30), "R027 original kbc bound");
    s.shared_bytes = mmq_get_nbytes_shared(c, d.cc);
    s.partial_bytes = g.tiles % g.grid ? size_t(g.grid) * best * 128 * sizeof(float) : 0;
#define R027_PREP(J) prepare_shared<J>(s.shared_bytes)
    switch (best) { R027_J_CASES(R027_PREP); default: throw std::runtime_error("R027 J mismatch"); }
#undef R027_PREP
    return s;
}

void enqueue_raw(const strata::prefill::mmq::Product& p, const Shape& s, const Binding& b) {
    product(p);
    const auto& g = s.geometry;
    need(b.owner == g.owner && b.owner >= 0 && b.owner < 3 && b.stream && b.epoch &&
         b.identity_ids == p.ids && b.identity_ids &&
         g.groups == p.n && g.max_rows == p.max_rows && g.total_rows == p.total_rows &&
         g.ld_dst == p.ld_dst && g.ntx == (p.max_rows + g.J - 1) / g.J &&
         g.tiles == g.ntx * 10 * p.n && b.partial_capacity >= s.partial_bytes &&
         (!s.partial_bytes || b.partial), "R027 raw owning product/lease binding mismatch");
    // Caller already owns this device/stream and has validated every allocation interval.
    // This function performs no allocator call, context switch, old dispatcher or old fixup.
#define R027_RAW(J) raw<J>(p, s, b)
    switch (g.J) { R027_J_CASES(R027_RAW); default: throw std::runtime_error("R027 unprepared J"); }
#undef R027_RAW
    CUDA_CHECK(cudaGetLastError());
}
#undef R027_J_CASES
} // namespace swift_tail_r027

// V16 production HOST-only preparation. The original V12 prefix above, including
// prepare_shape/enqueue_raw, remains unchanged. No raw numerical kernel is edited.
namespace swift_tail_r027 {
LaunchProfile prepare_launch_profile() {
    const int device = ggml_cuda_get_device();
    need(device >= 0 && device < 3, "R027 V16 current admitted owner required");
    const auto& d = ggml_cuda_info().devices[device];
    need(d.cc == 890 && d.nsm > 0 &&
         strata::prefill::mmq::nvfp4_mode() == strata::prefill::mmq::Nvfp4Mode::W4A8 &&
         !strata::prefill::mmq::fp4_activations(40), "R027 V16 original SM89/W4A8 required");
    LaunchProfile profile; profile.owner=device; profile.sm_count=d.nsm;
    for (int J=8; J<=128; J+=8) {
        const auto c=ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4,J,false,d.cc);
        if(c.type==GGML_TYPE_COUNT) continue;
        const size_t bytes=mmq_get_nbytes_shared(c,d.cc);
        if(bytes>d.smpbo) continue;
        need(c.J==J && c.I==128 && c.nthreads==256 && c.K_vram==256 && c.stream_k,
             "R027 V16 original config differs");
#define V16_PREP(JJ) case JJ: prepare_shared<JJ>(bytes); break
        switch(J) {
            V16_PREP(8);V16_PREP(16);V16_PREP(24);V16_PREP(32);V16_PREP(40);V16_PREP(48);
            V16_PREP(64);V16_PREP(80);V16_PREP(96);V16_PREP(112);V16_PREP(128);
            default: throw std::runtime_error("R027 V16 unexpected original J specialization");
        }
#undef V16_PREP
        need(bytes>0,"R027 V16 invalid shared bytes");
        profile.shared[J/8-1]=bytes;profile.largest_J=J;
    }
    need(profile.largest_J>0,"R027 V16 no original launch specialization");
    return profile;
}
Shape describe_shape(const strata::prefill::mmq::Product& p,const LaunchProfile& profile) {
    product(p);
    need(profile.owner>=0&&profile.owner<3&&profile.sm_count>0&&profile.largest_J>0,
         "R027 V16 unprepared launch profile");
    int best=0,tiles=INT_MAX;
    for(int J=8;J<=128&&tiles>1;J+=8) {
        if(!profile.shared[J/8-1])continue;
        const int n=(p.max_rows+J-1)/J;
        if(n<tiles){best=J;tiles=n;}
    }
    need(best!=0,"R027 V16 original J selection failed");
    Shape s;auto&g=s.geometry;
    g.J=best;g.groups=p.n;g.ntx=tiles;g.tiles=tiles*10*p.n;
    const int waves=(g.tiles+profile.sm_count-1)/profile.sm_count;
    g.grid=100*g.tiles/(profile.sm_count*waves)>=90?g.tiles:profile.sm_count;
    g.owner=profile.owner;g.max_rows=p.max_rows;g.total_rows=p.total_rows;g.ld_dst=p.ld_dst;
    need(int64_t(g.tiles)*40<(int64_t(1)<<30),"R027 V16 original kbc bound");
    s.shared_bytes=profile.shared[best/8-1];
    s.partial_bytes=g.tiles%g.grid?size_t(g.grid)*best*128*sizeof(float):0;
    return s;
}
} // namespace swift_tail_r027
