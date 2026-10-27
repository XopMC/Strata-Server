// Inside strata::ds41, after the original request/copy statistics definitions.
static int reset_request_locked528() {
        if (!serving516::enabled()) return 1;
        if (serving516::engines.load() || serving516::active.load() ||
            wavefront325::active.load() || prefill317_detail::stats[5].load()) return 2;
        if (pass554::stats[10].load() || pass554::stats[11].load()) return 3;
        if (pass554::stats[12].load() || pass554::stats[13].load()) return 4;
        if (pass558::stats[10].load() || pass558::stats[11].load()) return 3;
        if (pass558::stats[12].load() || pass558::stats[13].load()) return 4;
        if (fast534::words[18].load()) return 3;
        if (fast534::words[21].load()) return 4;
#if DS516_CONTEXT == 262144
        if (geometry335::stats[12].load()) return 3;
        if (geometry335::stats[11].load() || serial_copy_counts351[11].load()) return 4;
#else
        if (early337::stats[13].load() || early337::stats[14].load() ||
            early337::stats[16].load() || copy331::stats[13].load() ||
            copy331::stats[14].load()) return 3;
        if (early337::stats[12].load() || early337::stats[15].load() ||
            copy331::stats[10].load() || copy331::stats[11].load() ||
            copy331::stats[12].load()) return 4;
#endif
        // Preserve failures: a damaged request requires a fresh CUDA child.
        if (prefill317_detail::stats[4].load() || wavefront325::stats[4].load() ||
            wavefront325::stats[27].load()) return 4;
        for (auto& x : prefill317_detail::stats) x=0;
        for (auto& x : wavefront325::stats) x=0;
        for (auto& x : stage_copy_counts325) x=0;
#if DS516_CONTEXT == 262144
        for (auto& x : geometry335::stats) x=0;
        for (auto& x : serial_copy_counts351) x=0;
#else
        for (auto& x : early337::stats) x=0;
        for (auto& x : copy331::stats) x=0;
#endif
        serving516::mode=0; serving516::rows=0; serving516::pos=0;
        serving516::context=DS516_CONTEXT; serving516::cap=0;
        serving516::B=0; serving516::E=0; ++serving516::epoch;
        return 0;
}
extern "C" int ds516_prepare_request_v1() noexcept {
    try {std::lock_guard<std::mutex> lock(serving516::registry);
        const int rc=reset_request_locked528();if(!rc)serving516::prepared528={};return rc;
    }catch(...){return 5;}
}
extern "C" int ds528_prepare_request_v1(int context,int N,int pos,uint32_t flags) noexcept {
    try {
        const auto candidate=adaptive528::make(context,N,pos,flags);
        if(context!=DS516_CONTEXT)return 6;
        std::lock_guard<std::mutex> lock(serving516::registry);
        const int rc=reset_request_locked528();if(rc)return rc;
        auto plan=candidate;plan.epoch=serving516::epoch.load();serving516::prepared528=plan;
        return 0;
    }catch(const std::invalid_argument&){return 6;}catch(...){return 5;}
}
extern "C" int ds528_prefill_plan_info(uint64_t*out,int n) noexcept {
    if(!out||n!=16)return 1;
    try {std::lock_guard<std::mutex> lock(serving516::registry);
        const auto&p=serving516::prepared528;
        const uint64_t v[16]={528,p.epoch,uint64_t(p.armed),uint64_t(p.mode),uint64_t(p.g.context),uint64_t(p.N),uint64_t(p.pos),
            uint64_t((p.mode==2||p.mode==3)?p.g.split:0),uint64_t(p.g.cap),uint64_t(p.g.B),uint64_t(p.g.E),
            uint64_t(adaptive528::first_rows(p)),uint64_t(adaptive528::second_rows(p)),uint64_t((p.mode==2||p.mode==3)?p.N%2:0),
            serving516::engines.load(),serving516::active.load()};
        for(int i=0;i<16;++i)out[i]=v[i];return 0;
    }catch(...){return 2;}
}
extern "C" int ds516_serving_prefill_info(uint64_t* out,int n) noexcept {
    if (!out || n!=12) return 1;
    try {
        const uint64_t v[12]={516,serving516::epoch.load(),uint64_t(serving516::enabled()),
            serving516::mode.load(),serving516::rows.load(),serving516::pos.load(),
            serving516::context.load(),serving516::cap.load(),serving516::B.load(),
            serving516::E.load(),serving516::active.load(),serving516::engines.load()};
        for(int i=0;i<12;++i)out[i]=v[i];
        return 0;
    } catch (...) { return 2; }
}

extern "C" int ds534_fp8_fastpath_info(uint64_t*out,int n) noexcept {
 if(!out||n!=32)return 1;
 for(int i=0;i<32;++i)out[i]=strata::ds41::fast534::words[i].load();
 out[0]=534;out[1]=32;return 0;
}
