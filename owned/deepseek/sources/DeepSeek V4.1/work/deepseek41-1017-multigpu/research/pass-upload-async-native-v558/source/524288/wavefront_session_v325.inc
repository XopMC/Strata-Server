// Inside Engine::Impl. Literal317 layer bodies/252 math remain elsewhere.
    struct WaveFrame325 {
        HeadlessPrefixV320 prefix;int begin=0,end=0,carry=0,carry_source=-1;uint64_t carry_bytes=0;
        uint64_t bytes()const{return prefix.window_bytes+prefix.compressed_bytes+prefix.index_bytes+2*carry_bytes;}
    };
    using WaveCredits325=wavefront325::Credits<WaveFrame325>;
    struct WaveResult325 {
        std::vector<uint16_t>final_row,tail;int tail_first=-1;
        HeadlessFrequencyV320 frequency;
        std::array<uint64_t,16>output{};std::array<uint64_t,4>suffix{},compact{};
        std::array<uint64_t,3>packed{},memory{};PrefillTiming timing;
    };
    struct WaveShared325 {
        WaveCredits325 credits;
        std::array<std::shared_ptr<HeadlessPinnedPrefixV320>,2>forward,reverse;
        std::mutex mutex;std::condition_variable changed;
        bool ready0=false,ready1=false,done1=false,failed=false;
        std::exception_ptr error;WaveResult325 result;
        void fail(std::exception_ptr e)noexcept{
            {std::lock_guard<std::mutex>g(mutex);if(!error)error=e;failed=true;changed.notify_all();}
            credits.fail(e);
        }
        void wait(bool WaveShared325::*field){std::unique_lock<std::mutex>g(mutex);changed.wait(g,[&]{return this->*field||failed;});if(failed)std::rethrow_exception(error);}
        void set(bool WaveShared325::*field){std::lock_guard<std::mutex>g(mutex);this->*field=true;changed.notify_all();}
    };
    struct WaveTransport325 {
        int device=-1;std::thread::id issuer;
        cudaStream_t outbound=nullptr,inbound=nullptr;
        cudaEvent_t sent=nullptr,received=nullptr;
        explicit WaveTransport325(int expected):issuer(std::this_thread::get_id()){
            ck(cudaGetDevice(&device),"325 transport current device");if(device!=expected)throw std::logic_error("325 rank transport device");
            cudaStreamCaptureStatus capture{};ck(cudaStreamIsCapturing(0,&capture),"325 ordinary prefill capture guard");if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("325 refuses captured coordinator");
            try{ck(cudaStreamCreateWithFlags(&outbound,cudaStreamNonBlocking),"325 export stream");ck(cudaStreamCreateWithFlags(&inbound,cudaStreamNonBlocking),"325 import stream");ck(cudaEventCreateWithFlags(&sent,cudaEventDisableTiming),"325 export event");ck(cudaEventCreateWithFlags(&received,cudaEventDisableTiming),"325 import event");}
            catch(...){release();throw;}
        }
        void current()const{int d=-1;ck(cudaGetDevice(&d),"325 transport issuer device");if(d!=device||issuer!=std::this_thread::get_id())throw std::logic_error("325 transport changed issuer/device");}
        bool drain()noexcept{bool ok=true;if(outbound&&cudaStreamSynchronize(outbound)!=cudaSuccess)ok=false;if(inbound&&cudaStreamSynchronize(inbound)!=cudaSuccess)ok=false;if(cudaStreamSynchronize(0)!=cudaSuccess)ok=false;return ok;}
        void release()noexcept{
            // An unprovable CUDA drain cannot safely release pinned/model DMA
            // storage. CPU/protocol exceptions take the successful drain path.
            if(!drain())std::terminate();
            if(sent)cudaEventDestroy(sent);if(received)cudaEventDestroy(received);
            if(outbound)cudaStreamDestroy(outbound);if(inbound)cudaStreamDestroy(inbound);
            sent=received=nullptr;outbound=inbound=nullptr;
        }
        ~WaveTransport325(){release();}
    };
    static int wave_cpu325(const char*key){
        const char*v=std::getenv(key);if(!v||!*v)throw std::invalid_argument(std::string("325 explicit CPU required: ")+key);
        char*end=nullptr;long n=std::strtol(v,&end,10);if(*end||n<0||n>=CPU_SETSIZE)throw std::invalid_argument("325 CPU range");return int(n);
    }
    static void wave_pin325(int cpu){cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask))throw std::runtime_error("325 issuer CPU affinity rejected");cpu_set_t actual;CPU_ZERO(&actual);if(pthread_getaffinity_np(pthread_self(),sizeof(actual),&actual)||CPU_COUNT(&actual)!=1||!CPU_ISSET(cpu,&actual))throw std::runtime_error("325 actual issuer affinity differs");}
    static void wave_reserve325(int rank){size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"325 live reserve");wavefront325::minimum(32+rank,uint64_t(free));if(free<(size_t(512)<<20))throw std::runtime_error("325 actual device reserve below512MiB");}
    static HeadlessBoundaryV320 wave_boundary325(const PrefillBoundary317&b){
        HeadlessBoundaryV320 v;v.epoch=b.epoch;v.event=reinterpret_cast<uint64_t>(b.ready);v.layer=b.layer;v.next_layer=b.next_layer;v.c0=b.c0;v.rows=b.S;v.position=b.p0;v.absolute_end=b.absolute_end;v.ratio=b.ratio;v.source_layer=b.source_layer;v.compressed_rows=b.compressed_rows;v.carry=b.carry;
        v.window=reinterpret_cast<const uint16_t*>(b.window);v.compressed=reinterpret_cast<const uint16_t*>(b.comp);v.index_keys=reinterpret_cast<const uint16_t*>(b.index_keys);v.kv_state=b.kv_state;v.score_state=b.score_state;v.window_bytes=b.window_bytes;v.compressed_bytes=b.comp_bytes;v.index_bytes=b.index_bytes;v.carry_bytes=b.carry_bytes;return v;
    }
    static WaveFrame325 wave_export325(const HeadlessBoundaryV320&b,int layer,int begin,int end,
            const std::shared_ptr<HeadlessPinnedPrefixV320>&storage,WaveTransport325&t,const geometry335::Fixed&fixed335){
        t.current();if(layer<0||layer>=40)throw std::invalid_argument("325 export layer bounds");const int expected_ratio=kCompressRatio[layer];
        if(layer<0||layer>=40||b.version!=320||!b.epoch||b.layer!=layer||b.next_layer!=layer+1||b.absolute_end!=end||b.ratio!=expected_ratio||b.window_bytes!=uint64_t(kWindow)*kHeadDim*2||!b.window||!b.event||begin<0||end<=begin||end>fixed335.context||b.carry!=(expected_ratio>1?end%expected_ratio:0)||b.carry_bytes!=uint64_t(b.carry)*kHeadDim*sizeof(float)){wavefront325::add(27);throw std::invalid_argument("325 exact layer/end/window/carry certificate");}
        const bool source=is_kv_source(layer);const uint64_t first=source?uint64_t(begin/expected_ratio):0,rows=source?uint64_t((end-begin)/expected_ratio):0;
        if(source&&(begin%expected_ratio||b.source_layer!=layer||!b.compressed||!b.index_keys||b.compressed_rows!=end/expected_ratio||b.compressed_bytes!=uint64_t(end/expected_ratio)*kHeadDim*2||b.index_bytes!=uint64_t(end/expected_ratio)*kIndexDim*2))throw std::invalid_argument("325 source slice extent");
        WaveFrame325 out;out.begin=begin;out.end=end;out.carry=b.carry;
        if(source&&expected_ratio==2&&b.carry){
            if(begin==0||!b.kv_state||!b.score_state)throw std::invalid_argument("528 reverse source carry ownership");
            out.carry_source=layer;out.carry_bytes=b.carry_bytes;
        }
        auto&v=out.prefix;
        v.producer_epoch=b.epoch;v.layer=layer;v.prefix_end=end;v.source_layer=source?layer:-1;v.window_bytes=131072;
        v.compressed_offset=source?v.window_bytes:0;v.compressed_bytes=rows*kHeadDim*2;
        v.index_offset=source?v.window_bytes+v.compressed_bytes:0;v.index_bytes=rows*kIndexDim*2;v.storage=storage;
        const uint64_t payload=out.bytes();
        if(!storage||storage->capacity()<payload)throw std::invalid_argument("325 typed owned frame too small");
        auto*host=static_cast<uint8_t*>(storage->data());
        ck(cudaStreamWaitEvent(t.outbound,reinterpret_cast<cudaEvent_t>(b.event),0),"325 export waits original layer event");
        ck(cudaMemcpyAsync(host,b.window,size_t(v.window_bytes),cudaMemcpyDeviceToHost,t.outbound),"325 window D2H");
        if(source){ck(cudaMemcpyAsync(host+v.compressed_offset,b.compressed+first*kHeadDim,size_t(v.compressed_bytes),cudaMemcpyDeviceToHost,t.outbound),"325 exact source comp D2H");ck(cudaMemcpyAsync(host+v.index_offset,b.index_keys+first*kIndexDim,size_t(v.index_bytes),cudaMemcpyDeviceToHost,t.outbound),"325 exact source index D2H");}
        if(out.carry_bytes){
            const uint64_t offset=v.window_bytes+v.compressed_bytes+v.index_bytes;
            ck(cudaMemcpyAsync(host+offset,b.kv_state,size_t(out.carry_bytes),cudaMemcpyDeviceToHost,t.outbound),"528 pending kv state D2H");
            ck(cudaMemcpyAsync(host+offset+out.carry_bytes,b.score_state,size_t(out.carry_bytes),cudaMemcpyDeviceToHost,t.outbound),"528 pending score state D2H");
        }
        ck(cudaEventRecord(t.sent,t.outbound),"325 host frame D2H done");ck(cudaEventSynchronize(t.sent),"325 immutable HOST_READY frame");
        return out;
    }
    static WaveFrame325 wave_export_attention337(const AttentionBoundary337&b,int layer,int begin,int end,
            const std::shared_ptr<HeadlessPinnedPrefixV320>&storage,WaveTransport325&t,const geometry335::Fixed&fixed335){
        t.current();if(layer<0||layer>=40)throw std::invalid_argument("325 export layer bounds");const int expected_ratio=kCompressRatio[layer];
        if(layer<0||layer>=40||b.version!=337||b.phase!=337||!b.epoch||b.layer!=layer||b.absolute_end!=end||b.ratio!=expected_ratio||b.window_bytes!=uint64_t(kWindow)*kHeadDim*2||!b.window||!b.ready||begin<0||end<=begin||end>fixed335.context||b.carry!=(expected_ratio>1?end%expected_ratio:0)||b.carry_bytes!=uint64_t(b.carry)*kHeadDim*sizeof(float)){wavefront325::add(27);early337::add(12);throw std::invalid_argument("337 attention-only layer/end/window/carry certificate");}
        const bool source=is_kv_source(layer);const uint64_t first=source?uint64_t(begin/expected_ratio):0,rows=source?uint64_t((end-begin)/expected_ratio):0;
        if(source&&(begin%expected_ratio||b.source_layer!=layer||!b.comp||!b.index_keys||b.compressed_rows!=end/expected_ratio||b.comp_bytes!=uint64_t(end/expected_ratio)*kHeadDim*2||b.index_bytes!=uint64_t(end/expected_ratio)*kIndexDim*2))throw std::invalid_argument("325 source slice extent");
        WaveFrame325 out;out.begin=begin;out.end=end;out.carry=b.carry;
        if(source&&expected_ratio==2&&b.carry){
            if(begin==0||!b.kv_state||!b.score_state)throw std::invalid_argument("528 reverse source carry ownership");
            out.carry_source=layer;out.carry_bytes=b.carry_bytes;
        }
        auto&v=out.prefix;
        v.producer_epoch=b.epoch;v.layer=layer;v.prefix_end=end;v.source_layer=source?layer:-1;v.window_bytes=131072;
        v.compressed_offset=source?v.window_bytes:0;v.compressed_bytes=rows*kHeadDim*2;
        v.index_offset=source?v.window_bytes+v.compressed_bytes:0;v.index_bytes=rows*kIndexDim*2;v.storage=storage;
        const uint64_t payload=out.bytes();
        if(!storage||storage->capacity()<payload)throw std::invalid_argument("325 typed owned frame too small");
        auto*host=static_cast<uint8_t*>(storage->data());
        ck(cudaStreamWaitEvent(t.outbound,reinterpret_cast<cudaEvent_t>(b.ready),0),"337 export waits separate attention event, fullFFN still pending");
        ck(cudaMemcpyAsync(host,b.window,size_t(v.window_bytes),cudaMemcpyDeviceToHost,t.outbound),"325 window D2H");
        if(source){ck(cudaMemcpyAsync(host+v.compressed_offset,b.comp+first*kHeadDim,size_t(v.compressed_bytes),cudaMemcpyDeviceToHost,t.outbound),"325 exact source comp D2H");ck(cudaMemcpyAsync(host+v.index_offset,b.index_keys+first*kIndexDim,size_t(v.index_bytes),cudaMemcpyDeviceToHost,t.outbound),"325 exact source index D2H");}
        if(out.carry_bytes){
            const uint64_t offset=v.window_bytes+v.compressed_bytes+v.index_bytes;
            ck(cudaMemcpyAsync(host+offset,b.kv_state,size_t(out.carry_bytes),cudaMemcpyDeviceToHost,t.outbound),"528 pending kv state D2H");
            ck(cudaMemcpyAsync(host+offset+out.carry_bytes,b.score_state,size_t(out.carry_bytes),cudaMemcpyDeviceToHost,t.outbound),"528 pending score state D2H");
        }
        ck(cudaEventRecord(t.sent,t.outbound),"325 host frame D2H done");ck(cudaEventSynchronize(t.sent),"325 immutable HOST_READY frame");
        return out;
    }
    void wave_import_return325(const WaveFrame325&frame,int layer,int total,WaveTransport325&t){
        t.current();if(layer<0||layer>=40)throw std::invalid_argument("325 import layer bounds");const auto&v=frame.prefix;const bool source=is_kv_source(layer);const auto&y=L[layer];
        const uint64_t rows=source?uint64_t((total-fixed335.split)/y.ratio):0,first=source?uint64_t(fixed335.split/y.ratio):0;
        if(frame.begin!=fixed335.split||frame.end!=total||v.version!=320||!v.producer_epoch||v.layer!=layer||v.prefix_end!=total||v.window_offset||v.window_bytes!=131072||v.source_layer!=(source?layer:-1)||v.compressed_bytes!=rows*kHeadDim*2||v.index_bytes!=rows*kIndexDim*2||v.compressed_offset!=(source?131072:0)||v.index_offset!=(source?131072+v.compressed_bytes:0)||!v.storage||v.storage->capacity()<v.window_bytes+v.compressed_bytes+v.index_bytes)throw std::invalid_argument("325 exact return span/layer/source certificate");
        const int carry528=y.ratio>1?total%y.ratio:0;
        const uint64_t pending528=source&&y.ratio==2?uint64_t(carry528)*kHeadDim*sizeof(float):0;
        if(frame.carry!=carry528||frame.carry_bytes!=pending528||frame.carry_source!=(pending528?layer:-1)||
           v.storage->capacity()<frame.bytes()||(pending528&&(!y.kv_state||!y.score_state)))
            throw std::invalid_argument("528 reverse carry certificate");
        const auto*host=static_cast<const uint8_t*>(v.storage->data());
        ck(cudaMemcpyAsync(y.window,host,size_t(v.window_bytes),cudaMemcpyHostToDevice,t.inbound),"325 final physical window H2D");
        if(source){ck(cudaMemcpyAsync(y.comp+first*kHeadDim,host+v.compressed_offset,size_t(v.compressed_bytes),cudaMemcpyHostToDevice,t.inbound),"325 disjoint source suffix H2D");ck(cudaMemcpyAsync(y.idx_keys+first*kIndexDim,host+v.index_offset,size_t(v.index_bytes),cudaMemcpyHostToDevice,t.inbound),"325 disjoint index suffix H2D");}
        if(pending528){
            const uint64_t offset=v.window_bytes+v.compressed_bytes+v.index_bytes;
            ck(cudaMemcpyAsync(y.kv_state,host+offset,size_t(pending528),cudaMemcpyHostToDevice,t.inbound),"528 pending kv state H2D");
            ck(cudaMemcpyAsync(y.score_state,host+offset+pending528,size_t(pending528),cudaMemcpyHostToDevice,t.inbound),"528 pending score state H2D");
        }
        ck(cudaEventRecord(t.received,t.inbound),"325 return import done");ck(cudaEventSynchronize(t.received),"325 consumed immutable return frame");
        const auto bytes=frame.bytes();wavefront325::add(11);wavefront325::add(13,bytes);wavefront325::add(15,2*bytes);if(source)wavefront325::add(26);
    }
    int prefill_wavefront_impl325(const std::vector<int>&tokens,int position,std::vector<float>*nll,PrefillTiming&pt){
        usable();int current=-1;ck(cudaGetDevice(&current),"325 headful entry device");
        //407: PPgeometry/headless20GiB are unchanged. Only the later
        //GPU1whole-cache trigger may carry explicit403static22GiB.
        const char*whole407=std::getenv("DS1017_WHOLE_ENGINE_V387");
        const char*adaptive407=std::getenv("DS1017_WHOLE_ADAPTIVE_V370");
        const char*static407=std::getenv("DS1017_WHOLE_STATIC_BUDGET_GIB_V403");
        const bool is_whole407=whole407&&!std::strcmp(whole407,"1");
        const bool legacy20_407=native_gu_budget_v19==(size_t(20)<<30)&&
            (!is_whole407||!static407||!std::strcmp(static407,"20"));
        const bool static22_407=native_gu_budget_v19==(size_t(22)<<30)&&is_whole407&&
            adaptive407&&!std::strcmp(adaptive407,"0")&&static407&&!std::strcmp(static407,"22");
        if(!wavefront_latched325||headless320||current!=0||!outer325||!native_cpu||!pack.native()||position||!history.empty()||tokens.size()!=size_t(fixed335.prompt_end)||max_seq!=524288||opt.prefill_chunk!=fixed335.cap||opt.prefill_batch!=fixed335.B||native_host_expert_batch!=fixed335.E||nll||native_images||active_pass317||active_pair_indexer_v1||verify_pending||native_gpu_experts||native_gpu_pool||!native_gu_budget_v19||!(legacy20_407||static22_407)||native_cache_tail_rows||!native_capture_main||!native_main_ring||!pack.dense_stream_enabled_v190())throw std::invalid_argument("325 fresh exact filled512 text/native4/GU20 two-pass geometry required");
        const char*pair_helper=std::getenv("DS1017_PAIR_INDEXER_PREFILL_V1");if(!pair_helper||std::strcmp(pair_helper,"0"))throw std::invalid_argument("325 requires explicit local K14/old pairhelperOFF");
        const char*headless=std::getenv("DS1017_HEADLESS_PREFILL_V320");if(!headless||std::strcmp(headless,"1"))throw std::invalid_argument("325 explicit headless320 flag1 required");
        const int cpu0=wave_cpu325("DS1017_WAVEFRONT_GPU0_CPU_V325"),cpu1=wave_cpu325("DS1017_WAVEFRONT_GPU1_CPU_V325");if(cpu0==cpu1)throw std::invalid_argument("325 distinct issuer CPUs");
        bool expected=false;if(!wavefront325::active.compare_exchange_strong(expected,true))throw std::logic_error("325 N1 live coordinator only");
        struct Active{~Active(){wavefront325::active=false;}}active_guard;
        wavefront325::set(1,1);wavefront325::add(2);wavefront325::set(5,1);wavefront325::set(43,cpu0);wavefront325::set(44,cpu1);
        pt=PrefillTiming{};ptm=&pt;pt.chunk_tokens=fixed335.cap;pt.sub_batch=fixed335.B;const double started=now_ms();
        WaveShared325 shared;std::thread rank0,rank1;
        try{
            for(auto&f:shared.forward)f=HeadlessPinnedPrefixV320::allocate(fixed335.max_frame(false));
            for(auto&f:shared.reverse)f=HeadlessPinnedPrefixV320::allocate(fixed335.max_frame(true));
            wavefront325::set(16,2*(fixed335.max_frame(false)+fixed335.max_frame(true)));
            rank0=std::thread([&]{
                std::unique_ptr<PrefillPassState317>state;std::unique_ptr<WaveTransport325>transport;
                try{
                    wave_pin325(cpu0);ck(cudaSetDevice(0),"325 rank0 current");stage_copy_rank_mode325=true;wavefront325::add(17);
                    transport=std::make_unique<WaveTransport325>(0);wave_reserve325(0);
                    begin_streamed_dense_v191(0);prefill_begin(int(tokens.size()));
                    if(pf.cap!=fixed335.cap||pf.sub!=fixed335.B||pf.native_expert_batch!=fixed335.E)throw std::runtime_error("325 GPU0 actual geometry changed");
                    wavefront325::set(31,uint64_t(active_pair_indexer_v1!=nullptr));pf_started=true;
                    state=prefill_pass_begin317(tokens,0,fixed335.split,0,nullptr);
                    shared.set(&WaveShared325::ready0);shared.wait(&WaveShared325::ready1);
                    int returned=0,attention_ready0_337=-1;uint64_t reverse_epoch=0;
                    auto pump=[&]{while(returned<40){auto packet=shared.credits.try_take(1,returned);if(!packet)break;if(early_attention337&&returned>attention_ready0_337)throw std::logic_error("337 return import precedes own same-layer attention");if(reverse_epoch&&reverse_epoch!=packet->packet.prefix.producer_epoch)throw std::logic_error("325 reverse pass epoch changed");reverse_epoch=packet->packet.prefix.producer_epoch;wave_import_return325(packet->packet,returned,int(tokens.size()),*transport);shared.credits.release(1,packet->claim);++returned;}};
                    for(int l=0;l<40;++l){
                        shared.credits.check();pump();wave_reserve325(0);
                        if(early_attention337){
                            state->early_callback337=[&,l](const AttentionBoundary337&boundary){
                                if(boundary.layer!=l||boundary.phase!=337)throw std::logic_error("337 forward attention callback identity");attention_ready0_337=l;
                                std::optional<WaveCredits325::Claim>claim;
                                while(!claim){auto old=shared.credits.revision();pump();claim=shared.credits.try_claim(0);if(!claim)shared.credits.wait_progress(old);}
                                auto packet=wave_export_attention337(boundary,l,0,fixed335.split,shared.forward[claim->slot],*transport,fixed335);
                                const uint64_t bytes=packet.bytes();
                                shared.credits.publish(0,*claim,l,std::move(packet));wavefront325::add(8);wavefront325::add(12,bytes);wavefront325::add(14,2*bytes);if(is_kv_source(l))wavefront325::add(25);
                                early337::add(4);early337::add(is_kv_source(l)?17:18);pump();
                            };
                            prefill_pass_advance317(*state,l);state->early_callback337={};wavefront325::add(6);
                            ck(cudaEventSynchronize(state->layer_done),"337 separate GPU0 FULL layer completion AFTER originalFFN");early337::add(6);early337::set(22,l+1);pump();continue;
                        }
                        prefill_pass_advance317(*state,l);wavefront325::add(6);
                        const auto boundary=wave_boundary325(prefill_pass_boundary317(*state));
                        std::optional<WaveCredits325::Claim>claim;
                        while(!claim){auto old=shared.credits.revision();pump();claim=shared.credits.try_claim(0);if(!claim)shared.credits.wait_progress(old);}
                        auto packet=wave_export325(boundary,l,0,fixed335.split,shared.forward[claim->slot],*transport,fixed335);
                        const uint64_t bytes=packet.bytes();
                        // D2H is complete: original borrowed layer event may now
                        // be re-recorded. Host frame stays claimed until H2D ACK.
                        ck(cudaEventQuery(state->layer_done),"325 GPU0 source boundary ACK");
                        shared.credits.publish(0,*claim,l,std::move(packet));wavefront325::add(8);wavefront325::add(12,bytes);wavefront325::add(14,2*bytes);if(is_kv_source(l))wavefront325::add(25);
                        pump();
                    }
                    prefill_pass_finish317(*state);state.reset();
                    while(returned<40){auto old=shared.credits.revision();pump();if(returned<40)shared.credits.wait_progress(old);}
                    shared.wait(&WaveShared325::done1);shared.credits.require_drained();
                    const auto&result=shared.result;
                    if(result.final_row.size()!=size_t(kDim)||result.tail_first!=int(tokens.size())-128||result.tail.size()!=size_t(128)*3*kDim)throw std::runtime_error("325 final row/real native4 tail incomplete");
                    if(wavefront325::stats[6]!=40||wavefront325::stats[7]!=40||wavefront325::stats[8]!=40||wavefront325::stats[9]!=40||wavefront325::stats[10]!=40||wavefront325::stats[11]!=40||wavefront325::stats[25]!=4||wavefront325::stats[26]!=4||wavefront325::stats[27]||wavefront325::stats[12]!=fixed335.payload(false)||wavefront325::stats[13]!=fixed335.payload(true)||wavefront325::stats[14]!=2*fixed335.payload(false)||wavefront325::stats[15]!=2*fixed335.payload(true))throw std::runtime_error("325 incomplete layer/source/ACK/carry census");
                    // The actual320 typed merge validates realcheckpoint binding,
                    // disjoint prefix history and all route additions BEFORE any
                    // update. It neither fabricates a cache nor publishes history.
                    outer325->merge_headless_frequency_v320(result.frequency,tokens);wavefront325::add(22);
                    const int first=result.tail_first,initial=std::min(128,128-first%128);
                    ck(cudaMemcpy(native_main_ring+size_t(first%128)*3*kDim,result.tail.data(),size_t(initial)*3*kDim*2,cudaMemcpyHostToDevice),"325 exact native4 tail first segment");
                    if(initial<128)ck(cudaMemcpy(native_main_ring,result.tail.data()+size_t(initial)*3*kDim,size_t(128-initial)*3*kDim*2,cudaMemcpyHostToDevice),"325 exact native4 tail wrapped segment");
                    wavefront325::set(20,128ull*3*kDim*2);
                    ck(cudaMemcpy(final_x,result.final_row.data(),kDim*2,cudaMemcpyHostToDevice),"325 already-rounded final BF16 row");
                    ops::bf16_linear(final_x,nullptr,head,kDim,kVocab,nullptr,logits);lg.resize(kVocab);ck(cudaMemcpy(lg.data(),logits,kVocab*4,cudaMemcpyDeviceToHost),"325 ORIGINAL GPU0 M1 head/logits");wavefront325::set(21,kDim*2);
                    // Last imported source must remain GPU0-owned. No foreign
                    // addresses are ever installed in decode state.
                    cur_comp=L[20].comp;cur_index_k=L[20].idx_keys;
                    if(!cur_comp||!cur_index_k||history.size()!=fixed335.split)throw std::logic_error("325 final source/history ownership");
                    for(size_t i=fixed335.split;i<tokens.size();++i)history.push_back(pack.engram_hash().token_map[tokens[i]]);
                    if(history.size()!=tokens.size())throw std::logic_error("325 committed history/end");
                    for(size_t i=0;i<tokens.size();++i)if(history[i]!=pack.engram_hash().token_map[tokens[i]])throw std::logic_error("325 final full history hash");
                    // Actual Engine::step/verify consume EXPLICIT position and
                    // history.size(), not a hidden stored pos. Publish the last
                    // completed input's real StepParams and causal lengths.
                    *hp=StepParams{tokens.back(),int(tokens.size())-1,int(tokens.size()),int(tokens.size()/2)};
                    parity=(int(tokens.size())-1)&1;tcap1=cap_of(tokens.size(),max_seq+1);tcap2=cap_of(tokens.size()/2,max_seq/2+1);
                    wavefront_committed_end325=tokens.size();wavefront325::set(23,1);wavefront325::set(24,wavefront_committed_end325);
                    // Truthful aggregation of both ORIGINAL pass consumers. The
                    // registry145 is shared; these252/suffix fields are Engine-local.
                    output16_uploads_v252+=result.output[4];output16_bytes_v252+=result.output[5];output16_encodes_v252+=result.output[6];output16_max_triple_v252=std::max(output16_max_triple_v252,result.output[8]);output16_raw_uploads_v252+=result.output[9];
                    direct_fp8_projection_encodes+=result.packed[1];decoder_suffix_passes+=result.suffix[1];decoder_suffix_layers+=result.suffix[2];decoder_suffix_rows_skipped+=result.suffix[3];compact_candidate_packed_rows+=result.compact[1];compact_candidate_unpacked_rows+=result.compact[2];compact_candidate_saved_peak=std::max(compact_candidate_saved_peak,result.compact[3]);
                    native_prefill_sampled_passes+=result.memory[2];native_prefill_minimum_free_bytes=std::min(native_prefill_minimum_free_bytes,result.memory[1]);
                    pt.chunks=2;pt.engram_rows+=result.timing.engram_rows;pt.engram_unique+=result.timing.engram_unique;pt.engram_ms+=result.timing.engram_ms;pt.stream_wait_ms+=result.timing.stream_wait_ms;
                    prefill_end();restore_streamed_dense_v191();wave_reserve325(0);wavefront325::set(38,1);
                }catch(...){shared.fail(std::current_exception());broken=true;wavefront325::add(4);if(transport&&!transport->drain())std::terminate();if(cudaStreamSynchronize(0)!=cudaSuccess)std::terminate();if(state)state->early_abort_after_drain337();state.reset();prefill_end_quiet();}
            });
            rank1=std::thread([&]{
                std::unique_ptr<Engine>other;std::unique_ptr<WaveTransport325>transport;
                try{
                    wave_pin325(cpu1);ck(cudaSetDevice(1),"325 rank1 current");stage_copy_rank_mode325=true;wavefront325::add(18);
                    shared.wait(&WaveShared325::ready0);shared.credits.check();
                    NativeEngineOptions none;none.graph=false;none.a8=native_options.a8;none.mixed_f32=native_options.mixed_f32;
                    HeadlessOptionsV320 limits;
                    EngineOptions rank_options=opt;rank_options.prefill_chunk=0;
                    other=std::make_unique<Engine>(pack.native(),pack.engram_hash(),rank_options,none,EngineRoleV320::HeadlessPrefillRank1,limits);
                    auto&r=*other->impl_;transport=std::make_unique<WaveTransport325>(1);other->headless_begin_pass_v320(tokens,fixed335.split);
                    wavefront325::set(28,uint64_t(bool(r.native_cpu)));wavefront325::set(29,uint64_t(r.head!=nullptr));wavefront325::set(30,uint64_t(r.active_pair_indexer_v1!=nullptr));wavefront325::set(35,r.headless_options320.max_device_bytes);wavefront325::set(36,r.headless_own_bytes320+r.pack.headless_dense_info_v320()[7]);wavefront325::set(37,r.pack.headless_dense_info_v320()[7]);
                    if(r.native_cpu||r.worker.joinable()||r.db||r.head||r.active_pair_indexer_v1||!r.native_capture_main||!r.native_main_ring)throw std::logic_error("325 headless resource ownership violated");
                    wave_reserve325(1);shared.set(&WaveShared325::ready1);
                    for(int l=0;l<40;++l){
                        shared.credits.check();auto forward=shared.credits.take(0,l);
                        other->headless_import_prefix_v320(forward.packet.prefix);
                        // This is the precise receiver H2D ACK. Waiting is on
                        // GPU1's own event; release credit BEFORE long layerwork.
                        ck(cudaEventSynchronize(r.headless_import_event320),"325 prefix H2D consumed credit");
                        shared.credits.release(0,forward.claim);wavefront325::add(9);
                        if(early_attention337){
                            if(!r.early_attention337||!r.headless_state320)throw std::logic_error("337 headless mode/state mismatch");
                            r.headless_state320->early_callback337=[&,l](const AttentionBoundary337&boundary){
                                if(boundary.layer!=l||boundary.phase!=337)throw std::logic_error("337 reverse attention callback identity");
                                std::optional<WaveCredits325::Claim>claim;
                                while(!claim){auto old=shared.credits.revision();claim=shared.credits.try_claim(1);if(!claim)shared.credits.wait_progress(old);}
                                auto packet=wave_export_attention337(boundary,l,fixed335.split,int(tokens.size()),shared.reverse[claim->slot],*transport,fixed335);
                                shared.credits.publish(1,*claim,l,std::move(packet));wavefront325::add(10);early337::add(5);early337::add(is_kv_source(l)?17:18);
                            };
                            auto full=other->headless_advance_v320(l);r.headless_state320->early_callback337={};wavefront325::add(7);wave_reserve325(1);
                            ck(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(full.event)),"337 separate GPU1 FULL layer completion AFTER originalFFN");early337::add(7);early337::set(23,l+1);
                            other->headless_ack_layer_v320(full.epoch,l);continue;
                        }
                        auto boundary=other->headless_advance_v320(l);wavefront325::add(7);wave_reserve325(1);
                        std::optional<WaveCredits325::Claim>claim;
                        while(!claim){auto old=shared.credits.revision();claim=shared.credits.try_claim(1);if(!claim)shared.credits.wait_progress(old);}
                        auto packet=wave_export325(boundary,l,fixed335.split,int(tokens.size()),shared.reverse[claim->slot],*transport,fixed335);
                        other->headless_ack_layer_v320(boundary.epoch,l);
                        shared.credits.publish(1,*claim,l,std::move(packet));wavefront325::add(10);
                    }
                    WaveResult325 result;result.final_row=other->headless_finish_pass_v320();result.frequency=other->headless_frequency_v320();result.tail=other->native_main_tail(result.tail_first,128);
                    result.output=other->native_output16_info_v252();result.suffix=other->native_decoder_suffix_info();result.compact=other->native_compact_candidate_info();result.packed=other->native_packed_fp8_info();result.memory=other->native_prefill_memory_info();result.timing=r.headless_timing320;
                    other->headless_close_v320();other.reset();transport.reset();wavefront325::set(34,1);
                    {std::lock_guard<std::mutex>g(shared.mutex);shared.result=std::move(result);shared.done1=true;shared.changed.notify_all();}
                }catch(...){shared.fail(std::current_exception());wavefront325::add(4);
                    // Abort is NOT a successful publication ACK. Only after all
                    // source/transport work actually drained may we invalidate a
                    // borrow, reset the owned pass and let320 close normally.
                    if(transport&&!transport->drain())std::terminate();if(cudaStreamSynchronize(0)!=cudaSuccess)std::terminate();
                    if(other){auto&r=*other->impl_;r.headless_ack320=false;r.headless_import_owner320.reset();if(r.headless_state320)r.headless_state320->early_abort_after_drain337();r.headless_state320.reset();other->headless_close_v320();other.reset();wavefront325::add(39);}
                }
            });
        }catch(...){shared.fail(std::current_exception());}
        if(rank0.joinable()){rank0.join();wavefront325::add(19);}if(rank1.joinable()){rank1.join();wavefront325::add(19);}
        wavefront325::set(5,0);ck(cudaSetDevice(0),"325 caller GPU0 restore");
        if(shared.error){broken=true;std::rethrow_exception(shared.error);}
        shared.credits.require_drained();if(wavefront_committed_end325!=tokens.size()||history.size()!=tokens.size())throw std::logic_error("325 complete end publication missing");
        pt.total_ms=now_ms()-started;wavefront325::set(40,uint64_t(pt.total_ms*1000000.0));wavefront325::add(3);
        return int(std::max_element(lg.begin(),lg.end())-lg.begin());
    }
