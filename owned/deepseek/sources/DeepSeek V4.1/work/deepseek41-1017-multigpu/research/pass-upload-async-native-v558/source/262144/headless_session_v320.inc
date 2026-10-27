// Inside Engine::Impl, after the literal317 pass type/functions are complete.
// No second CPU projection pool, registry registration, decoder or target graph.
    void headless_begin_impl320(const std::vector<int>&full,int first){
        headless_current320();usable();
        if(headless_closed320||headless_state320||headless_ack320||active_pass317||
           full.size()<=size_t(fixed335.split)||full.size()>size_t(fixed335.context)||(fixed335.is256()&&full.size()!=size_t(fixed335.prompt_end))||(first!=0&&first!=fixed335.split))
            throw std::invalid_argument("320 bounded original512 pass geometry or lifetime");
        for(int t:full)check_token(t);
        if(!headless_pf320){
            if(!history.empty()||pf.own_scratch||!headless_tokens320.empty())throw std::logic_error("320 fresh headless PP state required");
            headless_tokens320=full;headless_timing320=PrefillTiming{};ptm=&headless_timing320;
            headless_observed_first320=first;
            native_cache_tail_active=native_cache_tail_rows!=0;
            native_cache_tail_begin=native_cache_tail_active?std::max(0,int(full.size())-native_cache_tail_rows):0;
            native_cache_tail_end=native_cache_tail_active?int(full.size()):0;
            native_cache_tail_counted_uses=0;for(auto&counts:native_expert_frequency)counts.fill(0);
            const char*s=std::getenv("DS41_SUFFIX_CANDIDATE_STORAGE_R1");suffix_candidate_active_r1=s&&!std::strcmp(s,"1");
            const char*r=std::getenv("DS41_RECOMPUTE_INPUTS_R1");recompute_inputs_active_r1=r&&!std::strcmp(r,"1");
            try{
                begin_streamed_dense_v191(0);prefill_begin(int(full.size()));
                if(pf.cap!=fixed335.cap||pf.sub!=fixed335.B||pf.native_expert_batch!=fixed335.E)throw std::runtime_error("320 actual scratch shrank original cap/B/E geometry");
                // Native Engram hashes use the ORIGINAL full absolute prefix.
                // This seeds metadata only; no HC/activation relay or CPU weights.
                if(first)for(int i=0;i<first;++i)history.push_back(pack.engram_hash().token_map[full[i]]);
                headless_pf320=true;
            }catch(...){broken=true;throw;}
        }else if(headless_tokens320!=full)throw std::invalid_argument("320 immutable fullprompt changed between passes");
        if(history.size()!=size_t(first))throw std::invalid_argument("320 pass order differs from owned absolute history");
        headless_final320.clear();headless_imported_layer320=-1;headless_import_epoch320=0;
        try{headless_state320=prefill_pass_begin317(headless_tokens320,first,first?int(full.size())-fixed335.split:fixed335.split,first,nullptr);}
        catch(...){broken=true;throw;}
    }
    static const void*headless_host_span320(const HeadlessPinnedPrefixV320&buffer,uint64_t offset,uint64_t bytes){
        if(!bytes||offset>buffer.capacity()||bytes>buffer.capacity()-offset)throw std::invalid_argument("320 pinned prefix offset/extent out of owned buffer");
        return static_cast<const uint8_t*>(buffer.data())+offset;
    }
    void headless_import_impl320(const HeadlessPrefixV320&v){
        headless_current320();usable();
        if(!headless_state320||headless_ack320||headless_import_owner320)throw std::logic_error("320 prefix import is not at an idle layer boundary");
        const auto&q=*headless_state320;check_pass317(q);
        if(v.version!=320||!v.producer_epoch||!v.storage||v.layer!=q.next_layer||v.layer<0||v.layer>=kLayers||q.p0!=fixed335.split||v.prefix_end!=q.p0||headless_imported_layer320==v.layer||(headless_import_epoch320&&v.producer_epoch!=headless_import_epoch320))
            throw std::invalid_argument("320 prefix epoch/layer/end/owner guard");
        const bool source=is_kv_source(v.layer);const auto&y=L[v.layer];
        const uint64_t expected_window=uint64_t(kWindow)*kHeadDim*2;
        const uint64_t rows=source?uint64_t(q.p0/y.ratio):0;
        if(v.window_offset||v.window_bytes!=expected_window||
           (source&&(v.compressed_offset!=expected_window||v.index_offset!=expected_window+rows*kHeadDim*2))||
           v.compressed_bytes!=rows*kHeadDim*2||v.index_bytes!=rows*kIndexDim*2||
           (source?v.source_layer!=v.layer:v.source_layer!=-1)||(!source&&(v.compressed_offset||v.index_offset))||
           (source&&q.p0%y.ratio))throw std::invalid_argument("320 exact prefix spans/ratio/source mismatch");
        const void*window=headless_host_span320(*v.storage,v.window_offset,v.window_bytes);
        const void*compressed=source?headless_host_span320(*v.storage,v.compressed_offset,v.compressed_bytes):nullptr;
        const void*index=source?headless_host_span320(*v.storage,v.index_offset,v.index_bytes):nullptr;
        if(!headless_import_event320)ck(cudaEventCreateWithFlags(&headless_import_event320,cudaEventDisableTiming),"320 prefix H2D event");
        // Retain BEFORE first DMA; any subsequent submission error poisons this
        // context and close drains this stream BEFORE releasing the host token.
        headless_import_owner320=v.storage;
        try{
            ck(cudaMemcpyAsync(y.window,window,size_t(v.window_bytes),cudaMemcpyHostToDevice,0),"320 exact window prefix H2D");
            if(source){
                ck(cudaMemcpyAsync(y.comp,compressed,size_t(v.compressed_bytes),cudaMemcpyHostToDevice,0),"320 exact compressed prefix H2D");
                ck(cudaMemcpyAsync(y.idx_keys,index,size_t(v.index_bytes),cudaMemcpyHostToDevice,0),"320 exact index prefix H2D");
            }
            ck(cudaEventRecord(headless_import_event320,0),"320 prefix retention event");
            headless_imported_layer320=v.layer;headless_import_epoch320=v.producer_epoch;
        }catch(...){broken=true;throw;}
    }
    HeadlessBoundaryV320 headless_advance_impl320(int layer){
        headless_current320();usable();
        if(!headless_state320||headless_ack320)throw std::logic_error("320 previous publication must be ACKed before advance");
        auto&q=*headless_state320;check_pass317(q);
        if(layer!=q.next_layer||layer<0||layer>=kLayers)throw std::invalid_argument("320 exact next layer required");
        if(q.p0){
            if(headless_imported_layer320!=layer||!headless_import_owner320||!headless_import_epoch320)throw std::logic_error("320 external prefix import required for EVERY second-pass layer");
            try{ck(cudaEventSynchronize(headless_import_event320),"320 release consumed pinned prefix source");headless_import_owner320.reset();}
            catch(...){broken=true;throw;}
        }
        try{prefill_pass_advance317(q,layer);}catch(...){broken=true;throw;}
        const auto b=prefill_pass_boundary317(q);HeadlessBoundaryV320 out;
        out.epoch=b.epoch;out.event=reinterpret_cast<uint64_t>(b.ready);out.layer=b.layer;out.next_layer=b.next_layer;
        out.c0=b.c0;out.rows=b.S;out.position=b.p0;out.absolute_end=b.absolute_end;out.ratio=b.ratio;out.source_layer=b.source_layer;out.compressed_rows=b.compressed_rows;out.carry=b.carry;
        out.window=reinterpret_cast<const uint16_t*>(b.window);out.compressed=reinterpret_cast<const uint16_t*>(b.comp);out.index_keys=reinterpret_cast<const uint16_t*>(b.index_keys);
        out.kv_state=b.kv_state;out.score_state=b.score_state;out.window_bytes=b.window_bytes;out.compressed_bytes=b.comp_bytes;out.index_bytes=b.index_bytes;out.carry_bytes=b.carry_bytes;
        headless_ack320=true;return out;
    }
    void headless_ack_impl320(uint64_t epoch,int layer){
        headless_current320();
        if(!headless_state320||!headless_ack320||headless_state320->epoch!=epoch||headless_state320->next_layer!=layer+1)
            throw std::invalid_argument("320 exact outstanding layer epoch required");
        // Only proves original producer event complete. The typed caller ALSO
        // promises all its own D2H consumers completed before this ACK.
        ck(cudaEventQuery(headless_state320->layer_done),"320 layer publication not complete");headless_ack320=false;
    }
    std::vector<uint16_t>headless_finish_impl320(){
        headless_current320();usable();
        if(!headless_state320||headless_ack320||headless_import_owner320||headless_state320->next_layer!=kLayers)
            throw std::logic_error("320 finish requires all40 original layers/ACKs/imports consumed");
        try{prefill_pass_finish317(*headless_state320);headless_state320.reset();++headless_timing320.chunks;}
        catch(...){broken=true;throw;}
        return headless_final320; // empty for first pass; final pass exactly5120 BF16
    }
    void headless_close_impl320(){
        if(headless_closed320)return;headless_current320();
        if(headless_ack320)throw std::logic_error("320 cannot close borrowed layer publication before consumer ACK");
        // On failure, do NOT drop host_owner/device storage with an unproved DMA.
        ck(cudaStreamSynchronize(0),"320 close/drain original PP stream");
        headless_import_owner320.reset();headless_state320.reset();
        if(headless_import_event320){ck(cudaEventDestroy(headless_import_event320),"320 import event close");headless_import_event320=nullptr;}
        prefill_end();headless_pf320=false;headless_closed320=true;
        // No post-PF full dense restore, vocabulary head, cache owner or graph.
        // Pack still owns its two banks until this context's destruction.
    }
