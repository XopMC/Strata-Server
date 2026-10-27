HeadlessPinnedPrefixV320::HeadlessPinnedPrefixV320(uint64_t bytes):bytes_(bytes){
    if(!bytes||bytes>(uint64_t(512)<<20))throw std::invalid_argument("320 pinned staging capacity1..512MiB");
    cudaStreamCaptureStatus capture{};ck(cudaStreamIsCapturing(0,&capture),"320 pinned staging capture guard");
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("320 staging allocation is idle/setup-only");
    ck(cudaHostAlloc(&data_,size_t(bytes),cudaHostAllocPortable),"320 portable pinned prefix staging");
}
std::shared_ptr<HeadlessPinnedPrefixV320>HeadlessPinnedPrefixV320::allocate(uint64_t bytes){
    return std::shared_ptr<HeadlessPinnedPrefixV320>(new HeadlessPinnedPrefixV320(bytes));
}
HeadlessPinnedPrefixV320::~HeadlessPinnedPrefixV320()noexcept{
    // Owning coordinator and receiver retain through every submitted DMA.
    if(data_&&cudaFreeHost(data_)!=cudaSuccess)std::terminate();
}
// Public typed C++ bridge. The real caller supplies the same shared_ptr<NativeCheckpoint>.
// There is no PID attach, fake borrowed token or new model-file/bank loader.
void Engine::headless_begin_pass_v320(const std::vector<int>&t,int first){impl_->headless_begin_impl320(t,first);}
void Engine::headless_import_prefix_v320(const HeadlessPrefixV320&v){impl_->headless_import_impl320(v);}
HeadlessBoundaryV320 Engine::headless_advance_v320(int layer){return impl_->headless_advance_impl320(layer);}
void Engine::headless_ack_layer_v320(uint64_t epoch,int layer){impl_->headless_ack_impl320(epoch,layer);}
std::vector<uint16_t> Engine::headless_finish_pass_v320(){return impl_->headless_finish_impl320();}
void Engine::headless_close_v320(){impl_->headless_close_impl320();}
std::array<uint64_t,16>Engine::headless_info_v320()const{
    const auto&p=*impl_;if(!p.headless320)return {320,0,0,uint64_t(p.native_math320()),uint64_t(bool(p.native_cpu)),uint64_t(p.worker.joinable()),uint64_t(bool(p.db)),uint64_t(p.head!=nullptr),0,0,0,0,0,0,0,0};
    p.headless_current320();const auto d=p.pack.headless_dense_info_v320();
    return {320,320,uint64_t(p.headless_device320),uint64_t(p.native_math320()),uint64_t(bool(p.native_cpu)),uint64_t(p.worker.joinable()),uint64_t(bool(p.db)),uint64_t(p.head!=nullptr),d[2],p.headless_startup_plan320,p.headless_own_bytes320,d[7],p.headless_options320.max_device_bytes,p.headless_options320.reserve_bytes,uint64_t(bool(p.headless_state320)),uint64_t(p.headless_ack320)};
}
std::array<uint64_t,8>Engine::pack_headless_info_v320()const{return impl_->pack.headless_dense_info_v320();}

HeadlessFrequencyV320 Engine::headless_frequency_v320()const{
    const auto&p=*impl_;p.headless_current320();p.usable();
    if(p.headless_state320||p.headless_ack320||p.history.size()!=p.headless_tokens320.size()||p.headless_tokens320.empty())
        throw std::logic_error("320 frequency export requires completed final pass");
    const auto binding=pair145::info(*p.pack.native()->weights());HeadlessFrequencyV320 out;
    out.owner_generation=binding[4];out.format_epoch=binding[5];out.pass_epoch=p.pass_epoch317;out.tail_counted_uses=p.native_cache_tail_counted_uses;
    out.observed_begin=p.headless_observed_first320;out.observed_end=int(p.history.size());out.tail_rows=p.native_cache_tail_rows;out.full_prompt=p.headless_tokens320;
    for(int l=0;l<kLayers;++l)out.counts[l]=p.native_expert_frequency[l];return out;
}
void Engine::merge_headless_frequency_v320(const HeadlessFrequencyV320&v,const std::vector<int>&expected){
    auto&p=*impl_;p.usable();int device=-1;ck(cudaGetDevice(&device),"320 frequency merge current device");
    const auto binding=p.pack.native()?pair145::info(*p.pack.native()->weights()):std::array<uint64_t,8>{};
    if(p.headless320||device!=0||!p.native_cpu||!p.native_gu_budget_v19||p.native_gpu_experts||p.verify_pending||p.active_pass317||p.headless_frequency_merged320||
       v.version!=320||!v.pass_epoch||v.owner_generation!=binding[4]||v.format_epoch!=binding[5]||binding[2]!=46080||
       v.observed_begin!=262144||p.history.size()!=262144||v.observed_end<=262144||v.observed_end>524288||
       v.full_prompt!=expected||expected.size()!=size_t(v.observed_end)||v.tail_rows!=p.native_cache_tail_rows)
        throw std::invalid_argument("320 disjoint actual suffix frequency merge/owner/geometry guard");
    const auto existing_gu=p.native_cpu->gu_statistics_v19();
    if(std::any_of(existing_gu.begin(),existing_gu.end(),[](uint64_t x){return x!=0;}))throw std::logic_error("320 frequency merge must precede any GU cache construction");
    if(v.tail_rows&&(p.native_cache_tail_begin!=std::max(0,v.observed_end-v.tail_rows)||p.native_cache_tail_end!=v.observed_end||!p.native_cache_tail_active))
        throw std::invalid_argument("320 original fullprompt tail window mismatch");
    for(size_t i=0;i<expected.size();++i){Impl::check_token(expected[i]);if(i<262144&&p.history[i]!=p.pack.engram_hash().token_map[expected[i]])throw std::invalid_argument("320 frequency prefix token/hash mismatch");}
    // Validate every addition before changing any counter. This is observed
    // route metadata only, not a future-token oracle or cache publication.
    const uint64_t bound=uint64_t(v.observed_end-v.observed_begin)*kTopK;
    uint64_t actual_uses=0;
    for(int l=0;l<kLayers;++l){uint64_t total=0;for(int e=0;e<kExperts;++e){const auto n=v.counts[l][e];if(n>bound||total>bound-n||p.native_expert_frequency[l][e]>UINT64_MAX-n)throw std::overflow_error("320 frequency merge bounds/overflow");total+=n;}actual_uses+=total;}
    if((v.tail_rows?v.tail_counted_uses!=actual_uses:v.tail_counted_uses!=0)||p.native_cache_tail_counted_uses>UINT64_MAX-v.tail_counted_uses)throw std::invalid_argument("320 actual tail-use accounting mismatch");
    for(int l=0;l<kLayers;++l)for(int e=0;e<kExperts;++e)p.native_expert_frequency[l][e]+=v.counts[l][e];
    p.native_cache_tail_counted_uses+=v.tail_counted_uses;
    p.headless_frequency_merged320=true;
}
