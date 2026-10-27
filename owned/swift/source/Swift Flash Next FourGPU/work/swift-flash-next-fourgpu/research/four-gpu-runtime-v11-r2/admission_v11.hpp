#pragma once
#include "reference_v11.hpp"
namespace four_runtime_main_v11 {
inline bool inactive(const char*k){auto*v=std::getenv(k);return !v||!*v||!std::strcmp(v,"0");}
inline void environment(){using four_reference_v11::need;const char*v=std::getenv("STRATA_FOUR_RUNTIME_V11");need(v&&!std::strcmp(v,"1"),"explicit experimental runtime gate required");
 for(auto k:{"STRATA_ADAPT_NOWAIT","STRATA_ADAPT_OVERLAP","STRATA_ADAPT_ASYNC","STRATA_RAM_ADAPT","STRATA_RAM_ADAPT2","STRATA_ROUTE_PRIOR","STRATA_ROUTE_RESIDENT","STRATA_ADMIT_MERGE_FIRST","STRATA_SPLIT_OVERLAP","STRATA_HANDOFF_PIPE","STRATA_HANDOFF_PIPE_WORKER","STRATA_PREFILL_HELP","STRATA_PREFILL_CPU_SHARE","STRATA_SPLIT_SMALL_OWN","STRATA_SPLIT_SMALL_MAX","STRATA_VERIFY_DEVICE_PLAN","STRATA_SPEC_COUPLED","STRATA_SPEC_PROB","STRATA_PREFILL_MERGE_FIRST","STRATA_PCIE_BALANCE","STRATA_IO_LOOKAHEAD","STRATA_FS","STRATA_FS_SLOTS","STRATA_PIPELINE_AUTO","STRATA_PREFILL_EQUAL","STRATA_PREFILL_PIPE"})need(inactive(k),k);
 for(auto k:{"STRATA_ROUTE_TAIL_SKIP","STRATA_SWIFT_NVFP4_ACTSTAGE_V264"}){auto*s=std::getenv(k);need(s&&!std::strcmp(s,"0"),k);}
}
template<class O>void options(const O&o){using four_reference_v11::need;environment();
 need(o.serve&&!o.no_capture&&!o.no_token_graph&&!o.no_pool&&!o.no_ple&&!o.no_fused_gr&&!o.no_fast_attn&&!o.no_publish_kernel&&!o.no_fused_gdn&&!o.no_fast_select,"original captured native serve required");
 need(o.adapt_swaps==0&&!o.adapt_async&&!o.pipeline_windows&&!o.short_read,"fixed0/no overlap/short-read first scope; adapt72 not qualified");
 need(o.batch==0&&o.batch_groups==1&&!o.vision&&!o.spec_split,"two dense N1 ordinary windows only");
 need(o.pcie_frac==0.0&&!o.pcie_frac_auto&&o.peer_device<0&&!o.remote_expert_opt,"no old transport/peer cache");
 need(!o.expert_cache_remote[0]&&!o.expert_cache_remote[1]&&!o.expert_cache_remote[2],"no old helper cache");
 need(!o.prompt_cache&&!o.prompt_cache_every&&!o.conversation_cache_slots&&!o.conversation_cache_mib,"one fresh GEN/no reuse");
 need(o.prefill_pipe<=0.0&&!o.kv_grow&&!o.vram_elastic&&!o.prefill_head_alias&&!o.prefill_auto&&o.no_prefill_borrow&&!o.prefill_until,"independent fixed PP workspace only");
 need(!o.mmap_experts&&!o.resident_cpu_experts&&!o.mtp.empty()&&!o.expert_profile.empty()&&o.expert_profile_save.empty(),"one original RAM arena/profile/MTP required");
 need(o.layer_split=="24"&&o.split_device=="1"&&o.pool_workers==32&&!o.split_skip_if_fits&&!o.expert_cache_per_layer,"two24 dense owners/32 original accounting workers");
 need(o.max_context==262144||o.max_context==524288||o.max_context==1048576,"selected context");
 need(o.kv=="int8"&&o.kv_resident==(o.max_context>262144?65536:32768),"unchanged selected KV policy");
 need(o.spec==8&&(o.mtp_max_t==6||o.mtp_max_t==8)&&o.suffix_draft==3&&!o.lookup_chain&&o.spec_min_p==0.5,"normalized spec8/MTP6or8/suffix3");
 need(o.max_new==1&&o.tokens.size()==1&&o.tokens.front()==248045&&o.stop_eos,"original normalized serve seed/stop_eos");
 need(o.spec_follow.empty()&&o.dump_final_r.empty(),"no unsupported serve oracle flags");
 const auto&m=four_reference_v11::reference();need(o.prefill_chunk==m.rows&&four_reference_v11::tuple(o)==m.options,"actual options/profile differ from independent matched16k/8k control");
}
}
