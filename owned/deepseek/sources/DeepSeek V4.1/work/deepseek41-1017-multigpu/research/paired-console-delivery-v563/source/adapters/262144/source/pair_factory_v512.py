from cohort_binding_v539 import WORKERS as WORKERS539, latch as latch_cohort539
"""Fresh-child factory, explicit selected DSO/SHA; no numerical work on import."""
import json
import os
from pathlib import Path
from pair_runtime_loader_v512 import load_context_runtime, FROZEN_SOURCE


def create(store, *, context, library_path, library_sha256, mtp='native4',
           pair_cache_mode='off', expert_cache_gib=0, packed_ram_gib=432,
           pair_prefill='0', gu_hidden='0', CPU_group_sum='0'):
    from serving_binding_v529 import latch as latch_serving
    latch_serving()
    latch_cohort539()
    from stage_devices_pair_v512 import activate_pair
    activate_pair()  # current NVIDIA indices0/1 BEFORE CUDA-capable imports
    root_index=json.loads((Path(__file__).resolve().parents[1]/'ROOT_INDEX_BINDING.json').read_text())
    if root_index.get('ready') is not True or root_index.get('context')!=262144:
        raise RuntimeError('Source-only index migration requires Root source/owner binding and seal')
    from adaptive_binding_v512 import latch as latch_adaptive
    latch_adaptive()
    from verify_woa_binding_v512 import check as check_woa
    check_woa()
    from prefill_seam_binding_v512 import latch,admission
    v=admission();latch()
    if context!=262144:raise ValueError('341 is the fixed256 wavefront samegeometry qualification only')
    row=v['contexts']['262144']
    if str(Path(library_path).absolute())!=row['path'] or library_sha256!=row['sha256']:raise ValueError('340 requires actual335 Engine262 DSO')
    if mtp!='native4':raise ValueError('340 qualification requires original native4')
    if any(type(v) is not str or v != '0' for v in (pair_prefill, gu_hidden)):
        raise ValueError('K14 indexer requires legacy pair_prefill and GU hidden literal0')
    if pair_cache_mode != 'off' or type(expert_cache_gib) is not int or expert_cache_gib != 0:
        raise ValueError('K14 indexer requires cache-off with exactly zero bytes')
    if mtp not in ('off', 'native4'):
        raise ValueError('K14 indexer retains only native4 or MTP-off')
    if Path(library_path).name != 'libds1017_pair_runtime_v1.so':
        raise ValueError('K14 indexer requires actual libds1017_pair_runtime_v1.so output')
    if not isinstance(library_sha256,str) or len(library_sha256)!=64 or any(c not in '0123456789abcdef' for c in library_sha256):
        raise ValueError('Explicit lowercase SHA256 of actual Root-built DSO required')
    if CPU_group_sum != '0':raise ValueError('CPU96 keeps group-sum literal0; node-local queue is a separate flag')
    from pair_format_v512 import binding
    actual_bank=binding(store)
    from serving_binding_v529 import assert_store
    if assert_store(store)!=actual_bank:raise RuntimeError('514 factory requires exact fresh inherited139 capability')
    os.environ.update(DS1017_CPU_PAIR32_LAYOUT_V143='139',DS1017_PAIR32_REGISTRY_V145='139',DS1017_PAIR32_BYTE_INVERSE_V141='1',DS1017_GPU_GU_PAIR32_OWNER_V147='0',DS1017_GPU_GU_STARTUP_SCRATCH_BYTES_V147='0',DS1017_COLD_FOOTER_MAX_V108='0')
    from cpu_projection_binding_v512 import resolve_cpu_projection
    resolve_cpu_projection()
    module=load_context_runtime(context)
    profiles=json.loads((FROZEN_SOURCE/'BASE_PROFILES.json').read_text())
    profile=next(r for r in profiles['profiles'] if r['context_capacity']==context)
    for key,value in profile['explicit_environment'].items():
        if key!='CUDA_VISIBLE_DEVICES':os.environ[key]=str(value)
    os.environ.update(DS1017_OLD_COMPLETE_RUNTIME_V1='1',DS41_ATTENTION_WOA_1017_SOURCE_PORT='1',
                      DS1017_CPU_NODE_LOCAL_QUEUE_V80='0',DS1017_GPU_GU_OVERLAP_V6='0',DS1017_GPU_GU_RESIDENT_OWNER_V19='1',
                      DS1017_GPU_GU_BYTEPERM_V13='1',DS1017_GPU_GU_BUDGET_GIB_V19='20',
                      DS1017_COMPACT_RUNTIME_FEATURES_V7='1',DS1017_PAIR_PREFILL_V1='0',
                      DS41_PREFILL_GU_HIDDEN_V41='0',DS1017_PAIR_STAGE_COPY_V4='0',
                      DS1017_PAIR_PREFILL_LOOKAHEAD_V5='0',DS1017_PAIR_INDEXER_PREFILL_V1='1',DS41_PF_PROFILE='1')
    latch()
    from adaptive_binding_v512 import latch as latch_whole
    latch_whole()
    s=profile['settings']
    if (s['chunk_tokens'],s['prefill_batch'],s['expert_batch'])!=(262144,2048,4096):raise ValueError('Original256 BASE_PROFILE provenance changed')
    kwargs=dict(model_path=store.source,workers=WORKERS539,coordinator_cpu=255,context=context,
                chunk_tokens=131072,prefill_batch=2048,expert_batch=4096,
                packed_ram_gib=packed_ram_gib,vision_cache_gib=s['vision_cache_gib'],graph=True,
                decoded_expert_cache=True,experimental=True,cpu_arithmetic=s['CPU_arithmetic'],
                upload_pipeline=True,expert_coordinator_cpu=254,dense_prefill=True,fp8_experts=True,
                expert_cache_gib=expert_cache_gib,pair_cache_mode=pair_cache_mode,weight_profile=None,
                library_name=s['library_name'],gpu_stage=0,dense_fp8=True,persistent_experts=False,
                mtp=mtp,packed_fp8_weights=True,decoder_suffix_prefill=True,compact_candidate_prefill=True,
                expert_cache_tail_rows=0,prefill_margin_mib=768,mtp_arena_upload=False,
                k3_causal_batch=True)
    if 'mtp_short_owned_pool' in module.CppRuntimeSettings.__dataclass_fields__:kwargs['mtp_short_owned_pool']=False
    runtime=module.NativeCppBundleRuntime(module.CppRuntimeSettings(**kwargs),store,
        library_path=Path(library_path).absolute(),library_sha256=library_sha256)
    from controller_metadata_compat_v512 import bind_frozen_controller_metadata
    bind_frozen_controller_metadata(library_path,library_sha256)
    return runtime
