from cohort_binding_v539 import WORKERS as WORKERS539, validate_settings as validate_cohort539, startup as startup_cohort539
from index_selection_v506 import uuids as _index_uuids
"""Private314 pure-relinked310 causal K3 on the frozen long-stream profile.

DefaultOFF. Same258 masked172 K5/Pack/CPU118/native4/cache0/prefill/cost math.
Only the library path, explicit K3 flag and readonly310 diagnostic are added.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import ctypes as C
import gc,hashlib,importlib.util,json,os,sys,tempfile,time
IMAGE_PROCESSOR=None
from pathlib import Path
from .runtime import NativeBundleRuntime,admit_production,admit_single_gpu_owner,resolve_images,UUID,ROOT
from .api_codec import MODEL_ID,ByteLevelDecoder,NativeDeltaParser,native_request,encode_messages
from .fifo import RequestError,RequestCancelled
from pair_runtime_policy_v506 import cache_settings_admitted,runtime_record
GPU0_PARALLEL_POST_LIBRARIES=('native-engine-activation-fused-v67.so','native-engine-packed-FP8-v78.so','native-engine-decoder-suffix-v83.so','native-engine-compact-candidates-v89.so','native-engine-cache-tail-v97.so','native-engine-bounded-margin-v101.so','native-engine-speed-combined-cache-host-topk-r6.so')
SUFFIX_LIBRARIES=GPU0_PARALLEL_POST_LIBRARIES[2:]
COMPACT_LIBRARIES=GPU0_PARALLEL_POST_LIBRARIES[3:]
NATIVE_LIBRARIES=('native-engine-hybrid-v20.so','native-engine-expert-device-v21.so','native-engine-expert-devices-v22.so','native-engine-expert-many-v23.so','native-engine-staged-experts-v24.so','native-engine-staged-many-v25.so','native-engine-staged-bounded-many-v26.so','native-engine-staged-bounded-experts-v27.so','native-engine-persistent-experts-v28.so','native-engine-persistent-many-v29.so','native-engine-verify-prepared-v36.so')+GPU0_PARALLEL_POST_LIBRARIES
STAGED_LIBRARIES=NATIVE_LIBRARIES[4:-len(GPU0_PARALLEL_POST_LIBRARIES)]
PERSISTENT_LIBRARIES=NATIVE_LIBRARIES[8:10]
MASK_SKIP_LIBRARY='native-engine-speed-combined-cache-host-topk-r6.so'
MARGIN_LIBRARIES=('native-engine-bounded-margin-v101.so',MASK_SKIP_LIBRARY)
@dataclass(frozen=True)
class CppRuntimeSettings:
    model_path:Path
    workers:tuple
    coordinator_cpu:int
    context:int=524288
    chunk_tokens:int=196608
    prefill_batch:int=2048
    expert_batch:int=2048
    packed_ram_gib:int=512
    vision_cache_gib:int=2
    graph:bool=True
    decoded_expert_cache:bool=True
    experimental:bool=False
    cpu_arithmetic:str="mixed-f32"
    upload_pipeline:bool=False
    expert_coordinator_cpu:int|None=None
    dense_prefill:bool=False
    fp8_experts:bool=False
    expert_cache_gib:int=0
    pair_cache_mode:str="off"
    weight_profile:str|None=None
    library_name:str="native-engine-hybrid-v20.so"
    gpu_stage:int=0
    dense_fp8:bool=False
    persistent_experts:bool=False
    mtp:str='off'
    packed_fp8_weights:bool=False
    decoder_suffix_prefill:bool=False
    compact_candidate_prefill:bool=False
    expert_cache_tail_rows:int=0
    prefill_margin_mib:int=1024
    mtp_arena_upload:bool=False
    k3_causal_batch:bool=False
    model_identity_path:Path|None=None
    model_identity_sha256:str|None=None
    mtp_binding_path:Path|None=None
    mtp_binding_sha256:str|None=None

    def __post_init__(self):
        # Controller214/404 supplies the exact stock254 carrier before factory.
        # Only that one tuple is canonicalized; all other worker lists fail below.
        if self.workers==tuple(range(254)) and self.coordinator_cpu==255 and self.expert_coordinator_cpu==254:
            object.__setattr__(self,'workers',WORKERS539)

def validate_cpp_settings(s):
    if type(s.k3_causal_batch)is not bool:raise ValueError("314 K3 feature must be explicit boolean")
    if ((s.context,s.chunk_tokens,s.prefill_batch,s.expert_batch) not in ((524288,262144,1024,2048),)
            or not cache_settings_admitted(s) or s.expert_cache_tail_rows!=0 or s.prefill_margin_mib!=512
            or s.mtp not in('off','native4') or s.mtp_arena_upload or s.persistent_experts
            or s.library_name!=MASK_SKIP_LIBRARY or s.cpu_arithmetic!='packed-parallel-post-v66'
            or not s.experimental):
        raise ValueError('257 requires258/selected246/CPU118/native4/cache0/context524288/canonicalcap196608')
    if type(s.context)is not int or s.context not in(4096,16384,131072,262144,524288):raise ValueError('Explicit context size up to524288 required')
    if not isinstance(s.workers,tuple)or not 1<=len(s.workers)<=256 or any(type(c)is not int for c in s.workers)or len(set(s.workers))!=len(s.workers):raise ValueError('Distinct explicit worker CPUs required')
    validate_cohort539(s)
    allowed=os.sched_getaffinity(0)
    if not {127,253}<=set(allowed):raise ValueError('470 requires helper127 and reserved uploader253 allowed')
    if any(c not in allowed for c in s.workers)or type(s.coordinator_cpu)is not int or s.coordinator_cpu not in allowed or s.coordinator_cpu in s.workers:raise ValueError('Separate allowed coordinator core required')
    if s.cpu_arithmetic not in("mixed-f32","a8","packed-vbmi","packed-compact","packed-local-v38","packed-parallel-post-v66"):raise ValueError("Unknown CPU arithmetic")
    if type(s.expert_cache_gib)is not int or not 0<=s.expert_cache_gib<=22:raise ValueError("Explicit bounded expert cache budget required")
    if type(s.expert_cache_tail_rows)is not int or s.expert_cache_tail_rows not in(0,512,1024,2048):raise ValueError('Choose an explicit cache tail window')
    if s.expert_cache_tail_rows and s.library_name not in GPU0_PARALLEL_POST_LIBRARIES[4:]:raise ValueError('Cache tail policy requires a preserved tail implementation')
    if type(s.prefill_margin_mib)is not int or s.prefill_margin_mib not in(512,768,1024,1536):raise ValueError('Choose an explicit generic prefill margin')
    if s.prefill_margin_mib!=1024 and s.library_name not in MARGIN_LIBRARIES:raise ValueError('Configurable margin requires the bounded-margin implementation')
    if type(s.mtp_arena_upload)is not bool or(s.mtp_arena_upload and(s.mtp!='own3' or s.cpu_arithmetic!='packed-parallel-post-v66')):raise ValueError('Arena upload requires the GPU0 own3 MTP profile')
    if s.weight_profile not in(None,"quality","balance","speed-memory","extreme"):raise ValueError("Unknown weight profile")
    if s.fp8_experts and not s.decoded_expert_cache:raise ValueError("FP8 expert prefill requires the decoded expert cache")
    extra=()if s.expert_coordinator_cpu is None else(s.expert_coordinator_cpu,)
    if extra and(type(extra[0])is not int or extra[0]not in allowed or extra[0]in s.workers or extra[0]==s.coordinator_cpu):raise ValueError("Separate allowed expert coordinator required")
    if type(s.gpu_stage)is not int or s.gpu_stage not in range(4):raise ValueError("Only GPU0..GPU3 stages are supported")
    if s.library_name not in NATIVE_LIBRARIES:raise ValueError("Choose an explicitly preserved native implementation")
    if s.gpu_stage and s.library_name not in STAGED_LIBRARIES:raise ValueError("Staged GPUs require the checked multi-device implementation")
    parallel_post=s.cpu_arithmetic=='packed-parallel-post-v66'
    if parallel_post or s.library_name in GPU0_PARALLEL_POST_LIBRARIES:
        if not parallel_post or s.library_name not in GPU0_PARALLEL_POST_LIBRARIES or s.gpu_stage!=0 or s.workers!=WORKERS539 or s.coordinator_cpu!=255 or s.expert_coordinator_cpu!=254:
            raise ValueError('V66/v67/v78 requires GPU0, exact248 workers with64..67 reserved for uploads, helper127, main255 and expert coordinator254')
        if os.environ.get('DEEPSEEK41_NUMA_POLICY')!='interleave-both':raise ValueError('V66/v67/v78 requires explicit NUMA interleave on nodes0,1')
        if not s.graph or not s.dense_fp8 or not s.fp8_experts or not s.upload_pipeline or not s.dense_prefill or not cache_settings_admitted(s):
            raise ValueError('V66/v67/v78 requires measured graph/FP8/prefill switches and expert cache0/2/4 GiB')
    if s.packed_fp8_weights and s.library_name not in GPU0_PARALLEL_POST_LIBRARIES[1:]:raise ValueError('Packed FP8 weights require explicitly selected v78')
    if s.decoder_suffix_prefill and (s.library_name not in SUFFIX_LIBRARIES or not s.packed_fp8_weights):raise ValueError('Decoder suffix requires explicit v83 + packed FP8 weights')
    if s.compact_candidate_prefill and s.library_name not in COMPACT_LIBRARIES:raise ValueError('Compact candidate metadata requires explicit v89')
    if s.persistent_experts and (not s.expert_cache_gib or s.library_name not in PERSISTENT_LIBRARIES):raise ValueError("Persistent expert weights require explicit cache budget and lifetime-capable native library")
    if s.mtp not in('off','native4','own3'):raise ValueError('Unknown GPU0 MTP mode')
    if s.mtp!='off' and not parallel_post and (s.gpu_stage!=0 or s.library_name!='native-engine-verify-prepared-v36.so' or s.cpu_arithmetic!='packed-local-v38' or len(s.workers)!=254 or s.coordinator_cpu!=255 or s.expert_coordinator_cpu!=254 or not s.graph or not s.dense_fp8 or not s.fp8_experts or not s.upload_pipeline or not s.dense_prefill or not cache_settings_admitted(s)):raise ValueError('MTP requires GPU0 target/native4 and a cache mode admitted by the private pair policy')
    if s.mtp=='own3' and not s.experimental:
        folder=Path(s.model_path).resolve();manifest=folder/'drafts/own3.manifest.json'
        if not manifest.is_file() or manifest.is_symlink():raise ValueError('Production own3 requires its standalone bundled head')
        head=json.loads(manifest.read_text());payload=folder/'drafts'/head.get('file','')
        if head.get('file')!='own3.bin' or head.get('bits')!=3 or head.get('model_bundle_sha256')!=hashlib.sha256((folder/'model.bundle.json').read_bytes()).hexdigest() or not payload.is_file() or payload.is_symlink() or payload.stat().st_size!=head.get('payload_bytes'):raise ValueError('Standalone own3 head differs from the main model')

    physical=[]
    for cpu in s.workers+(s.coordinator_cpu,)+extra:
        root=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology';physical.append(((root/'physical_package_id').read_text(),(root/'core_id').read_text()))
    if len(set(physical))!=len(physical):raise ValueError('Physical cores required, without SMT aliases')
    for value,lo,hi in[(s.chunk_tokens,1,1048576),(s.prefill_batch,1,4096),(s.expert_batch,1,4096),(s.packed_ram_gib,1,640),(s.vision_cache_gib,0,6)]:
        if type(value)is not int or not lo<=value<=hi:raise ValueError('Explicit bounded native runtime budgets required')
    if any(type(v)is not bool for v in(s.graph,s.decoded_expert_cache,s.experimental,s.upload_pipeline,s.dense_prefill,s.fp8_experts,s.dense_fp8,s.persistent_experts,s.packed_fp8_weights,s.decoder_suffix_prefill,s.compact_candidate_prefill)):raise ValueError('Native switches must be booleans')

def cpp_configuration(s):
    compiled_kernel_settings={'DS41_K5_MASK_SKIP_V172':int(s.library_name==MASK_SKIP_LIBRARY)}
    native_environment={'DS41_NATIVE_K3_CAUSAL_BATCH_V310':'1'if s.k3_causal_batch else'0','DS41_NATIVE_PACKED_FP8_WEIGHTS':'1'if s.packed_fp8_weights else'0','DS41_NATIVE_DECODER_SUFFIX':'1'if s.decoder_suffix_prefill else'0','DS41_NATIVE_COMPACT_CANDIDATES':'1'if s.compact_candidate_prefill else'0','DS41_NATIVE_CACHE_TAIL_ROWS':str(s.expert_cache_tail_rows),'DS41_NATIVE_PREFILL_MARGIN_MIB':str(s.prefill_margin_mib),'DS41_NATIVE_WOA_MANY':'0','DS41_NATIVE_PREFILL_STREAM_DENSE':'1','DS41_NATIVE_OUTPUT16_COMPACT_V252':'1','DS41_NATIVE_PREFILL_EXPECT_CAP_V252':str(s.chunk_tokens),'DS41_FP8_SCALED_V175':'0'}
    native_environment.update({'DS41_NATIVE_K3_CAUSAL_BATCH_V310': '1', 'DS41_NATIVE_PACKED_FP8_WEIGHTS': '1', 'DS41_NATIVE_DECODER_SUFFIX': '1', 'DS41_NATIVE_COMPACT_CANDIDATES': '1', 'DS41_NATIVE_CACHE_TAIL_ROWS': '0', 'DS41_NATIVE_PREFILL_MARGIN_MIB': '512', 'DS41_NATIVE_WOA_MANY': '0', 'DS41_NATIVE_PREFILL_STREAM_DENSE': '1', 'DS41_NATIVE_OUTPUT16_COMPACT_V252': '1', 'DS41_NATIVE_PREFILL_EXPECT_CAP_V252': '262144', 'DS41_FP8_SCALED_V175': '0', 'DS41_INDEXER_S8_R1': '1', 'DS41_ENGRAM_GATHER_WORKERS_R1': '8', 'DS41_SUFFIX_CANDIDATE_STORAGE_R1': '1', 'DS41_RECOMPUTE_INPUTS_R1': '1', 'DS41_MAPPED_EMBEDDING_R1': '1', 'DS41_ENGRAM_HASH_DEDUP_R1': '1', 'DS41_NATIVE_PREFILL_PIPELINE': '1', 'DS41_NATIVE_PREFILL_PARALLEL_STAGE_COPY': '1', 'DS41_NATIVE_TINY_PREFILL_CPU_R3': '0', 'DS41_NATIVE_HOST_TOPK_R6': '1', 'DS41_NATIVE_HOST_TOPK_R7': '0'})
    native_environment['DS41_INDEXER_S8_R1']='1'
    native_environment['DS41_ENGRAM_GATHER_WORKERS_R1']='8'
    native_environment.update(DS41_SUFFIX_CANDIDATE_STORAGE_R1='1',DS41_RECOMPUTE_INPUTS_R1='1',DS41_MAPPED_EMBEDDING_R1='1',DS41_ENGRAM_HASH_DEDUP_R1='1')
    native_environment.update(DS1017_PAIR_INDEXER_PREFILL_V1='0',DS1017_PAIR_PREFILL_V1='0',DS41_PREFILL_GU_HIDDEN_V41='0',DS1017_PAIR_STAGE_COPY_V4='0',DS1017_PAIR_PREFILL_LOOKAHEAD_V5='0')
    native_environment['DS1017_CANDIDATE_CUTOFF_V375']='1'
    from adaptive_binding_v506 import FLAGS as ADAPTIVE_FLAGS
    native_environment.update(ADAPTIVE_FLAGS)
    native_environment['DS41_PF_PROFILE']='1' # private V10 diagnostic, same flag as V8 baseline
    return dict(profiler_enabled=True,engram_parallel_gather_r1=dict(workers=8,physical_cpus=[0,32,68,96,128,160,192,224],original_decoder=True),indexer_S8_r1='lossless ratio1, exact BF16 fallback, bounded extra workspace, original256 score rows',engram_hash_dedup_r2='bounded2GiB',protected_ID_cache_r1=True,embedding_placement_r1="BF16 mapped pinned host, read-only inference, head GPU; +1323827200 host bytes",recompute_inputs_r1=True,suffix_candidate_storage_r1=True,K3_causal_batch_v310=dict(schema=310,enabled=s.k3_causal_batch,origin='validated310 causal overlay;314 preserves258 masked172 K5',new_GPU_scratch_bytes=0,graph_reuse=True,legacy_native_WOA_many=False,verify_WOA449='qualified FP8 M2..6 many116; M1 scalar; successful-window counters not launches',verify_WOA_quant456='same449 eligible windows; per-window quant_many before WO_A, not device-launch counters',ratio0_and_M1_attention_unchanged=True),prefill_dense_stream=dict(schema=190,enabled=True,banks=2,scope='ordinary-layer-weights',base='stable embed/head/norm/Engram',restore_before_decode=True),index_mask_skip=s.library_name==MASK_SKIP_LIBRARY,compiled_kernel_settings=compiled_kernel_settings,CPU_activation_prequantization='down-exact-post-v118',backend='native-cpp',torch_allocator_config=os.environ.get('PYTORCH_ALLOC_CONF','default'),context_capacity=s.context,chunk_tokens=s.chunk_tokens,prefill_batch=s.prefill_batch,expert_batch=s.expert_batch,packed_ram_gib=s.packed_ram_gib,vision_cache_gib=s.vision_cache_gib,physical_cpu_workers=list(s.workers),coordinator_cpu=s.coordinator_cpu,cpu_affinity=sorted(os.sched_getaffinity(0)),numa_policy=os.environ.get('DEEPSEEK41_NUMA_POLICY','unknown'),gpu_uuid=UUID,gpu_stage=s.gpu_stage,library_name=s.library_name,dense_fp8=s.dense_fp8,packed_fp8_weights=s.packed_fp8_weights,decoder_suffix_prefill=s.decoder_suffix_prefill,compact_candidate_prefill=s.compact_candidate_prefill,expert_cache_tail_rows=s.expert_cache_tail_rows,prefill_margin_mib=s.prefill_margin_mib,mtp_arena_upload=s.mtp_arena_upload,native_environment=native_environment,implementation_settings_sha256=hashlib.sha256(json.dumps(dict(native_environment=native_environment,compiled_kernel_settings=compiled_kernel_settings),sort_keys=True,separators=(',',':')).encode()).hexdigest(),persistent_experts=s.persistent_experts,expert_cache_policy='first-request-fixed'if s.persistent_experts else'per-request-tail-frequency'if s.expert_cache_tail_rows else'per-request-frequency',graph=s.graph,decoded_expert_cache=s.decoded_expert_cache,CPU_arithmetic=s.cpu_arithmetic,upload_pipeline=s.upload_pipeline,dense_prefill=s.dense_prefill,fp8_experts=s.fp8_experts,expert_cache_gib=s.expert_cache_gib,expected_weight_profile=s.weight_profile,expert_coordinator_cpu=s.expert_coordinator_cpu,kv_cache_type='bf16',mtp=s.mtp,mtp_sampling_policy='greedy-only; other sampling uses ordinary target',mtp_cost_prior=None if s.mtp=='off'else dict(source='tools/native_runtime_mtp_cost_v198.py:initial_costs',policy='378+382 reconstructed warm target keys; conservative unmeasuredM5; reused-only EMA'),prefix_cache_reused=False,proposal_rope_phase='runtime startup before target prefill'if s.cpu_arithmetic=='packed-parallel-post-v66'else'legacy',torch_blas_workspace_phase='runtime startup before target prefill'if s.cpu_arithmetic=='packed-parallel-post-v66'else'legacy')


def _dense_stream_info_v191(engine):
    values=(C.c_uint64*24)()
    fn=engine.lib.ds41_native_engine_dense_stream_info_v191
    fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];fn.restype=C.c_int
    if fn(engine.handle,values,24):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
    return dict(zip(('schema', 'enabled', 'active', 'begin_count', 'restore_count', 'base_alloc', 'layer_alloc', 'bank_capacity', 'current_GPU_banks', 'current_pinned_banks', 'current_dense_GPU', 'last_reclaimed_GPU', 'layer_upload_bytes', 'layer_upload_count', 'prefetch_calls', 'activate_calls', 'retire_calls', 'restore_bytes', 'restore_tensor_copies', 'free_before_begin', 'free_after_begin', 'free_after_restore', 'scratch_peak', 'failed'),map(int,values)))

class NativeCppBundleRuntime:
    _sample=NativeBundleRuntime._sample
    def __init__(self,settings,borrowed_store=None,*,library_path,library_sha256):
        from bootstrap_pair_v506 import initialize_pair as initialize
        initialize(self,settings,borrowed_store,library_path,library_sha256,sys.modules[__name__])
    def output16_info(self,engine,*,pair145_before,wavefront_info):
        values=(C.c_uint64*16)()
        fn=engine.lib.ds41_native_engine_output16_info_v252
        fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];fn.restype=C.c_int
        if fn(engine.handle,values,16):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
        keys=('schema','registered_matrices','registered_compact_bytes','compact_mode_enabled',
            'prefill_compact_projection_uploads','prefill_compact_H2D_bytes','prefill_compact_FP8_encodes',
            'last_staging_capacity','maximum_compact_triple_bytes','prefill_raw_projection_uploads',
            'cache_compact_entries','cache_compact_H2D_bytes','cache_inverse_original_row_bytes',
            'cache_max_device_temporary_bytes','last_actual_prefill_cap','last_actual_prefill_sub')
        info=dict(zip(keys,map(int,values)))
        if (info['schema']!=252 or info['registered_matrices']!=0
                or info['registered_compact_bytes']!=0 or info['compact_mode_enabled']!=1
                or info['prefill_raw_projection_uploads']!=0
                or info['prefill_compact_projection_uploads']!=info['prefill_compact_FP8_encodes']
                or info['maximum_compact_triple_bytes']>info['last_staging_capacity']
                or info['last_staging_capacity']!=37602048):
            raise ValueError('Actual compact owner/feeding/staging differs from252 contract')
        paired=self.registry.pair32_info();upload=self.registry.pair32_upload_info()
        from pair_format_v506 import binding
        b=binding(self.registry.store)
        if paired!=(145,139,46080,305764761600,b['owner_generation'],b['format_epoch'],1,0):raise ValueError('actual paired145 registry accounting differs')
        if len(pair145_before)!=12 or pair145_before[:2]!=(145,139) or any(pair145_before[10:]) or any(upload[i]<pair145_before[i]for i in range(2,6)):raise ValueError('paired145 cumulative counter baseline differs')
        delta=tuple(upload[i]-pair145_before[i]if i in range(2,6)else upload[i]for i in range(12))
        if (upload[0:2]!=(145,139) or delta[2]!=2 or delta[3]!=info['prefill_compact_projection_uploads'] or delta[4]!=info['prefill_compact_H2D_bytes'] or delta[5]!=delta[4] or upload[6]!=19906560 or upload[7]!=info['last_actual_prefill_cap'] or upload[8]!=info['last_actual_prefill_sub'] or upload[9]!=19906560 or any(upload[10:])):raise ValueError('actual paired145 dual-rank inverse/upload counters differ')
        if wavefront_info['after_prefill']['GPU0_layers']!=40 or wavefront_info['after_prefill']['GPU1_layers']!=40:raise ValueError('actual two325 consumers required before dual-rank145 acceptance')
        info['prefill_counter_scope']='252 Engine0 counters include explicit325 merge of actual Engine1 projection uploads/encodes;145 registry counters are shared across both consumers'
        info['paired145_actual_prefill_consumers']=delta[2]
        info['paired145_registry']=list(paired);info['paired145_upload']=list(upload);info['paired145_upload_before']=list(pair145_before);info['paired145_upload_delta']=list(delta)
        return info
    def k3_causal_info_v310(self,engine):
        values=(C.c_uint64*7)();fn=engine.lib.ds41_native_engine_k3_causal_batch_info_v310
        fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];fn.restype=C.c_int
        if fn(engine.handle,values,7):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
        keys=('schema','enabled','workspace_created','successful_verify_calls','scheduled_batch_dispatches_on_success','batch_query_rows_on_success','reused_target_verify_calls')
        info=dict(zip(keys,map(int,values)))
        inactive_serial=(info['workspace_created']==0 and all(info[k]==0 for k in ('enabled','successful_verify_calls','scheduled_batch_dispatches_on_success','batch_query_rows_on_success','reused_target_verify_calls')))
        if info['workspace_created']==0 and not inactive_serial:raise ValueError('Verifier counters without a verifier workspace')
        if info['schema']!=310 or (not inactive_serial and info['enabled']!=int(self.settings.k3_causal_batch)):raise ValueError('Actual310 K3 activation differs from settings')
        info['lazy_verifier_not_created_for_this_request']=inactive_serial
        info['counters_increment_after_successful_stream_completion_not_at_capture']=True
        info['hardware_kernel_time_claimed']=False
        return info
    def configuration(self):
        from pair_format_v506 import binding
        return {'bank_format_binding_v146':binding(self.registry.store),**cpp_configuration(self.settings),'pair_runtime':self._pair_policy.record(self.settings),'RAM_layout':'new1017-one-paired32-bank-v150','raw_expert_resident':False,'compact_expert_encoding':'paired32-lossless-layout139','query175_enabled':False,'index_mask_library_sha256':self.implementation[str(self.library)] if self.settings.library_name==MASK_SKIP_LIBRARY else None,'model':MODEL_ID,'model_path':str(self.path),'model_bundle_sha256':self.bundle_sha256,'weight_profile':self.manifest['recipe']['profile'],'main_quant_policy':self.manifest['recipe']['policy'],'MTP_model_binding':self.mtp_binding.configuration(),'implementation_sha256':self.implementation,'api_key_required':False,'host_topk_shadow_r6':True,'topk_shadow_math':'unchanged INT32 indices copied without conversion'}
    def run(self,body,cancel,emit):
        with self._pair_policy.request_scope(self.settings):
            from serving_binding_v529 import check as serving_check
            from serving_validation_v529 import Request528
            serving528=Request528(self.library,self.settings,serving_check())
            return self._run_pair_body(body,cancel,emit,serving528)
    def _run_pair_body(self,body,cancel,emit,serving528):
        if any(os.environ.get(k,"0")!=v for k,v in {'DS41_NATIVE_PREFILL_PIPELINE': '1', 'DS41_NATIVE_PREFILL_PARALLEL_STAGE_COPY': '1', 'DS41_NATIVE_TINY_PREFILL_CPU_R3': '0', 'DS41_NATIVE_HOST_TOPK_R6': '1', 'DS41_NATIVE_HOST_TOPK_R7': '0'}.items()):raise ValueError("Exact delivery environment changed")
        if os.environ.get("DS41_NATIVE_HOST_TOPK_R6")!="1":raise ValueError("Host topk flag changed between requests")
        if os.environ.get('DS41_MAPPED_EMBEDDING_R1')!='1':raise ValueError('Explicit mapped embedding profile required')
        if os.environ.get('DS41_SUFFIX_CANDIDATE_STORAGE_R1')!='1' or os.environ.get('DS41_RECOMPUTE_INPUTS_R1')!='1':raise ValueError('Explicit lossless memory features required')
        if os.environ.get("DS41_NATIVE_K3_CAUSAL_BATCH_V310")!=("1"if self.settings.k3_causal_batch else"0"):raise ValueError("314 K3 environment differs from explicit settings")
        if (os.environ.get('DS41_NATIVE_OUTPUT16_COMPACT_V252')!='1' or os.environ.get('DS41_NATIVE_PREFILL_STREAM_DENSE')!='1'
                or os.environ.get('DS41_NATIVE_PREFILL_EXPECT_CAP_V252')!=str(self.settings.chunk_tokens)
                or os.environ.get('DS41_FP8_SCALED_V175')!='0' or os.environ.get('DS41_NATIVE_WOA_MANY')!='0'):
            raise ValueError('Explicit258 compact/stream/cap/queryOFF/WOAOFF environment required')
        self.reference.store.validate_selected_owner()
        import serving_validation_v529 as sv
        if self.settings.cpu_arithmetic=='packed-parallel-post-v66':
            from native_engine_pair_v506 import NativeCppEngine
        elif self.settings.cpu_arithmetic=='packed-local-v38':
            from native_cpp_mapped_vector_output_v38 import NativeCppEngine
        elif self.settings.cpu_arithmetic=='packed-compact':
            from native_cpp_mapped_packedcompact import NativeCppEngine
        elif self.settings.cpu_arithmetic=='packed-vbmi':
            from native_cpp_mapped_packedvbmi import NativeCppEngine
        elif self.settings.cpu_arithmetic=='a8':
            from native_cpp_mapped_a8_own_fast import NativeCppEngine
        else:
            from native_cpp_mapped_mixed_v1 import NativeCppEngine
        from native_cpp_prefill_fast import FastPrefill
        from native_cpp_vision import NativeImageMixin,encode_image_features
        import tvm_ffi
        from native_cpp_main_recent import NativeMainMixin
        from native_cpp_verify_prepared import NativeVerifyMixin
        class Engine(NativeMainMixin,NativeVerifyMixin,NativeImageMixin,NativeCppEngine):
            def verify(self,tokens,position):
                old=os.sched_getaffinity(0)
                try:
                    os.sched_setaffinity(0,self.control_cpus)
                    return NativeVerifyMixin.verify(self,tokens,position)
                finally:os.sched_setaffinity(0,old)
            enable_gpu_prefill=FastPrefill.enable_gpu_prefill
            decoded_expert_cache=FastPrefill.decoded_expert_cache
            prefill_shape=FastPrefill.prefill_shape
            prefill=FastPrefill.prefill
        try:request=native_request(body,context_capacity=self.settings.context)
        except (ValueError,TypeError,KeyError)as error:raise RequestError(str(error))from error
        s=self.settings;ref=self.reference;torch=self.torch;started=time.perf_counter();emitted=[];first_at=last_at=None;configuration=self.configuration()
        decoder=ByteLevelDecoder(ref.tokenizer);parser=NativeDeltaParser(request['thinking_mode'])
        def output(token):
            nonlocal first_at,last_at
            if first_at is None:first_at=time.perf_counter()
            last_at=time.perf_counter();emitted.append(token)
            for delta in parser.feed(decoder.token(token)):
                if not emit({'event':'delta','delta':delta}):cancel.set()
        old_affinity=os.sched_getaffinity(0)
        try:
            with tempfile.TemporaryDirectory(prefix='deepseek41-image-')as folder,torch.inference_mode(),tvm_ffi.use_torch_stream():
                if cancel.is_set():raise RequestCancelled('Cancelled before prefill')
                messages=resolve_images(request['messages'],folder)
                try:prompt,media=encode_messages(messages,thinking_mode=request['thinking_mode'],reasoning_effort=request['reasoning_effort'],return_multi_modal_data=True)
                except (ValueError,AssertionError,KeyError,TypeError)as error:raise RequestError(str(error))from error
                positions=features=None;vision_started=time.perf_counter()
                if media['images']:
                    spec=importlib.util.spec_from_file_location('native_bundle_image_processor',IMAGE_PROCESSOR);processor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=processor;spec.loader.exec_module(processor)
                    ids,type_ids,images=processor.prepare_vl_inputs(prompt,media['images'],ref.tokenizer,ref.graph.ModelArgs(**self.config))
                    positions,features=encode_image_features(ref,images,len(ids))
                else:ids=ref.tokenizer.encode(prompt,add_special_tokens=False)
                vision_seconds=time.perf_counter()-vision_started if positions is not None else 0.
                if not ids or len(ids)+request['max_new_tokens']>s.context:raise RequestError('Prompt plus output exceeds configured context')
                prefill_plan=serving528.prepare(len(ids),images=positions is not None)
                configuration['actual_prefill_plan']=dict(prefill_plan)
                if not emit({'event':'usage','prompt_tokens':len(ids)}):raise RequestCancelled('Client disconnected before prefill')
                ref.ops.weight_cache.clear();gc.collect();torch.cuda.synchronize(ref.device);torch.cuda.empty_cache()
                request_pool_trim=None
                if s.cpu_arithmetic=='packed-parallel-post-v66':
                    from request_memory_pair_v506 import trim_unused,pool_snapshot
                    request_pool_trim=trim_unused(self.library,torch)
                memory_before_engine=dict(free_bytes=torch.cuda.mem_get_info(0)[0],Torch_allocated=torch.cuda.memory_allocated(0),Torch_reserved=torch.cuda.memory_reserved(0))
                pair_device_memory_before_engine={str(i):dict(uuid=(_index_uuids()[0] if i==0 else _index_uuids()[1]),free_bytes=int(torch.cuda.mem_get_info(i)[0]),total_bytes=int(torch.cuda.mem_get_info(i)[1])) for i in (0,1)}
                os.sched_setaffinity(0,set(s.workers)|{127,253,s.coordinator_cpu}|(set()if s.expert_coordinator_cpu is None else{s.expert_coordinator_cpu}));load_started=time.perf_counter()
                with Engine(self.registry,self.library,pair_policy=self._pair_policy,cpus=s.workers,context=s.context,graph=s.graph,**({'a8':True,'projection_revision':1}if s.cpu_arithmetic in('a8','packed-compact','packed-vbmi','packed-local-v38','packed-parallel-post-v66')else{'mixed_f32':True}))as engine:
                    self._cohort539 = startup_cohort539(engine.lib)
                    configuration['CPU_reserved_upload_cohort539'] = self._cohort539
                    os.sched_setaffinity(0,{s.coordinator_cpu});engine.enable_gpu_prefill(expert_batch=prefill_plan['E'],alias_hc=True);engine.decoded_expert_cache(s.decoded_expert_cache)
                    self._pair_policy.assert_current(self.settings)
                    cache=engine.lib.ds41_native_engine_expert_cache;cache.argtypes=[C.c_void_p,C.c_size_t];cache.restype=C.c_int
                    if cache(engine.handle,s.expert_cache_gib*1024**3):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
                    engine.configure_gu_v19()
                    expert_weights_reused=self.expert_pool.attach(engine)if self.expert_pool is not None else False
                    if positions is not None:engine.image_inputs(positions,features)
                    use_mtp=s.mtp!='off' and request['temperature']==0 and request['repetition_penalty']==1 and len(ids)>=6
                    mtp_metadata=dict(mode='off',requested_mode=s.mtp)
                    if use_mtp or s.cpu_arithmetic=='packed-parallel-post-v66':
                        engine.control_cpus={s.coordinator_cpu,s.expert_coordinator_cpu}
                        engine.capture_main()
                        for rows in range(1,7):engine.prepare_verify_cpu(rows)
                    pair_indexer_before=engine.pair_indexer_counters()
                    pair145_before=self.registry.pair32_upload_info()
                    load_seconds=time.perf_counter()-load_started;memory_before_prefill=dict(free_bytes=torch.cuda.mem_get_info()[0],Torch_allocated=torch.cuda.memory_allocated(),Torch_reserved=torch.cuda.memory_reserved());prefill_started=time.perf_counter()
                    from wavefront_counters_v506 import read,validate_before,validate_after,unchanged,validate_old_helper_off
                    from prefill_seam_counters_v506 import read as read_passes,validate_before as validate_passes_before,validate_after as validate_passes_after,unchanged as unchanged_passes
                    from k14_register_counters_v506 import read as read_k14,before as before_k14,validate as validate_k14,unchanged as unchanged_k14
                    k14_before=read_k14(engine.lib);sv.k14_before(k14_before)
                    from early_attention_counters_v506 import read as read_early,validate_before as validate_early_before,validate_after as validate_early_after,unchanged as unchanged_early
                    from rank_copy_counters_v506 import read as read_copy,validate_before as validate_copy_before,validate_after as validate_copy_after,unchanged as unchanged_copy,validate_projection_totals
                    from candidate_cutoff_counters_v506 import read as read_cutoff,validate as validate_cutoff
                    from verify_woa_counters_v506 import read as read_verify_woa,delta as verify_woa_delta
                    from verify_woa_quant_counters_v506 import read as read_quant456,delta as delta_quant456
                    verify_woa_delta=sv.verify_delta;delta_quant456=sv.verify_delta
                    quant456_info=dict(before_prefill=read_quant456(engine.lib,'before-prefill',absent=True))
                    verify_woa_info=dict(before_prefill=read_verify_woa(engine.lib,'before-prefill',absent=True))
                    cutoff_initial=read_cutoff(engine);validate_cutoff(cutoff_initial,initial=True)
                    cutoff_info=dict(schema=375,scope='actual typed Handle device last-call state; not cumulative counters',before_first_decode=cutoff_initial)
                    early_before=read_early(engine.lib);validate_early_before(early_before)
                    copy_before=read_copy(engine.lib);validate_copy_before(copy_before)
                    seam_before=read(engine.lib);validate_before(seam_before)
                    passes_before=read_passes(engine.lib);validate_passes_before(passes_before)
                    dense_stream_before=_dense_stream_info_v191(engine)
                    self._pair_policy.assert_current(self.settings)
                    _,logits,_,_=engine.prefill(ids,chunk=prefill_plan['cap'],batch=prefill_plan['B'],score_targets=False);torch.cuda.synchronize(ref.device);prefill_ended=time.perf_counter();shape=engine.prefill_shape()
                    serving528_info=serving528.after_prefill(rows=len(ids),shape=shape);serving_wavefront=serving528_info['mode']==2
                    verify_woa_info['after_prefill']=read_verify_woa(engine.lib,'after-prefill',allow_absent=True)
                    quant456_info['after_prefill']=read_quant456(engine.lib,'after-prefill',allow_absent=True)
                    cutoff_info['after_prefill']=read_cutoff(engine);validate_cutoff(cutoff_info['after_prefill'])
                    seam_after_prefill=read(engine.lib);passes_after_prefill=read_passes(engine.lib)
                    k14_after_prefill=read_k14(engine.lib);early_after_prefill=read_early(engine.lib);copy_after_prefill=read_copy(engine.lib)
                    generic512_info=sv.generic512(engine.lib,plan=prefill_plan)
                    seam_info=dict(before_prefill=seam_before,after_prefill=seam_after_prefill,actual_native528=generic512_info)
                    passes_info=sv.generic317(passes_before,passes_after_prefill,plan=prefill_plan,shape=shape)
                    k14_info=sv.ordinary_k14(k14_before,k14_after_prefill,prompt_tokens=len(ids))
                    early_info=dict(before_prefill=early_before,after_prefill=early_after_prefill,scope='actual337 immutable528 geometry')
                    copy_info=dict(before_prefill=copy_before,after_prefill=copy_after_prefill,scope='actual331 dynamic copy counts; original per-projection bytes')
                    seam_info['original_passes_v317']=passes_info
                    seam_info['early_attention_v337']=early_info;seam_info['rank_parallel_stage_copy_v331']=copy_info
                    gu_before_v19=engine.gu_statistics_v19()
                    engine.require_no_gu147()
                    from adaptive_gap_v506 import GapControl386
                    bank=engine.owner.bank_format
                    gap_control=GapControl386(engine,generation=bank['owner_generation'],format_epoch=bank['format_epoch'],adaptive=False)
                    engine._adaptive_control=gap_control
                    bulk_info=dict(after_prefill=engine.bulk418_info())
                    epoch_info=dict(after_prefill=engine.active_epoch427_info())
                    from generation_off_counters_v506 import read as read_task489,validate as validate_task489,delta as delta_task489
                    task489_info=dict(after_prefill=read_task489('after-prefill'))
                    adaptive_info=dict(after_prefill=gap_control.snapshot(),scope='actual387 Handle whole370 receipt; sameDSO OFF/ON; no147')
                    from fixed_counters_v506 import read as read_fixed469
                    fixed469_info=dict(after_prefill=read_fixed469(engine.lib,bank=bank,phase='after-prefill'))
                    from adaptive_validation_v506 import validate_receipt
                    from capacity_policy_v506 import sample as sample_capacity,validate_memory
                    from transport_model_v506 import read as read_transport,validate as validate_transport,delta as transport_delta
                    validate_transport=lambda raw,whole,**kw:sv.transport_validate(serving528.transport_request(raw),whole,**kw);transport_delta=sv.transport_delta
                    from boundary_receipt_v506 import save as save_boundary
                    capacity_memory=dict(postprime_after_prefill=sample_capacity(torch,'postprime_after_prefill'))
                    transport_info=dict(after_prefill=read_transport(engine.lib))
                    transport_info['raw_after_prefill']=save_boundary('after-prefill',adaptive_info['after_prefill'],transport_info['after_prefill'],capacity_memory['postprime_after_prefill'],task489_info['after_prefill'])
                    validate_task489(task489_info['after_prefill'])
                    validate_receipt(adaptive_info['after_prefill'],bank=bank)
                    validate_transport(transport_info['after_prefill'],adaptive_info['after_prefill'],startup=True)
                    validate_memory(capacity_memory['postprime_after_prefill'])
                    print(json.dumps(dict(actual_whole370_after_prefill=adaptive_info)),flush=True)
                    dense_stream_after=_dense_stream_info_v191(engine)
                    dense_stream_after['scope']='GPU0 own190 dense banks only; GPU1 headless closure and reserve are actual325 counters'
                    pair_indexer_after=engine.pair_indexer_counters()
                    pair_indexer_info=sv.helper_off528(pair_indexer_before,pair_indexer_after,self._pair_policy,s,plan=prefill_plan)
                    print(json.dumps(dict(pair_indexer_prefill_v1=pair_indexer_info)),flush=True)
                    output16_info=sv.output16_generic(self,engine,pair145_before=pair145_before,shape=shape,plan=prefill_plan,direct=serving528.direct531_info)
                    if serving_wavefront:
                        c=list(copy_after_prefill.values());sv.need(c[4]+c[5]==output16_info['prefill_compact_projection_uploads'] and c[8]+c[9]==output16_info['prefill_compact_H2D_bytes'],'actual331 copies ==252 uploads/H2D; inverse/direct split independently proved')
                    if (dense_stream_after['schema']!=190 or dense_stream_after['enabled']!=1 or dense_stream_after['active'] or dense_stream_after['failed']
                            or dense_stream_after['current_GPU_banks'] or dense_stream_after['current_pinned_banks']
                            or dense_stream_after['begin_count']!=1 or dense_stream_after['restore_count']!=1
                            or dense_stream_after['base_alloc']!=dense_stream_before['base_alloc']
                            or dense_stream_after['layer_alloc']!=dense_stream_before['layer_alloc']):
                        raise ValueError('Dense streaming did not restore original ordinary/base weights before decode')
                    prefill_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest()
                    saved_serial_prefill_admission_v199=None
                    suffix_info=None
                    if s.library_name in SUFFIX_LIBRARIES:
                        values=(C.c_uint64*4)();fn=engine.lib.ds41_native_engine_decoder_suffix_info;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];fn.restype=C.c_int
                        if fn(engine.handle,values,4):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
                        suffix_info=dict(zip(('enabled','passes_applied','layers_trimmed','rows_skipped'),values))
                        if values[0]!=int(s.decoder_suffix_prefill):raise ValueError('Actual decoder suffix differs from configured flag')
                    compact_info=None
                    if s.library_name in COMPACT_LIBRARIES:
                        values=(C.c_uint64*4)();fn=engine.lib.ds41_native_engine_compact_candidate_info;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];fn.restype=C.c_int
                        if fn(engine.handle,values,4):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
                        compact_info=dict(zip(('enabled','packed_rows','unpacked_rows','saved_peak_bytes'),values))
                        if values[0]!=int(s.compact_candidate_prefill):raise ValueError('Actual compact candidate flag differs')
                    if cancel.is_set():raise RequestCancelled('Cancelled during prefill')
                    cache_tail_info=prefill_memory_info=None
                    if s.library_name in GPU0_PARALLEL_POST_LIBRARIES[4:]:
                        values=(C.c_uint64*5)();fn=engine.lib.ds41_native_engine_cache_tail_info;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];fn.restype=C.c_int
                        if fn(engine.handle,values,5):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
                        cache_tail_info=dict(zip(('configured_rows','active','global_begin','global_end','counted_uses'),values))
                        if values[0]!=s.expert_cache_tail_rows:raise ValueError('Actual cache tail policy differs')
                    if s.library_name in MARGIN_LIBRARIES:
                        values=(C.c_uint64*3)();fn=engine.lib.ds41_native_engine_prefill_memory_info;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];fn.restype=C.c_int
                        if fn(engine.handle,values,3):raise RuntimeError(engine.lib.ds41_native_engine_error().decode())
                        prefill_memory_info=dict(zip(('generic_margin_mib','minimum_end_pass_free_bytes','sampled_passes'),values))
                        if values[0]!=s.prefill_margin_mib or values[2]!=shape['passes']:raise ValueError('Actual prefill margin/sample count differs')
                    self._pair_policy.assert_current(self.settings)
                    def capture_before_generation():
                        verify_woa_info['before_generation']=read_verify_woa(engine.lib,'before-generation',allow_absent=True)
                        quant456_info['before_generation']=read_quant456(engine.lib,'before-generation',allow_absent=True)
                        cutoff_info['before_generation']=read_cutoff(engine);validate_cutoff(cutoff_info['before_generation'])
                        capacity_memory['postwarm_before_generation']=sample_capacity(torch,'postwarm_before_generation')
                        transport_info['before_generation']=read_transport(engine.lib)
                        bulk_info['before_generation']=engine.bulk418_info()
                        epoch_info['before_generation']=engine.active_epoch427_info()
                        task489_info['before_generation']=read_task489('before-generation')
                        adaptive_info['before_generation']=gap_control.snapshot()
                        fixed469_info['before_generation']=read_fixed469(engine.lib,bank=bank,phase='before-generation')
                        transport_info['raw_before_generation']=save_boundary('before-generation',adaptive_info['before_generation'],transport_info['before_generation'],capacity_memory['postwarm_before_generation'],task489_info['before_generation'])
                        validate_task489(task489_info['before_generation'])
                        validate_transport(transport_info['before_generation'],adaptive_info['before_generation'],startup=True)
                        validate_memory(capacity_memory['postwarm_before_generation'])
                    if use_mtp:
                        from mtp_driver_pair_v506 import greedy_driver
                        with greedy_driver(ref,engine,mode=s.mtp,context=s.context,pending=int(logits.argmax()),arena_upload=s.mtp_arena_upload,gap_control386=None,replay_proof386=False)as(driver,mtp_metadata):
                            if cancel.is_set():raise RequestCancelled('Cancelled while loading MTP')
                            gap_control.arm() # real Handle ARM after driver warmup; no per-proposal gap callback
                            capture_before_generation()
                            driver.generate(int(logits.argmax()),request['max_new_tokens'],cancel=cancel,on_tokens=lambda values:[output(v)for v in values])
                    else:
                        gap_control.arm()
                        capture_before_generation()
                        output(self._sample(torch.from_numpy(logits).to(ref.device),request,emitted))
                        while len(emitted)<request['max_new_tokens']and emitted[-1]!=ref.tokenizer.eos_token_id and not cancel.is_set():
                            _,logits,_,_,_=engine.step(emitted[-1],len(ids)+len(emitted)-1);output(self._sample(torch.from_numpy(logits).to(ref.device),request,emitted))
                    verify_woa_info['after_generation']=read_verify_woa(engine.lib,'after-generation',allow_absent=True)
                    quant456_info['after_generation']=read_quant456(engine.lib,'after-generation',allow_absent=True)
                    verify_woa_info['generation_delta']=verify_woa_delta(verify_woa_info['before_generation'],verify_woa_info['after_generation'])
                    quant456_info['generation_delta']=delta_quant456(quant456_info['before_generation'],quant456_info['after_generation'])
                    if quant456_info['after_generation']['words'][2:]!=verify_woa_info['after_generation']['words'][2:]:raise RuntimeError('456/449 actual successful-window census differs')
                    cutoff_info['after_generation']=read_cutoff(engine)
                    cutoff_info['generation_state_advanced']=sv.cutoff_after(cutoff_info['before_generation'],cutoff_info['after_generation'],prompt_tokens=len(ids),verify_calls=verify_woa_info['generation_delta']['successful_calls'],validate=validate_cutoff)
                    if (use_mtp and mtp_metadata.get('mode')!=s.mtp)or(not use_mtp and mtp_metadata.get('mode')!='off'):raise RuntimeError('Actual MTP/sampling mode differs from selected request path')
                    mtp_metadata['requested_mode']=s.mtp
                    k14_after_generation=read_k14(engine.lib);unchanged_k14(k14_after_prefill,k14_after_generation);k14_info['after_generation']=k14_after_generation
                    seam_after_generation=read(engine.lib);unchanged(seam_after_prefill,seam_after_generation)
                    seam_info['after_generation']=seam_after_generation
                    passes_after_generation=read_passes(engine.lib);unchanged_passes(passes_after_prefill,passes_after_generation)
                    passes_info['after_generation']=passes_after_generation
                    early_after_generation=read_early(engine.lib);unchanged_early(early_after_prefill,early_after_generation);early_info['after_generation']=early_after_generation
                    copy_after_generation=read_copy(engine.lib);unchanged_copy(copy_after_prefill,copy_after_generation);copy_info['after_generation']=copy_after_generation
                    k3_info=self.k3_causal_info_v310(engine)
                    pair_gpu_cache_info=engine.pair_gpu_cache_stats()
                    gu_after_v19=engine.gu_statistics_v19()
                    engine.require_no_gu147()
                    bulk_info['after_generation']=engine.bulk418_info(completed=False)
                    from bulk_counters_v506 import validate as validate_bulk514
                    bulk_info['generation_delta']=sv.bulk_delta(bulk_info['before_generation'],bulk_info['after_generation'],validate_bulk514)
                    from bulk_counters_v506 import receipts as bulk_receipts
                    bulk_info['raw_receipts']=bulk_receipts()
                    epoch_info['after_generation']=engine.active_epoch427_info(completed=False)
                    from epoch_counters_v506 import validate as validate_epoch514,receipts as epoch_receipts
                    epoch_info['generation_delta']=sv.epoch_delta(epoch_info['before_generation'],epoch_info['after_generation'],bulk_info['generation_delta'],validate_epoch514)
                    epoch_info['raw_receipts']=epoch_receipts()
                    task489_info['after_generation']=read_task489('after-generation')
                    adaptive_info['after_generation']=gap_control.snapshot()
                    fixed469_info['after_generation']=read_fixed469(engine.lib,bank=bank,phase='after-generation',completed=False)
                    adaptive_info['gap_calls']=gap_control.record()
                    from adaptive_validation_v506 import validate_receipt
                    transport_info['after_generation']=read_transport(engine.lib)
                    transport_info['raw_after_generation']=save_boundary('after-generation',adaptive_info['after_generation'],transport_info['after_generation'],task489=task489_info['after_generation'])
                    task489_info['generation_delta']=sv.task489_delta(task489_info['before_generation'],task489_info['after_generation'],bulk_info['generation_delta'],validate_task489)
                    validate_receipt(adaptive_info['after_generation'],bank=bank,completed=False)
                    validate_transport(transport_info['after_generation'],adaptive_info['after_generation'])
                    transport_info['generation_delta']=transport_delta(transport_info['before_generation'],transport_info['after_generation'])
                    adaptive_info['terminal_close']=gap_control.close()
                    fixed469_info['terminal_absence']=read_fixed469(engine.lib,bank=bank,phase='after-close',absent=True)
                    transport_info['after_operator_close']=read_transport(engine.lib)
                    transport_info['raw_after_operator_close']=save_boundary('after-operator-close',adaptive_info['terminal_close'],transport_info['after_operator_close'])
                    validate_transport(transport_info['after_operator_close'],adaptive_info['terminal_close'],closed=True)
                    validate_receipt(adaptive_info['terminal_close'],bank=bank,closed=True)
                    adaptive_info['actual_engine_destroy_pending']=True
                    if gu_after_v19["failures"] or gu_after_v19["live_tickets"] or gu_after_v19["borrowed_views"]:raise RuntimeError("GU lease did not drain cleanly")
                    if gu_after_v19["submitted_commands"]!=gu_after_v19["released_views"]:raise RuntimeError("GU submitted/released mismatch")
                verify_woa_info['after_engine_close']=read_verify_woa(engine.lib,'after-engine-close',absent=True)
                quant456_info['after_engine_close']=read_quant456(engine.lib,'after-engine-close',absent=True)
                k14_after_close=read_k14(engine.lib);unchanged_k14(k14_after_prefill,k14_after_close);k14_info['after_close']=k14_after_close
                seam_after_close=read(engine.lib);unchanged(seam_after_prefill,seam_after_close)
                if engine.handle is not None:raise RuntimeError('328 Engine handle not closed')
                adaptive_info['actual_engine_destroyed']=True;adaptive_info['actual_engine_destroy_pending']=False
                seam_info['after_close']=seam_after_close
                passes_after_close=read_passes(engine.lib);unchanged_passes(passes_after_prefill,passes_after_close)
                passes_info['after_close']=passes_after_close
                early_after_close=read_early(engine.lib);unchanged_early(early_after_prefill,early_after_close);early_info['after_close']=early_after_close
                copy_after_close=read_copy(engine.lib);unchanged_copy(copy_after_prefill,copy_after_close);copy_info['after_close']=copy_after_close
                serving528_info=serving528.after_engine_close()
                tail=decoder.finish()
                if tail:
                    for delta in parser.feed(tail):emit({'event':'delta','delta':delta})
                message,reason=parser.finish(bool(emitted and emitted[-1]==ref.tokenizer.eos_token_id))
                memory_after_request=dict(free_bytes=torch.cuda.mem_get_info(0)[0],Torch_allocated=torch.cuda.memory_allocated(0),Torch_reserved=torch.cuda.memory_reserved(0))
                pair_device_memory_after_request={str(i):dict(uuid=(_index_uuids()[0] if i==0 else _index_uuids()[1]),free_bytes=int(torch.cuda.mem_get_info(i)[0]),total_bytes=int(torch.cuda.mem_get_info(i)[1])) for i in (0,1)}
                if s.cpu_arithmetic=='packed-parallel-post-v66':memory_after_request.update(pool_snapshot(self.library))
                result=dict(GPU1_generation_compute_on489=task489_info,fixed_plan469=fixed469_info,verify_WOA_quant_v456=quant456_info,verify_WOA_v449=verify_woa_info,CPU_active_epoch427=epoch_info,CPU_bulk418=bulk_info,static_capacity403=dict(gib=20,physical=1078,GPU1_memory=capacity_memory,transport391=transport_info),candidate_cutoff375=cutoff_info,K14_register334=k14_info,prefill_wavefront_v325=seam_info,performance_eligible=False,model_candidate_only=True,whole_adaptive370=adaptive_info,pair_gpu_gu_upload_v147=dict(enabled=False,actual_native_zero=True),pair_gpu_gu_cache_v19=dict(before_generation=gu_before_v19,after_generation=gu_after_v19,route_count_scope="actual GU offloaded route pairs; lookup hits are NOT route coverage",quality_admitted=False),pair_indexer_prefill_v1=pair_indexer_info,pair_gpu_expert_cache=pair_gpu_cache_info,K3_causal_batch_v310=k3_info,saved_serial_prefill_admission_v199=saved_serial_prefill_admission_v199,message=message,dense_stream_before_v191=dense_stream_before,dense_stream_after_v191=dense_stream_after,output16_info_v252=output16_info,index_mask_info_v172=self.index_mask_info_v172,GPU_memory=dict(before_engine=memory_before_engine,before_prefill=memory_before_prefill,after_request=memory_after_request,request_boundary_trim=request_pool_trim,pair_device_before_engine=pair_device_memory_before_engine,pair_device_after_request=pair_device_memory_after_request),generated_token_ids=list(emitted),encoded_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),encoded_prompt_token_ids_sha256=hashlib.sha256(json.dumps([int(v)for v in ids],separators=(',',':')).encode()).hexdigest(),decoder_suffix_info=suffix_info,compact_candidate_info=compact_info,cache_tail_info=cache_tail_info,prefill_memory_info=prefill_memory_info,prefill_logits_sha256=prefill_logits_sha256,finish_reason=reason,usage=dict(prompt_tokens=len(ids),completion_tokens=len(emitted),total_tokens=len(ids)+len(emitted)),timings=dict(ttft_seconds=first_at-started,prefill_seconds=prefill_ended-prefill_started,new_prompt_tps=len(ids)/(prefill_ended-prefill_started),generation_tps=max(0,len(emitted)-1)/max(last_at-first_at,1e-30),prefix_cache_reused=False,expert_weights_reused=expert_weights_reused,vision_seconds=vision_seconds,native_engine_load_seconds=load_seconds),mtp=mtp_metadata,cancelled=cancel.is_set(),runtime_configuration=configuration,actual_prefill_shape=shape,checkpoint=dict(model_bundle_sha256=self.bundle_sha256,weight_profile=self.manifest['recipe']['profile']))
                result['serving528']=serving528_info;result['fixture_comparison_applied']=False
                emit({'event':'result','result':result});print(json.dumps({'request_complete':result['usage'],'timings':result['timings'],'actual_prefill_shape':shape}),flush=True)
        finally:os.sched_setaffinity(0,old_affinity)
    def close(self):
        self.resources.close();self.registry=None;self.expert_pool=None
        if self.reference is not None and self.settings.mtp_arena_upload:
            from native_mtp_selected_arena_v251 import close_head_payloads
            close_head_payloads(self.reference)
        if self.reference is not None:self.reference.close();self.reference=None
