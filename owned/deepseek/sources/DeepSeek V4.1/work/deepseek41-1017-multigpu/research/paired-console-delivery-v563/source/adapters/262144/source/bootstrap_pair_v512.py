from index_selection_v512 import uuids as _index_uuids
"""Surgical startup edge for literal selected old run()/C++/CPU interfaces.
No model/DSO/CUDA on import. Source-only until Root's packer/operator/model proof.
"""
from contextlib import ExitStack
import ctypes as C,hashlib,json,os,sys,time
from pathlib import Path
from private_source_loader_v512 import private_source
from one_bank_store import HERE,Q,D,OneBankStore,OUTPUT_BYTES,need,digest,source
from stage_devices_pair_v512 import activate_pair,validate_visible_pair,require_gpu1_no_foreign_compute
from pair_runtime_policy_v512 import environment_for,runtime_record

def register_cpu(store):
 from pair_format_v512 import binding,descriptor,Descriptor
 from cpu_projection_binding_v512 import resolve_cpu_projection
 bank=binding(store);store.validate_selected_owner(deep=True)
 need(len(store.cpu_blocked_cache)==46080 and store.cpu_blocked_report['bytes']==OUTPUT_BYTES,'all unchanged matrix descriptors')
 from candidate_cpu_local import verify_local
 verify_local(store)
 path=resolve_cpu_projection().library;lib=C.CDLL(str(path));fn=lib.ds143_pair32_cache_register;fn.argtypes=[C.c_char_p,C.POINTER(Descriptor)];fn.restype=C.c_int
 for name,r in store.cpu_blocked_cache.items():
  d=descriptor(store,name,r,bank=bank);need(fn(name.encode(),C.byref(d))==0,'typed CPU143 registration '+name)
 need(binding(store)==bank,'paired139 format changed during CPU143 registration')
 os.environ['DS41_CPU_BLOCK_CACHE_GIB']='320';os.environ['DS41_CPU_BLOCK_CACHE_BORROWED']='1';store._projection_lib=lib

def profile_manifest(store):
 # Display contract only, never change the authenticated source manifest.
 m=dict(store.manifest);recipe=dict(m['recipe']);recipe.update(profile='new1017-lossless-experts-engram-q4',policy=dict(experts='source-E2M1+E8 lossless; CPU paired32 lossless139 from oldOutput16',engram='same approved Q4g32 BF16 scales',CPU_arithmetic='old WXA8/CPU118; not source-native identity'));m['recipe']=recipe;return m

def initialize_pair(runtime,settings,borrowed_store,library_path,library_sha256,module):
 from pair_policy_snapshot_v512 import PolicySnapshot
 from owner_contract_v512 import claim_runtime
 claim_runtime(borrowed_store)
 from pair_format_v512 import binding
 binding(borrowed_store)
 activate_pair()
 need(os.environ.get('DS1017_OLD_COMPLETE_RUNTIME_V1')=='1'and os.environ.get('DS41_ATTENTION_WOA_1017_SOURCE_PORT')=='1','oldcomplete1017 defaultOFF')
 pair_env=environment_for(settings)
 for key,value in pair_env.items():os.environ[key]=str(value)
 need(validate_visible_pair()==(_index_uuids()[0],_index_uuids()[1]),'exact pair GPU identities/order')
 need(isinstance(borrowed_store,OneBankStore),'explicit NEW1017 owner, never old store/fallback')
 need(settings.experimental and settings.weight_profile is None and settings.mtp in('off','native4')and not settings.mtp_arena_upload and not getattr(settings,'mtp_short_owned_pool',False),'private new native4 only')
 module.validate_cpp_settings(settings);borrowed_store.validate_selected_owner(deep=True)
 from contiguous_rows_v39 import install
 install(borrowed_store)
 need(settings.packed_ram_gib*(1<<30)>=borrowed_store.ram_preload_report['resident_bytes'],'truthful new435GB resident budget')
 runtime.settings=settings;runtime.resources=ExitStack();runtime.reference=runtime.registry=runtime.expert_pool=None;runtime.index_mask_info_v172=None;runtime.mtp_owned_pool=None
 runtime.path=borrowed_store.source;runtime.manifest=profile_manifest(borrowed_store);runtime.bundle_sha256=borrowed_store.model_bundle_sha256;runtime.library=Path(library_path).absolute()
 need(runtime.library.name=='libds1017_pair_runtime_v1.so','K14 actual runtime DSO basename required')
 need(digest(runtime.library)==library_sha256,'explicit new1017 whole DSO SHA required')
 runtime.implementation={str(runtime.library):library_sha256,'source_packet':str(HERE),'frozen_dependencies':digest(HERE/'DEPENDENCIES.json')}
 runtime.pair_runtime=runtime_record(settings)
 runtime.implementation.update(pair_source_packet=str(Path(__file__).resolve().parents[1]),pair_dependency_pins_sha256=digest(Path(__file__).resolve().parents[1]/'DEPENDENCY_PINS.json'))
 # Preserve selected per-context environment verbatim; do not erase the fast
 # fullpass memory/candidate/suffix settings to manufacture a literal slow base.
 for key,value in module.cpp_configuration(settings)['native_environment'].items():os.environ[key]=str(value)
 os.environ.update(DS41_NATIVE_DENSE_FP8='1',DS41_NATIVE_DENSE_PREFILL='1',DS41_NATIVE_FP8_EXPERTS='1',DS41_NATIVE_COORDINATOR=str(settings.expert_coordinator_cpu))
 # Reapply pair policy after the frozen CPU/GPU environment so it cannot fall back to GPU0 or an old prefix.
 for key,value in pair_env.items():os.environ[key]=str(value)
 runtime._pair_policy=PolicySnapshot.capture(settings)
 need(validate_visible_pair()[0]==_index_uuids()[0],'logical GPU0 backbone')
 health=private_source(Path(__file__).resolve().parent/'gpu0_health_index_v512.py','_index_gpu0_health_v512')
 sys.modules['gpu0_health']=health # dedicated N1 child; same qualified UUID health, current PCI
 h=health.query_gpu0();h['compute_pids']=[p for p in h['compute_pids']if p!=os.getpid()];health.require_idle_healthy(h)
 require_gpu1_no_foreign_compute()
 import native_cuda_loader_pair_v512
 loaded=native_cuda_loader_pair_v512.load_isolated(runtime.library)
 if settings.context in (262144,524288):
  values=(C.c_uint64*5)();fn=loaded.ds41_index_info_v172;fn.argtypes=[C.c_void_p,C.c_int];fn.restype=C.c_int
  need(fn(values,5)==0 and list(values)==[172,1,64,128,0],'literal selected index-mask geometry')
  runtime.index_mask_info_v172=dict(zip(('version','enabled','tile_keys','CTA_threads','extra_explicit_buffer_bytes'),map(int,values)))
 from reference_pair_v512 import NativeReference
 from mtp_binding_1017 import ModelBinding,ATTRIBUTE
 from native_registry_pair_v512 import NativeBundleRegistry
 try:
  import torch
  from index_selection_v512 import verify_cuda_pair
  verify_cuda_pair(torch)
  torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');runtime.torch=torch
  runtime.reference=NativeReference(borrowed_store,packed_budget=settings.vision_cache_gib<<30)
  runtime.mtp_binding=ModelBinding(borrowed_store);setattr(borrowed_store,ATTRIBUTE,runtime.mtp_binding);runtime.mtp_binding.attach(runtime.reference)
  from mtp_driver_pair_v512 import prepare_mtp_context
  prepare_mtp_context(runtime.reference,settings.context)
  native_cuda_loader_pair_v512.load_isolated(D/'build/native-mtp-resident-moe-stream-v41.so');native_cuda_loader_pair_v512.load_isolated(D/'build/native-mtp-dense-stream-v41.so')
  torch.cuda.current_blas_handle();warm=torch.zeros((16,16),dtype=torch.bfloat16,device=runtime.reference.device);result=torch.mm(warm,warm,out_dtype=torch.float32);torch.cuda.synchronize(runtime.reference.device);del result,warm;torch.cuda.empty_cache()
  runtime.ram_proof=borrowed_store.ram_preload_report;register_cpu(borrowed_store)
  runtime.registry=runtime.resources.enter_context(NativeBundleRegistry(borrowed_store,runtime.library).finish_checkpoint())
  runtime.config={k:v for k,v in borrowed_store.config.items()if k not in('native_geometry_contract','runtime_asset_map','compact_mixed_source_provenance')};runtime.startup_warmup=dict(skipped=True,reason='same selected request-local warmup after PP')
  # Bind the bundled1017 DSML codec and processor; never old source metadata.
  features=source(Q/'research/compact-runtime-feature-closure-v7/features_v7.py','_old1017_bundle_codec')
  codec=features.api_codec(borrowed_store)
  for n in('MODEL_ID','ByteLevelDecoder','NativeDeltaParser','native_request','encode_messages'):setattr(module,n,getattr(codec,n))
  module.IMAGE_PROCESSOR=next(borrowed_store.source/a['output_name']for a in borrowed_store.config['runtime_asset_map']['items']if a['runtime_path']=='inference/image_processor.py')
 except BaseException:runtime.close();raise
 print(json.dumps(dict(phase='old_complete_new1017_runtime_ready_source_candidate',context=settings.context,owner=runtime.ram_proof,DSO_sha256=library_sha256,quality_admitted=False)),flush=True)
