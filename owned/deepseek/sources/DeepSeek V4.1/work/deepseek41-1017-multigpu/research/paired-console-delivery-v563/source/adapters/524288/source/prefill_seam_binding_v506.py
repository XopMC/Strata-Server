"""Import-safe actual325/326 binding; no native/Torch/model imports."""
import hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PHASE='ROOT325_ALL7_MATCHED_BUILD_STEPS_PASS_CPU_CREDIT_AND_NATIVE_MODEL_PENDING'
RESOURCE_PHASE='HEADFUL_GPU0_HEADLESS_GPU1_RESOURCE_CONSTRUCTOR_CLOSE_PASS_NO_PREFILL'
FLAGS={'DS1017_CANDIDATE_CUTOFF_V375':'1','DS1017_PREFILL_PASS_SEAM_V317':'1','DS1017_PREFILL_WAVEFRONT_V325':'1',
 'DS1017_HEADLESS_PREFILL_V320':'1','DS1017_PAIR_INDEXER_PREFILL_V1':'0',
 'DS1017_WAVEFRONT_GPU0_CPU_V325':'126','DS1017_WAVEFRONT_GPU1_CPU_V325':'252',
 'DS1017_WAVEFRONT_COPY0_CPUS_V325':'0,1,2,3','DS1017_WAVEFRONT_COPY1_CPUS_V325':'128,129,130,131','DS1017_K14_PACKED_REGISTER_V334':'1','DS1017_PREFILL_EARLY_ATTENTION_V337':'1','DS1017_PARALLEL_HEADLESS_STAGE_COPY_V331':'1'}
from adaptive_binding_v506 import FLAGS as ADAPTIVE_FLAGS
FLAGS.update(ADAPTIVE_FLAGS)
def need(value,message):
 if not value:raise RuntimeError('328 binding: '+message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_pin(row,label):
 need(isinstance(row,dict) and set(row)>={'path','sha256'},'missing '+label)
 path=Path(row['path']);need(path.is_absolute() and path.is_file() and not path.is_symlink() and digest(path)==row['sha256'],'changed '+label)
 return path
def check(value,*,require_ready=True):
 need(value.get('schema')==408 and (not require_ready or value.get('ready')is True),'Root review/binding pending')
 need(value.get('flags')==FLAGS,'explicit immutable325 affinity/local-K14 flags')
 for key in('build','source_pins','recipe','counter_contract','credit_proof','resource_build','affinity'):
  check_pin(value[key],key)
 build=json.loads(Path(value['build']['path']).read_text())
 need(build.get('phase')==PHASE and len(build.get('steps',[]))==7 and all(r.get('returncode')==0 for r in build['steps']),'actual seven Root325 build steps')
 need(build.get('native_or_GPU_run')is False and build.get('source_pins_sha256')==value['source_pins']['sha256'],'actual325 source manifest/build scope')
 need(set(value['contexts'])=={'524288'},'328 only filled512 qualification')
 row=value['original325_DSO'];artifact=build['artifacts'].get(row['path'])
 need(artifact is not None and artifact['sha256']==row['sha256'] and digest(row['path'])==row['sha256'],'unchanged matched-header325 parent DSO')
 from k14_register_binding_v506 import check as check_k14
 k14=check_k14(require_ready=require_ready)
 need(value['contexts']['524288']==k14['DSO'],'actual349 composed DSO uses qualified334R4 archive')
 credit=json.loads(Path(value['credit_proof']['path']).read_text())
 need(credit.get('phase')=='ROOT325_CPU_CREDIT_COMPONENT_PASS_NOT_GPU_MODEL' and credit.get('actual_exit_code')==0 and credit.get('actual_GPU_used')is False and (credit.get('layers_each'),credit.get('packet_checks'),credit.get('guards'))==(40,80,7),'actual325 CPU credit fixture only')
 need(credit.get('actual_build')=={'path':value['build']['path'],'sha256':value['build']['sha256']},'credit proof actual build identity')
 if not require_ready:return value
 check_pin(value['resource_proof'],'fresh actual326 resource')
 resource=json.loads(Path(value['resource_proof']['path']).read_text());a=resource.get('native',{})
 need(resource.get('schema')==326 and resource.get('phase')==RESOURCE_PHASE and resource.get('native_rc')==0 and resource.get('prefill_called')is False and resource.get('generation_called')is False,'actual326 constructor/close proof; not prefill proof')
 need(resource.get('headful_closed')is True and resource.get('headful_close',{}).get('destroyed')is True and resource.get('headful_close',{}).get('current_device')==0,'actual326 headful cleanup')
 need((a.get('version'),a.get('stage'),a.get('context'),a.get('headless_info_version'),a.get('role'),a.get('logical_device'),a.get('native_math'))==(320,6,524288,320,320,1,1),'actual320 role/native math')
 need(all(a.get(k)==0 for k in('CPU_owner','worker_joinable','doorbell','head_pointer','active_pass','pending_layer_ACK')),'no second CPU owner/worker/decoder/pass')
 need(a.get('head_omitted_bytes')==a.get('pinned_embedding_bytes')==1323827200 and a.get('headless_destroyed')==a.get('restored_GPU0')==1,'actual omitted head/pinned embedding/close')
 need((a.get('budget_bytes'),a.get('latched_budget_bytes'),a.get('reserve_bytes'),a.get('latched_reserve_bytes'))==(20<<30,20<<30,512<<20,512<<20),'actual326 budget')
 need(a.get('planned_startup_device_bytes')==a.get('own_requested_device_bytes',-1)+a.get('current_Pack_device_bytes',-1)<=20<<30 and min(a.get('GPU1_free_constructed',0),a.get('GPU1_free_destroyed',0))>=512<<20,'actual326 requested ledger/reserve samples')
 need(resource.get('actual_build',{}).get('path')==value['resource_build']['path'] and resource['actual_build']['sha256']==value['resource_build']['sha256'],'actual326 matched320R2 build identity')
 affinity=json.loads(Path(value['affinity']['path']).read_text())
 need(affinity.get('phase')=='ROOT_READ_ONLY_CPU_PCI_TOPOLOGY_NOT_RUNTIME_AFFINITY' and affinity.get('settings_changed')is False,'Root prospective topology only')
 # This pinned affinity receipt is historical topology evidence only. Current
 # device admission is exclusively the fresh index0/1 snapshot and native501.
 from composition_binding_v506 import check as check_composed
 c=check_composed(require_ready=require_ready)
 need(value['contexts']['524288']==c['library'],'actual349 composedDSO required')
 return value
def admission():
 from index_native_binding_v506 import overlay
 return overlay(check(json.loads((P/'WAVEFRONT_BINDING.json').read_text())))
def latch():
 # BASE_PROFILES can carry old helper=1. This explicit profile replaces it BEFORE
 # any Engine constructor; all other new settings refuse conflicting prevalues.
 for key,wanted in FLAGS.items():
  if key!='DS1017_PAIR_INDEXER_PREFILL_V1':need(os.environ.get(key)in(None,wanted),'conflicting '+key)
 os.environ.update(FLAGS)
def record():
 v=admission()
 from composition_binding_v506 import check as composition_check
 c=composition_check()
 return dict(composed_actual_build=c['build'],source_variant=407,flags=dict(v['flags']),actual_build=v['build'],CPU_credit_proof=v['credit_proof'],headless_resource_only_proof=v['resource_proof'],prospective_CPU_affinity=v['affinity'],
  prefill_kind='original two-pass layer wavefront; qualified334R4 K14 + literal337earlyattention + literal331privatecopy; no329/335; S8/compact literal',
  index_native_admission501=v['index_native_admission501'],historical_proofs_not501_qualification=True,native_model_proof_pending=True,quality_admitted=False,performance_eligible=False)
