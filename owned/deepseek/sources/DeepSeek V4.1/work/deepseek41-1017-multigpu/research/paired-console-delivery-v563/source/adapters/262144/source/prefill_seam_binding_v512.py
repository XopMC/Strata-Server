"""Pure352 serial351 admission; no native/Torch/model import."""
import json,hashlib,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
FLAGS={'DS1017_CANDIDATE_CUTOFF_V375':'1','DS1017_PREFILL_GEOMETRY_V335': '256', 'DS1017_PREFILL_SERIAL_TWO_PASS_V335': '0', 'DS1017_PREFILL_PASS_SEAM_V317': '1', 'DS1017_PREFILL_WAVEFRONT_V325': '1', 'DS1017_HEADLESS_PREFILL_V320': '1', 'DS1017_PAIR_INDEXER_PREFILL_V1': '0', 'DS1017_WAVEFRONT_GPU0_CPU_V325': '126', 'DS1017_WAVEFRONT_GPU1_CPU_V325': '252', 'DS1017_WAVEFRONT_COPY0_CPUS_V325': '0,1,2,3', 'DS1017_WAVEFRONT_COPY1_CPUS_V325': '128,129,130,131', 'DS1017_SERIAL_RANK_STAGE_COPY_V351': '0', 'DS41_NATIVE_PREFILL_PIPELINE': '1', 'DS41_NATIVE_PREFILL_PARALLEL_STAGE_COPY': '1'}
from adaptive_binding_v512 import FLAGS as ADAPTIVE_FLAGS
FLAGS.update(ADAPTIVE_FLAGS)
PHASE='ROOT351_ALL9_BUILD_STEPS_PASS_CPU_AND_MODEL_PENDING'
CPU_WRAPPER_PHASE='ROOT351_CPU_RANK_POOL_COPY_FIXTURE_PASS_NO_GPU'
CPU_PHASE='ROOT351_CPU_ACTUAL_RANK_POOL_COPY_PASS_NOT_CUDA_MODEL'
def need(v,m):
 if not v:raise RuntimeError('353 binding: '+m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pinned(row):
 p=Path(row['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and digest(p)==row['sha256'],'changed/missing '+str(p));return p

def check(value,require_ready=True):
 need(value.get('schema')==442 and(not require_ready or value.get('ready')is True),'actual351Root binding pending')
 need(value.get('flags')==FLAGS,'literal352 serial351 immutableflags')
 g=value['geometry335']
 for k in('build','source_pins','recipe','CPU_proof'):pinned(g[k])
 gb=json.loads(Path(g['build']['path']).read_text());need(gb.get('phase')=='ROOT335_ALL9_BUILD_STEPS_PASS_CPU_AND_MODEL_PENDING'and gb.get('source_pins_sha256')==g['source_pins']['sha256'],'original335geometry build provenance')
 gp=json.loads(Path(g['CPU_proof']['path']).read_text());need(gp.get('phase')=='ROOT335_CPU_GEOMETRY_FIXTURE_PASS_NO_GPU'and gp.get('exit_code')==0 and json.loads(gp['stdout'])==dict(phase='ROOT335_FIXED_GEOMETRY_CPU_CONTRACT_PASS_NOT_GPU_MODEL',checks=189,guards=8,GPU_used=False,model_used=False),'actual335CPUgeometry proof')
 for k in('build','source_pins','recipe','CPU_proof'):pinned(value[k])
 b=json.loads(Path(value['build']['path']).read_text());need(b.get('phase')==PHASE and len(b.get('steps',[]))==9 and all(x.get('returncode')==0 for x in b['steps']),'actual nine351Rootbuild steps')
 need(b.get('source_pins_sha256')==value['source_pins']['sha256'],'actual351source closure')
 need(set(value['contexts'])=={'262144'},'ONLYwavefront256 admitted')
 z=value['contexts']['262144'];pinned(z);need(z['sha256']==b['artifacts'][z['path']]['sha256'],'actual351DSO262')
 proof=json.loads(Path(value['CPU_proof']['path']).read_text());need(proof.get('phase')==CPU_WRAPPER_PHASE and proof.get('exit_code')==0,'actual351CPUwrapper proof')
 native=json.loads(proof['stdout']);need(native==dict(phase=CPU_PHASE,copies=2,parallel_copies=1,bytes=8388656,guards=4,workers=4,CPUs=[0,1,2,3],mocked_CUDA_device=0,GPU_used=False,model_used=False),'actual351CPU rankpool bytes/counter/guard proof, notCUDAmodel')
 from serial_reference_v512 import get_reference
 get_reference() # Explicit deny until actualcompleted352serial result is bound.
 return value

def admission():
 value=check(json.loads((P/'SEAM_BINDING.json').read_text()))
 from candidate_cutoff_binding_v512 import check as candidate_check
 c=candidate_check()
 from index_native_binding_v512 import overlay
 return overlay(dict(value,original_351_contexts=value['contexts'],contexts={'262144':c['library']},candidate381=c))
def latch():
 for k,v in FLAGS.items():
  if k!='DS1017_PAIR_INDEXER_PREFILL_V1':need(os.environ.get(k)in(None,v),'conflicting '+k)
 os.environ.update(FLAGS)
def record():
 v=admission();return dict(candidate381_actual_build=v['candidate381']['build'],candidate381_library=v['candidate381']['library'],source_variant=351,geometry_source335=v['geometry335'],mode='MATCHED353_PREFILL_PLUS381_DECODE_CUTOFF375',flags=dict(FLAGS),actual_build=v['build'],CPU_copy_only_proof=v['CPU_proof'],actual_context=262144,split=131072,cap=131072,B=2048,E=4096,actual_GPU1_headless_live_budget_required=True,failed347_and_draft348_are_not_references=True,index_native_admission501=v['index_native_admission501'],historical_proofs_not501_qualification=True,native_model_proof_pending=True,quality_admitted=False,performance_eligible=False)
