"""Strict stdlib-only admission of actual Root94 build and both Root95 native results."""
from pathlib import Path
import hashlib,json
BASE_SHA='d31fcad41cff9aa75bfffab2824019d859c176d3ceffc3b32ed6eabf95c6e9d3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,message):
 if not v:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text())
def prove(build_path,proof_path):
 build_path=Path(build_path).absolute();proof_path=Path(proof_path).absolute();build=read(build_path);proof=read(proof_path)
 need(build['phase']=='COMPILED_NOT_EXECUTED','Root94 build receipt phase differs')
 dsos=build['DSOs'];need(len(dsos)==2 and {r['name'] for r in dsos}=={'control','queue'},'exact control/queue94 build rows required')
 variants={r['name']:r for r in dsos}
 for name,row in variants.items():
  p=Path(row['path']);need(p.is_absolute() and p.is_file() and sha(p)==row['sha256'],'actual '+name+'94 DSO SHA differs')
 need(proof['phase']=='NATIVE_CPU_COMPONENT_PASSED_NOT_MODEL_ADMITTED','Root95 proof is not exact native PASS phase')
 need(proof['candidate_path']==variants['queue']['path'] and proof['candidate_sha256']==variants['queue']['sha256'],'Root95 candidate is not actual queue94 build')
 baseline=Path(proof['baseline_path']);need(baseline.is_absolute() and sha(baseline)==BASE_SHA==proof['baseline_sha256'],'selected original CPU125 baseline differs')
 results=proof['results'];need(len(results)==2 and {r['name'] for r in results}=={'selected','fixed-control'},'both native95 result arms required')
 expected={'compared_cases':992,'compared_float_words_per_arm':1577184,'explicit_prequant_calls_per_arm':496,'alias_cases':12,'post_calls_all_arms':90,'guard_refusals':28,'fixture_count':47,'permanent_fixture_bytes':30549952,'working_reserve_bytes':8388608,'fixture_cap_bytes':67108864,'independent_DSO_files':2,'DSO_arms':3}
 actual_results=[]
 for row in sorted(results,key=lambda r:r['name']):
  p=Path(row['result_path']);need(row['returncode']==0 and type(row['returncode'])is int,'native95 result returncode differs')
  need(p.is_absolute() and sha(p)==row['result_sha256'],'native95 actual result SHA differs');result=read(p);need(result==row['result'],'Root95 saved result differs from actual JSON')
  need(result['phase']=='PASS' and result['runtime_proof']is True and result['synthetic_only']is True,'native95 result is not synthetic runtime PASS')
  need(result['model_files_opened']is False and result['CUDA_used']is False and result['timing_claim']is False,'native95 scope differs')
  need(result['post_lifetime94_candidate']is True and result['baseline_progress_pool_lifetime']is True,'native95 progress lifetime guards missing')
  need(result['ON_policy']==[80,1,16,2] and result['OFF_policy']==[80,0,16,2],'native95 latched queue policy differs')
  need(result['workers']==[0,1,128,129],'native95 admitted worker fixture differs')
  for key,value in expected.items():need(result[key]==value and type(result[key])is int,'native95 coverage field differs: '+key)
  actual_results.append(dict(name=row['name'],path=str(p),sha256=sha(p)))
 exe_sha=proof['executable_sha256'];need(isinstance(exe_sha,str) and len(exe_sha)==64 and all(c in '0123456789abcdef'for c in exe_sha),'Root95 actual executable SHA required')
 return dict(schema='ds1017-CPU94-native95-admission/104',selected_role='control',build=dict(path=str(build_path),sha256=sha(build_path)),proof=dict(path=str(proof_path),sha256=sha(proof_path)),library=dict(path=variants['control']['path'],sha256=variants['control']['sha256']),queue_reference=dict(path=variants['queue']['path'],sha256=variants['queue']['sha256']),control=dict(path=variants['control']['path'],sha256=variants['control']['sha256']),baseline=dict(path=str(baseline),sha256=BASE_SHA),results=actual_results,executable_sha256=exe_sha,model_pending=True,quality_admitted=False,model_speed_admitted=False)
