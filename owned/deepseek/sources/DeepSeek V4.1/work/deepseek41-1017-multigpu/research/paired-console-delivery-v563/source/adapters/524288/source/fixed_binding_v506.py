
def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from cohort_binding_v539 import WORKERS
"""Explicit469 host overlay on immutable457 numerical provenance. Stdlib only.

Not admissible until Root supplies actual469 recipe/build/DSO identities and
actual468 execution. Historical artifacts remain historical, never relabelled.
"""
from pathlib import Path
import hashlib,importlib,json,sys
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
FOUNDATION_WORKERS=tuple(range(127))+tuple(range(128,253))
def need(v,m):
 if not v:raise RuntimeError('470 fixed-plan: '+m)
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'changed canonical pin '+str(p));return p
def prior(name):
 """Load only immutable stdlib metadata checks; no native/runtime import."""
 need(name in ('bulk_binding','adaptive_binding','verify_woa_binding'),'finite prior metadata module')
 base=str(M/'research/pair-runtime-adapter-v457/source');old=list(sys.path)
 try:
  sys.path.insert(0,base)
  return importlib.import_module(name+'_v457')
 finally:sys.path[:]=old

def check():
 v=json.loads((P/'FIXED_PLAN_BINDING.json').read_text());need(v.get('schema')==470 and v.get('ready')is True,'unbound source draft; no469 admission')
 need(type(v.get('mode'))is int and v['mode']in(0,1),'immutable literal OFF/ON')
 need(v['workers']==list(FOUNDATION_WORKERS)and v['coordinators']==[127,254]and v['main_cpu']==255 and v['uploader_cpu']==253,'actual252 partition')
 need(v['capacity']==dict(catalog=15360,physical=1078,active=1070,spares=8,permanent_statics=0,budget=20<<30),'fixed full-movable capacity')
 r=json.loads(pin(v['recipe']).read_text());b=json.loads(pin(v['build']).read_text());pin(v['source_pins'])
 need(b['phase']=='ROOT469_ALL_MATCHED_HOST_STEPS_PASS_MODEL_PENDING'and b['source_pins_sha256']==v['source_pins']['sha256'],'actual469 host overlay build')
 need(len(b['steps'])==len(r['commands'])and len(b['steps'])>0 and all(x['returncode']==0 and x['argv']==y['argv']for x,y in zip(b['steps'],r['commands'])),'exact executed recipe')
 need(set(v['contexts'])=={'262144','524288'},'two real matched DSO witnesses')
 for row in v['contexts'].values():
  pin(row);need(b['artifacts'][row['path']]['sha256']==row['sha256'],'actual built DSO')
 for key in ('cpu','task'):pin(v[key])
 proof=json.loads(pin(v['proof468']).read_text());run=json.loads(pin(v['proof468_run']).read_text())
 need(run['phase']=='COMPLETE'and run['rc']==0 and run['reaped']is True,'actual468 finite successful process')
 need(run['proof_sha256']==v['proof468']['sha256'],'actual468 executed proof identity')
 for key in ('cpu','task'):
  actual=v['task427_reference']if key=='task'else v[key]
  need(run['actual_loaded_artifacts'][actual['path']]['sha256']==actual['sha256'],'same actual468 reference '+key)
 from generation_off_binding_v506 import check as task489_check
 task489=task489_check();need(task489['task']==v['task']and task489['reference427']==v['task427_reference'],'actual489 Task overlay on qualified427')
 need(proof['phase']=='CPU468_ACTUAL252_TASK427_ORACLE_EXACT_COMPONENT_PASS_NOT_MODEL'and proof['candidate_workers']==252 and proof['case_count']==36 and proof['CUDA_used']is False and proof['model_files_opened']is False,'actual468 numerical scope')
 need(proof['candidate_home_CPU_ids']==[list(range(127)),list(range(128,253))]and proof['task418_info'][:6]==[418,1,127,125,127,254]and proof['task427_epoch_info'][3:6]==[36,36,36],'actual252 IDs/counts/epoch work')
 need(proof['BF16_words']==4792320 and proof['ordered_F32_words']==798720,'complete actual468 comparison')
 return v

def flags():
 v=check();return {'DS1017_FIXED_PLAN_UPLOAD_V465':'1','DS1017_FIXED_PLAN_REFRESH_V469':str(v['mode']),'DS1017_GPU1_GENERATION_OFF_V489':'0'}

def call_prior(name,method='check',*args,**kwargs):
 base=str(M/'research/pair-runtime-adapter-v457/source');old=list(sys.path)
 try:
  sys.path.insert(0,base)
  return getattr(prior(name),method)(*args,**kwargs)
 finally:sys.path[:]=old
