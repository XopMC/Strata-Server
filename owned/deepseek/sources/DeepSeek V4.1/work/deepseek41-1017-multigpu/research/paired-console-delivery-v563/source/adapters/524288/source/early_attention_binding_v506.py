"""Root337 actual build/CPU protocol binding; no native imports or execution."""
import hashlib,json
from pathlib import Path
PHASE='ROOT337_ALL5_BUILD_STEPS_PASS_CPU_AND_MODEL_PENDING'
def need(v,m):
 if not v:raise RuntimeError('342 early-attention binding: '+m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(v,*,require_ready=True):
 need(v.get('schema')==342 and(not require_ready or v.get('ready')is True),'Root binding pending')
 need(v.get('flag')=={'DS1017_PREFILL_EARLY_ATTENTION_V337':'1'},'explicit337 flag1')
 for k in('build','source_pins','recipe','CPU_proof','library'):
  r=v[k];p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and digest(p)==r['sha256'],'changed '+k)
 b=json.loads(Path(v['build']['path']).read_text());need(b.get('phase')==PHASE and b.get('native_or_GPU_run')is False and len(b.get('steps',[]))==5 and all(r.get('returncode')==0 for r in b['steps']),'actual five Root337 build steps')
 need(b.get('source_pins_sha256')==v['source_pins']['sha256'],'actual337 source manifest')
 a=b['artifacts'].get(v['library']['path']);need(a is not None and a['sha256']==v['library']['sha256'],'actual337 DSO')
 p=json.loads(Path(v['CPU_proof']['path']).read_text());need(p.get('phase')=='ROOT337_CPU_PHASE_PROTOCOL_PASS_NO_GPU'and p.get('exit_code')==0 and p.get('GPU_used')is False and p.get('model_used')is False,'actual CPU protocol scope')
 x=json.loads(p['stdout']);need(x=={'phase':'CPU337_ATTENTION_FFN_PHASE_ACK_PROTOCOL_PASS_NOT_GPU','full_layers':80,'guards':12,'borrow_abort_bookkeeping':True,'CUDA_used':False,'model_used':False},'actual CPU protocol output')
 need(p.get('argv')==[str(Path(v['library']['path']).parents[1]/'component_protocol_v337')],'actual protocol executable identity')
 return v
