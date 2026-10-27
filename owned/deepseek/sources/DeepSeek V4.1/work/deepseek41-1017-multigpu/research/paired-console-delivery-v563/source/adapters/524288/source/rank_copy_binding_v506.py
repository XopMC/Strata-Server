"""Root331 source/build/CPU-only fixture binding. No native import or execution."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PHASE='ROOT331_ALL5_BUILD_STEPS_PASS_CPU_AND_MODEL_PENDING'
def need(v,m):
 if not v:raise RuntimeError('333 rank-copy binding: '+m)
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def check(v,*,require_ready=True):
 need(v.get('schema')==333 and(not require_ready or v.get('ready')is True),'Root binding pending')
 need(v.get('flag')=={'DS1017_PARALLEL_HEADLESS_STAGE_COPY_V331':'1'},'explicit331 flag1')
 for key in('build','source_pins','recipe','CPU_execution','CPU_proof','library'):
  row=v[key];p=Path(row['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and digest(p)==row['sha256'],'changed '+key)
 b=json.loads(Path(v['build']['path']).read_text());need(b.get('phase')==PHASE and b.get('native_or_GPU_run')is False and len(b.get('steps',[]))==5 and all(r.get('returncode')==0 for r in b['steps']),'actual331five Root steps')
 need(b.get('source_pins_sha256')==v['source_pins']['sha256'],'actual331sourcepins')
 a=b['artifacts'].get(v['library']['path']);need(a is not None and a['sha256']==v['library']['sha256'],'actual331DSO artifact')
 execution=json.loads(Path(v['CPU_execution']['path']).read_text());proof=json.loads(Path(v['CPU_proof']['path']).read_text())
 need(execution.get('exit_code')==0 and execution.get('proof_path')==v['CPU_proof']['path'],'actual CPUfixture process')
 need(json.loads(execution['stdout'])==proof,'actual CPUfixture result/receipt match')
 need(proof.get('phase')=='CPU331_ACTUAL325_POOL_BYTE_AND_LIFETIME_PASS_NO_GPU' and proof.get('cases')==18 and proof.get('guards')==6 and proof.get('real_threads_joined')is True and proof.get('CUDA_used')is False and proof.get('model_used')is False and proof.get('performance_claim')is False,'actual CPU class proof scope')
 for r in(0,1):need((proof.get('rank%d_jobs'%r),proof.get('rank%d_parallel_jobs'%r),proof.get('rank%d_bytes'%r))==(9,3,21660752),'actual CPUfixture rank counters')
 return v
