"""Typed511 host composition; no native load or device query."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import hashlib,importlib,json,sys
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
PHASE='ROOT511_AUTHENTIC256_503_469_LINK_PASS_NATIVE_PENDING'
def need(v,m):
 if not v:raise RuntimeError('512 composition: '+m)
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink(),'canonical direct pin '+str(p));need(hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']and('bytes'not in r or p.stat().st_size==r['bytes']),'changed '+str(p));return p
def parent(name,version):
 need((name,version)in {('fixed_binding',506),('k5_composite_binding',506),('index_native_binding',506),('bulk_binding',499),('adaptive_binding',499),('verify_woa_binding',499)},'finite historical metadata source')
 base=str(M/f'research/pair-runtime-adapter-v{version}/source');old=list(sys.path)
 try:
  sys.path.insert(0,base);return importlib.import_module(f'{name}_v{version}')
 finally:sys.path[:]=old

def call_parent(name,version,method='check',*args,**kwargs):
 base=str(M/f'research/pair-runtime-adapter-v{version}/source');old=list(sys.path)
 try:
  sys.path.insert(0,base);return getattr(parent(name,version),method)(*args,**kwargs)
 finally:sys.path[:]=old

def expected_argv(base,output):
 n=M/'build';oldnative=str(n/'cpu-numa-bulk-engine-v424/libds41_native_experts.a');oldowner=str(n/'index-native-admission-v501/whole_owner_capacity_v403.o');oldbank=str(n/'index-native-admission-v501/adaptive_bank_capacity_v403.o');oldstage=str(n/'checked-stage-copy-v378/stage_copy_v378.o')
 need(all(base.count(x)==1 for x in(oldnative,oldowner,oldbank,oldstage)),'one of each original403 host input')
 out=[];oldout=base[base.index('-o')+1]
 for a in base:
  if a==oldout:out.append(output)
  elif a==oldnative:out.append(str(n/'whole-fixed-plan-owner-v469/libds41_native_experts.a'))
  elif a==oldowner:out.extend([str(n/'index-native-admission-v501/whole_owner_fixed_v469.o'),str(n/'index-native-admission-v501/resources_v469.o')])
  elif a in(oldbank,oldstage):continue
  else:out.append(a)
 return out

def check():
 v=json.loads((P/'PORT_BINDING.json').read_text());need(v.get('schema')==512 and v.get('ready')is True and v.get('context')==262144,'Root511 composition/current owner binding pending')
 for k in('parent499_seal','parent506_seal','recipe511','source_ready511','build511','link504','baseline499_raw','baseline499_reap','baseline499_close'):pin(v[k])
 b=json.loads(Path(v['build511']['path']).read_text());r=json.loads(Path(v['recipe511']['path']).read_text());old=json.loads(Path(v['link504']['path']).read_text())
 need(b['phase']==PHASE and b['source_recipe_sha256']==v['recipe511']['sha256']and b['GPU_execution']is False and b['native_model_run']is False and b['numerical_weights_changed']is False,'actual511 host link scope')
 need(len(b['steps'])==len(r['commands'])==1 and b['steps'][0]['exit_code']==0 and b['steps'][0]['argv']==r['commands'][0]['argv'],'one actual exact511 command')
 base=next(x['argv']for x in old['steps']if x['context']=='262144');need(r['base_argv']==base,'actual504256 base')
 z=v['contexts'];need(set(z)=={'262144'},'one reachable selected256 context');row=z['262144'];pin(row)
 need(set(v['controller_carriers'])=={'524288'},'frozen controller requires separately labelled512 carrier');pin(v['controller_carriers']['524288'])
 need(v['controller_carriers']['524288']==json.loads((M/'research/pair-runtime-adapter-v506/K5_COMPOSITE_BINDING.json').read_text())['contexts']['524288'],'carrier is actual accepted506512, never256 proof')
 need(b['artifact']==b['artifacts']['DSO']==row and b['steps'][0]['argv']==expected_argv(base,row['path']),'actual artifact and only declared host substitutions')
 need(r['commands'][0]['cwd']==str(M.parent/'deepseek41/build/native-adapter-sm89')and r['Engine_CUDA_compile']is False,'original link cwd/no Engine compile')
 bank=json.loads((P/'BANK_BINDING.json').read_text());need(bank.get('ready')is True and bank['bank_format_binding']==v['bank_format_binding'],'same actual live139 bank')
 raw=json.loads(Path(v['baseline499_raw']['path']).read_text());reap=json.loads(Path(v['baseline499_reap']['path']).read_text())
 need(raw['usage']==dict(prompt_tokens=261568,completion_tokens=512,total_tokens=262080)and raw['cancelled']is False and raw['mtp']['mode']=='native4','completed499 full native4 fixture')
 need(raw['actual_prefill_shape']==dict(passes=2,chunk_tokens=131072,attention_batch=2048),'actual499 fixed256 geometry')
 need(len(raw['generated_token_ids'])==512 and len(set(raw['generated_token_ids']))>=16,'actual varied499 outputs')
 need(reap.get('phase')=='CHILD_REAPED'and reap.get('normal_exit')is True and reap.get('exit_code')==0,'actual499 child reaped normally')
 close=json.loads(Path(v['baseline499_close']['path']).read_text());need(close.get('phase')=='CHILD_RUNTIME_CLOSED_ZERO_LEASES'and close.get('runtime_close_completed')is True and close.get('native_leases')==0,'actual499 closed/drained')
 dso=v['baseline499_dso'];need(raw['runtime_configuration']['implementation_sha256'].get(dso['path'])==dso['sha256']and reap['runtime_dso_sha256']==close['runtime_dso_sha256']==dso['sha256'],'actual499 artifact, not a carrier')
 return v
