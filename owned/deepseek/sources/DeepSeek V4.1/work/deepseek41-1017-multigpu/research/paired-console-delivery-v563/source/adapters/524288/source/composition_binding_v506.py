"""Root349 exact source composition admission; stdlib only, no DSO import."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
PHASE='ROOT349_ALL4_BUILD_STEPS_PASS_MODEL_PENDING'
def need(v,m):
 if not v:raise RuntimeError('350 composition binding: '+m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and digest(p)==r['sha256'],'changed actualpin '+str(p));return p
def check(require_ready=True):
 v=json.loads((P/'COMPOSED_BINDING.json').read_text());need(v.get('schema')==408 and(not require_ready or v.get('ready')is True),'Root binding pending')
 need(v.get('flags')=={'DS1017_K14_PACKED_REGISTER_V334':'1','DS1017_PREFILL_EARLY_ATTENTION_V337':'1','DS1017_PARALLEL_HEADLESS_STAGE_COPY_V331':'1'}and v.get('no329_or335')is True,'exact three flags/no329or335')
 pin(v['source_pins']);pin(v['recipe'])
 prior=json.loads((P/'PRIOR_COMPONENT_BINDINGS.json').read_text())
 from early_attention_binding_v506 import check as early_check
 from rank_copy_binding_v506 import check as copy_check
 early_check(prior['early337']);copy_check(prior['copy331'])
 if require_ready or v['build'].get('sha256'):
  b=json.loads(pin(v['build']).read_text());r=json.loads(pin(v['recipe']).read_text())
  need(b.get('phase')==PHASE and len(b.get('steps',[]))==4 and all(x.get('returncode')==0 for x in b['steps']),'actual four Root349 steps')
  need(b.get('source_pins_sha256')==v['source_pins']['sha256']and b.get('native_or_GPU_run')is False,'actual349source scope')
  need(all(x['argv']==y['argv']for x,y in zip(b['steps'],r['commands'])),'actual349argv; cwd declared in pinned recipe, not separately recorded in actual receipt')
  pin(v['library']);need(b['artifacts'][v['library']['path']]['sha256']==v['library']['sha256'],'actual349DSO')
 if require_ready:
  from candidate_cutoff_binding_v506 import check as candidate_check
  selected=candidate_check();v=dict(v,library=selected['library'],build=selected['build'],selected_Engine=407)
 return v
