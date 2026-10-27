"""Actual381 fixed256 build + qualified369/375 provenance; pure metadata only."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
PHASE='ROOT381_ALL5_MATCHED_BUILD_STEPS_PASS_MODEL_PENDING'
def need(v,m):
 if not v:raise RuntimeError('382 candidate binding: '+m)
def pin(r):
 f=Path(r['path']);need(f.is_absolute()and f.is_file()and not f.is_symlink()and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],'actual pin '+str(f));return f
def prior381_check(require_ready=True):
 v=json.loads((P/'CANDIDATE_BINDING.json').read_text());need(v['schema']==442 and(not require_ready or v['ready']is True),'Root381 actual binding pending')
 need(v['flag']=={'DS1017_CANDIDATE_CUTOFF_V375':'1'},'literal375 flag')
 for k in('source_pins','recipe','counter_contract','feature_source_pins','feature_source_check','component_proof'):pin(v[k])
 sc=json.loads(pin(v['feature_source_check']).read_text());need(sc['returncode']==0 and sc['source_unchanged']is True and sc['original375_pins_sha256']==v['feature_source_pins']['sha256'],'actual375 R2 feature source check')
 cp=json.loads(pin(v['component_proof']).read_text());need(cp['phase']=='BOTH_GPUS_369_CANDIDATE_CUTOFF_COMPONENT_PASS_NO_MODEL'and cp['model_TPS_proof']is False and len(cp['devices'])==2,'actual373 component-only scope')
 for d,row in enumerate(cp['devices']):need((row['device'],row['cases'],row['mask_bytes'],row['graph_replays'],row['guards'])==(d,26,13631488,6,10),'bothGPU parity scope')
 if require_ready:
  b=json.loads(pin(v['build']).read_text());r=json.loads(pin(v['recipe']).read_text())
  need(b['phase']==PHASE and len(b['steps'])==5 and all(x['returncode']==0 for x in b['steps']),'actual381 five steps')
  need(b['source_pins_sha256']==v['source_pins']['sha256'] and b['native_or_GPU_run']is False,'actual381 source/build scope')
  need(all(x['argv']==y['argv']for x,y in zip(b['steps'],r['commands'])),'actual381 argv equals recipe')
  f=pin(v['library']);need(f==M/'build/candidate-cutoff-fixed256-v381/262144/libds1017_pair_runtime_v1.so'and b['artifacts'][str(f)]['sha256']==v['library']['sha256'],'actual381 selectedDSO')
 return v

def check(require_ready=True):
 v=prior381_check(require_ready)
 from bulk_binding_v512 import check as bulk_check
 b=bulk_check(require_ready)
 if require_ready:
  from verify_woa_binding_v512 import check as woa_check
  w=woa_check();v=dict(v,library=w['contexts']['262144'],build=w['build'],selected_Engine=449)
 return v
