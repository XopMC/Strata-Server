"""456 actual-build selection; stdlib only, no native/model imports."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
PHASE='ROOT456_ALL8_MATCHED_ENGINE_BUILD_STEPS_PASS_MODEL_PENDING'
FLAGS={'DS1017_VERIFY_WOA_QUANT_MANY_V456':'1','DS1017_VERIFY_WOA_MANY_V449':'1','DS41_NATIVE_WOA_MANY':'0'}
def need(v,m):
 if not v:raise RuntimeError('458 verifier WOA: '+m)
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'actual pin '+str(p));return p
def artifact(b,row):
 a=b.get('artifacts',{});x=a.get(row['path'])if isinstance(a,dict)else None
 if x is None:x=next((v for v in(a.values()if isinstance(a,dict)else a)if v.get('path')==row['path']),None)
 need(x is not None and x['sha256']==row['sha256'],'actual456 artifact '+row['path'])
def check(require_ready=True):
 prior=json.loads(pin(dict(path=str(P/'PRIOR_449_BINDING.json'),sha256=json.loads((P/'VERIFY_WOA_BINDING.json').read_text())['prior449_sha256'])).read_text());need(prior['ready']is True and prior['source_pins']['sha256']=='247fca56d397a7c6b3185e6ce457dc7c2082b859df9a31fa3e740d8aa4edd50c','frozen452 actual449 prerequisite');
 v=json.loads((P/'VERIFY_WOA_BINDING.json').read_text());need(v['schema']==458 and v['flag']==FLAGS and v['runtime_contexts']==[262144]and v['metadata_only_contexts']==[524288],'strict256 runtime and genuine524 witness')
 need(not require_ready or v['ready']is True,'Root456 actual binding pending')
 for k in('source_pins','recipe','counter_contract'):pin(v[k])
 r=json.loads(Path(v['recipe']['path']).read_text());need(len(r['commands'])==8 and r['public_Engine_and_Pack_headers_changed']is False,'actual456 matched closure')
 for ctx,row in v['contexts'].items():need(row['path']==str(M/f'build/verifier-woa-quant-many-engine-v456/{ctx}/libds1017_pair_runtime_v1.so'),'finite actual456 DSO path')
 if require_ready:
  b=json.loads(pin(v['build']).read_text());need(b['phase']==PHASE and b['source_pins_sha256']==v['source_pins']['sha256'],'actual456 phase/source')
  need(len(b['steps'])==8 and all(x.get('returncode',x.get('exit_code'))==0 for x in b['steps']),'actual456 all8steps')
  need(all(x['argv']==y['argv']for x,y in zip(b['steps'],r['commands'])),'actual456 argv equals pinned recipe')
  from verify_woa_hash_cache_v512 import verify_once
  verify_once(list(v['contexts'].values()),policy=('456',v['source_pins']['sha256'],tuple(sorted(FLAGS.items()))),author_pins=P/'AUTHOR_SOURCE_PINS.json',seal=P/'SEALED.json')
  for row in v['contexts'].values():artifact(b,row)
  need(r['CPU_unchanged']['sha256']==json.loads((P/'CPU_BINDING.json').read_text())['library']['sha256'],'CPU418 unchanged')
 return v
