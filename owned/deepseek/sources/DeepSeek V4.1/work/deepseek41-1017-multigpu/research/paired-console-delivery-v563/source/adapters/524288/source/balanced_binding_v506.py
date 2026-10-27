"""432 exact407link plusNativeExperts424; no runtime or native import."""
import json,hashlib
from pathlib import Path
PHASE='ROOT432_ONE_HOST_LINK_PASS_MODEL_PENDING'
def check_balanced(v):
 from bulk_binding_v506 import need,pin
 e=v['balanced432'];b=json.loads(pin(e['build']).read_text());r=json.loads(pin(e['recipe']).read_text());mf=pin(e['source_pins'])
 need(b['phase']==PHASE and b['source_pins_sha256']==e['source_pins']['sha256'],'actual432 phase/source')
 need(len(b['steps'])==len(r['commands'])==1 and b['steps'][0]['returncode']==0 and b['steps'][0]['argv']==r['commands'][0]['argv'],'exact one432 link')
 need(b['artifacts']['Engine']['path']==v['engine']['path']and b['artifacts']['Engine']['sha256']==v['engine']['sha256'],'actual432selectedEngine')
 for row in json.loads(mf.read_text())['files']:
  q=Path(row['path']);q=q if q.is_absolute()else mf.parent/q;need(hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'],'frozen432source '+str(q))
 from balanced_hash_cache_v506 import verify_once
 verify_once(json.loads((mf.parent/'INPUT_PINS.json').read_text())['files'],policy=(tuple(sorted(v['flag'].items())),v['engine']['path'],v['engine']['sha256'],e['source_pins']['sha256']),author_pins=Path(__file__).resolve().parents[1]/'AUTHOR_SOURCE_PINS.json',seal=Path(__file__).resolve().parents[1]/'SEALED.json')
 baseline=json.loads(pin(r['base407_recipe']).read_text())['commands'][-1];actual=json.loads(pin(r['base407_build']).read_text());need(actual['phase']=='ROOT407_ALL4_ENGINE_GUARD_BUILD_STEPS_PASS_MODEL_PENDING'and all(x['returncode']==0 for x in actual['steps'])and actual['steps'][-1]['argv']==baseline['argv'],'actual407baseline link')
 cmd=r['commands'][0];diff=[dict(index=i,old=x,new=y)for i,(x,y)in enumerate(zip(baseline['argv'],cmd['argv']))if x!=y]
 need(len(cmd['argv'])==len(baseline['argv'])and len(diff)==2 and diff==r['argv_replacements']and cmd['cwd']==baseline['cwd'],'ONLY output and NativeExperts archive replaced')
 need(r['CPU_unchanged']['sha256']==v['cpu']['sha256']and r['Task_unchanged']['sha256']==v['task']['sha256'],'sameCPU418Task427')
 return e
