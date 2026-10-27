"""Finite Task427-only overlay; CPU418 and Engine424 remain selected."""
import json,hashlib
from pathlib import Path
PHASE='ROOT427_TASK_ONLY_HOST_BUILD_PASS_TASK419_AND_MODEL_PENDING'
def check_epoch_build(v):
 from bulk_binding_v512 import need,pin
 e=v['epoch427'];b=json.loads(pin(e['build']).read_text());r=json.loads(pin(e['recipe']).read_text());s=pin(e['source_pins'])
 need(b['phase']==PHASE and b['source_pins_sha256']==e['source_pins']['sha256'],'actual Task427 build/source')
 need(len(b['steps'])==len(r['commands'])==1 and all(x['returncode']==0 and x['argv']==y['argv']for x,y in zip(b['steps'],r['commands'])),'one exact Task-only build command')
 a=b['artifacts']['Task'];need(a['path']==v['task']['path']and a['sha256']==v['task']['sha256'],'selected Task427 artifact')
 for row in json.loads(s.read_text())['files']:
  q=Path(row['path']);q=q if q.is_absolute()else s.parent/q;need(hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'],'Task427 source identity '+str(q))
 return e
