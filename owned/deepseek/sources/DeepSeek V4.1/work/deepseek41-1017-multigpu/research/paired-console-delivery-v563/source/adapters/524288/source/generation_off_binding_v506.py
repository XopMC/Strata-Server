"""Actual Task489 host-only overlay; no new native/model qualification claimed."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parents[1]
def need(v,m):
 if not v:raise RuntimeError('490 Task489: '+m)
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'changed actualpin '+str(p));return p
def check():
 v=json.loads((P/'GENERATION_OFF_BINDING.json').read_text());need(v.get('schema')==490 and v.get('ready')is True and v.get('mode')==0,'Task489 binding pending/mode')
 b=json.loads(pin(v['build']).read_text());r=json.loads(pin(v['recipe']).read_text());pin(v['source_pins']);pin(v['header']);pin(v['reference427']);pin(v['task'])
 need(b['phase']=='ROOT489_TASK_ONLY_BUILD_PASS_ABLATION_MODEL_PENDING'and b['source_pins_sha256']==v['source_pins']['sha256'],'actual489 build/source')
 need(len(b['steps'])==len(r['commands'])==1 and b['steps'][0]['returncode']==0 and b['steps'][0]['argv']==r['commands'][0]['argv'],'actualTask-only command')
 need(b['artifacts'][v['task']['path']]['sha256']==v['task']['sha256']and b['CPU_rebuilt']is False and b['Engine_rebuilt']is False and b['CUDA_compilation']is False and b['native_executed']is False,'actualTask489/no numerical device or native proof claim')
 return v
