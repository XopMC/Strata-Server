"""Mandatory524 artifact metadata only. No512 runtime selection or DSO load."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parents[1]
def check():
 v=json.loads((P/'CONTEXT_WITNESS_BINDING.json').read_text())
 def need(ok,m):
  if not ok:raise RuntimeError('442 witness: '+m)
 def pin(r):
  p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'actual '+str(p));return p
 need(v['schema']==442 and v['context']==524288 and v['runtime_route_enabled']is False and v['model_qualification_claimed']is False,'artifact-only contract')
 b=json.loads(pin(v['build']).read_text());r=json.loads(pin(v['recipe']).read_text());pin(v['library']);pin(v['source_pins'])
 need(b['phase']=='ROOT432_ONE_HOST_LINK_PASS_MODEL_PENDING'and b['source_pins_sha256']==v['source_pins']['sha256'],'actual432 build/source')
 need(len(b['steps'])==len(r['commands'])==1 and b['steps'][0]['returncode']==0 and b['steps'][0]['argv']==r['commands'][0]['argv'],'actual432 one link')
 need(b['artifacts']['Engine']==v['library'],'actual432 artifact')
 row=v['context_row'];need(row['context']==524288 and row['DSO']==v['library']['path']and row['DSO_sha256']==v['library']['sha256']and row['carrier_only_factory_unreachable']is True,'truthfulboth-context witness')
 return v
