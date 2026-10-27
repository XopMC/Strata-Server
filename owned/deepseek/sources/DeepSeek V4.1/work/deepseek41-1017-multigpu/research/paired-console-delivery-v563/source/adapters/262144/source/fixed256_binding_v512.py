"""Actual440 authentic256 Engine381 plus explicit typed387 lease host seam."""
import json,hashlib
from pathlib import Path
PHASE='ROOT440_ALL5_MATCHED_BUILD_STEPS_PASS_256_MODEL_PENDING'
def check_fixed(v):
 from bulk_binding_v512 import need,pin
 e=v['fixed256440'];b=json.loads(pin(e['build']).read_text());r=json.loads(pin(e['recipe']).read_text());mf=pin(e['source_pins'])
 need(b['phase']==PHASE and b['source_pins_sha256']==e['source_pins']['sha256'],'actual440 phase/source')
 need(len(b['steps'])==len(r['commands'])==5 and all(x['returncode']==0 and x['argv']==y['argv'] and x['cwd']==y['cwd']for x,y in zip(b['steps'],r['commands'])),'exact five440 matched build steps')
 need(b['artifacts']['Engine']['path']==v['engine']['path'] and b['artifacts']['Engine']['sha256']==v['engine']['sha256'],'actual440 selectedEngine')
 for row in json.loads(mf.read_text())['files']:
  q=Path(row['path']);q=q if q.is_absolute()else mf.parent/q
  need(hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'],'frozen440 source '+str(q))
 from balanced_hash_cache_v512 import verify_once
 rows=json.loads((mf.parent/'INPUT_PINS.json').read_text())['files'];resolved={};aliases=[]
 for row in rows:
  original=Path(row['path']);canonical=original.resolve(strict=True)
  # Root440's system-library aliases are explicit manifest inputs. Follow ONLY
  # their current canonical targets, include the alias->target map in immutable
  # policy, and retain the original declared digest/extent. Retargeting refuses.
  aliases.append((str(original),str(canonical)))
  r=dict(row,path=str(canonical));prior=resolved.setdefault(str(canonical),r)
  need(prior==r,'conflicting aliases for same canonical input')
 verify_once(list(resolved.values()),policy=(tuple(sorted(v['flag'].items())),v['engine']['path'],v['engine']['sha256'],e['source_pins']['sha256'],tuple(aliases)),author_pins=Path(__file__).resolve().parents[1]/'AUTHOR_SOURCE_PINS.json',seal=Path(__file__).resolve().parents[1]/'SEALED.json')
 return e
