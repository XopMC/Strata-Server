"""Explicit admission-only501 rebuild; original numerical receipts stay historical."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1]
PHASE='ROOT501_INDEX_SELECTION_BUILD_PASS_NATIVE_PENDING'
def need(v,m):
 if not v:raise RuntimeError('index501: '+m)
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink(),'canonical pin '+str(p));need(hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'changed '+str(p));return p
def check():
 v=json.loads((P/'INDEX_NATIVE_BINDING.json').read_text());need(v.get('ready')is True and v['selected_system_indices']==[0,1]and v['persistent_UUID_allowlist']is False,'Root501 binding pending')
 b=json.loads(pin(v['build']).read_text());r=json.loads(pin(v['recipe']).read_text());s=json.loads(pin(v['source_check']).read_text());pin(v['source_pins'])
 need(b['phase']==PHASE and b['native_or_GPU_run']is False and b['numerical_source_bodies_changed']is False,'actual501 compile/link-only receipt')
 need(b['source_recipe_sha256']==v['recipe']['sha256']and len(b['steps'])==len(r['commands'])==18,'exact501 recipe')
 need(all(x['exit_code']==0 and x['argv']==y['argv']for x,y in zip(b['steps'],r['commands'])),'all501 executed commands')
 need(s['phase']=='SOURCE_ONLY_ADMISSION_REGIONS_EXCLUDED_ALL_OTHER_TEXT_EXACT'and len(s['translation_units'])==10 and all(x['numerical_body_text_unchanged']is True for x in s['translation_units']),'normalized501 numerical body preservation')
 for row in v['contexts'].values():
  p=pin(row);need(b['artifacts'][str(p)]['sha256']==row['sha256'],'actual501 artifact')
 bank=json.loads((P/'NUMERICAL_PARENT_BANK_BINDING.json').read_text());need(bank.get('ready')is True and bank['bank_format_binding']==v['bank_format_binding'],'historical501 numerical-bank witness; live capability is independently asserted by serving514')
 need(v['context']in(262144,524288),'finite faithful context')
 return v
def overlay(value):
 v=check();key=str(v['context'])
 need(set(value['contexts'])=={key},'factory context scope')
 from k5_composite_binding_v512 import check as k5_check
 k5=k5_check(v)
 from serving_binding_v529 import overlay as serving_overlay
 return serving_overlay(dict(value,historical_contexts=value['contexts'],contexts={key:k5['contexts'][key]},index_native_admission501=v,K5_candidate503=k5))
def record():
 v=check()
 from k5_composite_binding_v512 import record as k5_record
 return dict(admission_foundation501_build=v['build'],admission_foundation501_contexts=v['contexts'],K5_candidate503=k5_record(v),selected_context=v['context'],selected_system_indices=[0,1],persistent_UUID_allowlist=False,numerical_body_text_unchanged=True,native_parity_pending=True,quality_admitted=False,performance_admitted=False,historical_proofs_are_not501_native_qualification=True)
