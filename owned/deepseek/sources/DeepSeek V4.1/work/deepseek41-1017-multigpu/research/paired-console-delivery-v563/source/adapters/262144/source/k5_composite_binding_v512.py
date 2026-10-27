"""Explicit511256 host composition over actual501/503/504 evidence."""
from pathlib import Path
import json
from port_binding_v512 import check as port_check,pin,parent,call_parent,need
P=Path(__file__).resolve().parents[1]
def check(admission501):
 v=json.loads((P/'K5_COMPOSITE_BINDING.json').read_text());need(v.get('ready')is True and v['schema']=='deepseek1017-pair256-503-469-composite/512','Root512 composite pending')
 port=port_check();old501=call_parent('index_native_binding',506);leaf=call_parent('k5_composite_binding',506,'check',old501)
 need(admission501['context']==262144 and admission501['build']==old501['build']and admission501['contexts']==old501['contexts'],'actual501 foundation and finite256 selection')
 need(admission501['bank_format_binding']==v['bank_format_binding']==port['bank_format_binding'],'actual current139 owner')
 need(v['contexts']==port['contexts']and v['actual511']==port['build511'],'fresh511, not504carrier')
 need(v['leaf_archive504']==leaf['leaf_archive504']and v['parent_composite506']['sha256']==__import__('hashlib').sha256(pin(v['parent_composite506']).read_bytes()).hexdigest(),'actual503 leaf foundation')
 return v
def record(admission501):
 v=check(admission501);return dict(schema=v['schema'],active_contexts=v['contexts'],actual511=v['actual511'],parent_composite506=v['parent_composite506'],leaf_archive504=v['leaf_archive504'],native_model_pending=True,quality_admitted=False,performance_admitted=False,old501_or504_hash_is_not511_qualification=True)
