"""CPU418: actual143 paired139 leaves; two local pools, explicit248workers (retained252worker foundation)."""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json,os
from bulk_binding_v512 import check,FLAG,record as bulk_record
P=Path(__file__).resolve().parents[1]
def admission():
 b=check();v=json.loads((P/'CPU_BINDING.json').read_text());need(v['schema']==442 and v['ready']is True and v['library']==b['cpu'],'actual418 CPU selection')
 for row in v['reference143'].values():need(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],'unchanged143 arithmetic reference')
 return v
def need(v,m):
 if not v:raise RuntimeError('425 CPU: '+m)
@dataclass(frozen=True)
class Binding:
 mode:str='0'
 @property
 def library(self):return Path(admission()['library']['path'])
 def record(self):
  v=admission();return dict(path=v['library']['path'],sha256=v['library']['sha256'],mode='0',environment_key='DS1017_CPU_GROUP_SUM_V2',layout_tag=139,CPU_schema=418,leaf_math_schema=143,partition='two home-local bulk pipelines,248physicalworkers (retained252worker foundation)',post_lifetime_fix=94,actual418=bulk_record(),original143_provenance=v['reference143'],quality_admitted=False,model_speed_admitted=False,proof_scope='synthetic-correctness-only',model_pending=True)
def resolve_cpu_projection():
 # Called by controller BEFORE factory: choose/prove first, then latch explicitly.
 admission()
 for k,w in {**FLAG,'DS1017_CPU_GROUP_SUM_V2':'0','DS1017_CPU_NODE_LOCAL_QUEUE_V80':'0'}.items():
  need(os.environ.get(k)in(None,w),'conflicting immutable '+k);os.environ[k]=w
 for k in('CUDA_MODULE_LOADING','CUDA_MODULE_DATA_LOADING'):need(os.environ.get(k)in(None,'LAZY'),'retain matched LAZY')
 return Binding()
