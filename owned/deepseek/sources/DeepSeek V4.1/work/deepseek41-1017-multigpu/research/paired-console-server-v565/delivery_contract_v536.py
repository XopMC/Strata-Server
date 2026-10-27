"""Pure explicit529-family / ABI528 metadata checks; no DSO or factory access."""
import re
from pathlib import Path
M=Path(__file__).resolve().parents[2]
CONTEXTS=('262144','524288')
NATIVE_CONTRACT=dict(revision=528,prepare_symbol='ds528_prepare_request_v1',getter_symbol='ds528_prefill_plan_info',getter_words=16,geometry='generic-prefill-immutable-v528')
EXPECTED_MATH={
 'CPU_projection':dict(path=str(M/'build/cpu-numa-bulk-task-v418/cpu.so'),bytes=140576,sha256='7f5204fbd660804caffefe654ee3ef191a528a929d04731eee66fb932ea38d86'),
 'Task_library':dict(path=str(M/'build/whole-generation-compute-off-task-v489/task.so'),bytes=147880,sha256='6b21def7528646d3dd2ccb35cdc36c387aaa22b97f12f6fb848fb15945311eff')}
def need(ok,text):
 if not ok:raise ValueError(text)
def artifact_pin(row,context):
 need(isinstance(row,dict)and set(row)=={'path','bytes','sha256'},'exact ABI528 artifact path/size/SHA required')
 raw=row['path'];need(isinstance(raw,str),'explicit artifact path required');p=Path(raw)
 need(p.is_absolute()and str(p)==raw and '..'not in p.parts and p.is_relative_to(M/'build')and len(p.relative_to(M/'build').parts)>=3 and p.parent.name==str(context)and p.name=='libds1017_pair_runtime_v1.so','explicit per-context canonical M/build artifact path required')
 need(type(row['bytes'])is int and row['bytes']>0 and isinstance(row['sha256'],str)and re.fullmatch(r'[0-9a-f]{64}',row['sha256']),'actual ABI528 context DSO pin')
 # Filesystem regularity, canonical resolution and exact content are checked by
 # bindings.pin during execution admission. This pure helper never loads a DSO.
def header(v):
 need(type(v.get('schema'))is int and v['schema']==529 and type(v.get('native_revision'))is int and type(v.get('source_revision'))is int and v['source_revision']>=529 and v['native_revision']==528 and v.get('ready')is True,'genuine Root529-family delivery / stable ABI528 pending')
 need(v.get('factory_module')=='serving_runtime_v529'and v.get('native_contract')==NATIVE_CONTRACT,'exact529 factory and native ABI528 getter16 contract required')
 need(v.get('quality_admitted')is False and v.get('performance_admitted')is False,'no invented quality/performance admission')
 need(isinstance(v.get('contexts'),dict)and set(v['contexts'])==set(CONTEXTS),'both529 contexts required')
 for c,row in v['contexts'].items():
  artifact_pin(row['library'],c)
  for key,expected in EXPECTED_MATH.items():need(row[key]==expected,'ABI528 serving must retain exact CPU418/Task489 '+key)
 return v
def source_revision(document):
 version=document.get('native_source_revision',document.get('schema'))
 need(type(version)is int and version>=528,'actual native source revision unavailable')
 if 'source_revision'in document:
  need(type(document['source_revision'])is int and document['source_revision']==version,'ambiguous native source revision')
 return version
def provenance_revision(delivery,build,source,*,initial):
 version=source_revision(build);need(source_revision(source)==version,'native build/source freeze revision mismatch')
 if not initial or 'native_source_revision'in delivery:
  need(type(delivery.get('native_source_revision'))is int and delivery['native_source_revision']==version,'future factory must explicitly bind actual native source revision')
 return version
def receipts(build,idle,contexts):
 need(build.get('phase')=='ROOT528_NATIVE_BUILD_PASS_API_PENDING'and build.get('GPU_execution')is False and build.get('weights_changed')is False,'genuine Root ABI528 host-build receipt scope')
 steps=build.get('steps');need(isinstance(steps,list)and 6<=len(steps)<=64 and all(isinstance(x,dict)and type(x.get('exit_code'))is int and x['exit_code']==0 for x in steps),'nonempty successful Root ABI528 build steps')
 paths={row['library']['path']for row in contexts.values()}
 artifacts=build.get('artifacts');need(len(paths)==2 and isinstance(artifacts,dict)and set(artifacts)==paths,'build must qualify exactly the two explicit delivery artifacts')
 for c in CONTEXTS:
  artifact_pin(contexts[c]['library'],c);need(contexts[c]['library']==artifacts[contexts[c]['library']['path']],'delivery DSO differs from actual Root build')
 need(idle.get('phase')=='ROOT528_ACTUAL_CABI_IDLE_PASS_MODEL_PENDING'and idle.get('GPU_model_run')is False and idle.get('GPU_execution')is False and idle.get('weights_changed')is False,'genuine actual ABI528 idle CABI receipt scope')
 rows=idle.get('contexts');need(isinstance(rows,list)and len(rows)==2 and {x.get('context')for x in rows}=={262144,524288},'both actual ABI528 idle contexts')
 for row in rows:
  c=str(row['context']);need(type(row.get('exit_code'))is int and row['exit_code']==0 and row.get('library')==contexts[c]['library'],'idle must bind successful exact actual DSO')
 return True
