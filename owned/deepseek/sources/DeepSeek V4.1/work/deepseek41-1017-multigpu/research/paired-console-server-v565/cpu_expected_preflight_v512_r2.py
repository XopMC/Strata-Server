"""Exact frozen497 CPU validator AST, executed without importing its controller."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[5] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import ast,copy,hashlib,json,re
CONTROLLER=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu/research/pair-controller-index-v497/controller_v146.py'))
EXPECTED_SHA='503146dd1329bad594b2eb9d780eebebe2372011dbbf50de9cbfc70d7c1b7991'
SCOPES=('synthetic-correctness-only','synthetic148-and-retained18-matrices108-component-correctness; full-model pending')
FUNCTIONS=('_need','_cpu_group_sum','_validate_expected_cpu_projection_v174')
def validator():
 data=CONTROLLER.read_bytes()
 if hashlib.sha256(data).hexdigest()!=EXPECTED_SHA:raise RuntimeError('frozen497 controller source changed')
 tree=ast.parse(data,filename=str(CONTROLLER));nodes=[]
 for n in tree.body:
  if isinstance(n,ast.ClassDef)and n.name=='ControllerError':nodes.append(copy.deepcopy(n))
  elif isinstance(n,ast.FunctionDef)and n.name in FUNCTIONS:nodes.append(copy.deepcopy(n))
  elif isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('SHA_RE','CPU_COMPONENT_SCOPES_V174'):nodes.append(copy.deepcopy(n))
 if len(nodes)!=6:raise RuntimeError('exact497 validator/class/two constants unavailable')
 module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));ns={'Path':Path,'re':re}
 exec(compile(module,str(CONTROLLER)+'#CPU-validator-only','exec'),ns)
 if ns['CPU_COMPONENT_SCOPES_V174']!=frozenset(SCOPES):raise RuntimeError('actual497 proof_scope enum changed')
 return ns

def check_expected(expected,job):
 ns=validator();ns['_validate_expected_cpu_projection_v174'](expected,job)
 return dict(scope='actual expected CPU record checked by exact frozen497 AST; no controller import/native/device access',controller=dict(path=str(CONTROLLER),bytes=CONTROLLER.stat().st_size,sha256=EXPECTED_SHA),extracted_functions=list(FUNCTIONS),accepted_proof_scope_enums=list(SCOPES),expected_record=expected,expected_record_sha256=hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest(),job_CPU_group_sum=job.get('CPU_group_sum','0'),passed=True,native_GPU_or_model=False)
