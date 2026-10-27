"""Only private health/aux modules, authenticated by this adapter's own pins.
Canonical one_bank_store.source and OneBankStore are never changed or replaced.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys
PACKET=Path(__file__).resolve().parents[1]
SOURCE=PACKET/'source'
PINS=PACKET/'DEPENDENCY_PINS.json'
VERSION=506
ALLOWED={f'gpu0_health_index_v{VERSION}.py',f'auxiliary_index_v{VERSION}.py'}
def private_source(path,name):
 path=Path(path).absolute()
 if path.parent!=SOURCE or path.name not in ALLOWED or path.is_symlink()or not path.is_file():raise RuntimeError('Private helper outside finite adapter source closure')
 rows=[r for r in json.loads(PINS.read_text())['files']if r['path']==str(path)]
 if len(rows)!=1:raise RuntimeError('Private helper requires exactly one own dependency pin')
 row=rows[0];data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
 if row['sha256']!=actual or row['bytes']!=len(data):raise RuntimeError('Private helper dependency pin changed')
 existing=sys.modules.get(name)
 if existing is not None:
  if getattr(existing,'__file__',None)!=str(path)or getattr(existing,'_private_source_sha256',None)!=actual:raise RuntimeError('Private helper module-name collision')
  return existing
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 module._private_source_sha256=actual;sys.modules[name]=module
 try:exec(compile(data,str(path),'exec'),module.__dict__)
 except BaseException:sys.modules.pop(name,None);raise
 return module
