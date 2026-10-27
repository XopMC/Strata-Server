"""Durable request-boundary evidence beforeguards; no perTask logging."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import os,json,hashlib,uuid
M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
OUT=M.parents[1]/'outputs/deepseek41-1017-multigpu/pair256-fixed-plan-v512-model'
def save(phase,whole,transport,memory=None,task489=None):
 OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
 if OUT.is_symlink()or OUT.resolve()!=OUT or OUT.stat().st_uid!=os.getuid():raise RuntimeError('unsafe boundaryreceipt output')
 p=OUT/(str(os.getpid())+'-'+phase+'-'+uuid.uuid4().hex+'.json')
 b=(json.dumps(dict(schema=408,phase=phase,pid=os.getpid(),whole=whole,transport391=transport,GPU1_memory=memory,task489=task489,scope='actualraw BEFOREboundaryguards, notmodelacceptance'),ensure_ascii=False,indent=2)+'\n').encode()
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
 with os.fdopen(fd,'wb')as h:h.write(b);h.flush();os.fsync(h.fileno())
 d=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(d)
 finally:os.close(d)
 row=dict(path=str(p),sha256=hashlib.sha256(b).hexdigest());print(json.dumps(dict(actual_static403_raw_before_guards=row)),flush=True);return row
