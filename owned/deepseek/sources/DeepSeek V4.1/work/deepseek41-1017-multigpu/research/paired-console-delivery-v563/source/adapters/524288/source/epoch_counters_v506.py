"""Real loaded Task427 typed receipt. Never accepts an Engine/Task pointer."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import ctypes as C,os,json,uuid
from pathlib import Path
_RECEIPTS=[]
def receipts():return list(_RECEIPTS)
def need(v,m):
 if not v:raise RuntimeError('429 actualTask427: '+m)
def validate(v,*,completed=False):
 need(len(v)==16 and v[:3]==[427,1,10000000]and v[12:]==[0,0,127,254],'native mode/window/error/idle/CPU')
 need(v[3]==v[4]==v[5]==v[10]==v[11],'publication/claim/completion conservation')
 if completed:need(v[3]>0 and v[6]+v[7]>0,'actual jobs and active polling observed')
 return v
def read(*,completed=False):
 from bulk_binding_v506 import check
 b=check();lib=C.CDLL(b['task']['path'],mode=os.RTLD_NOW|os.RTLD_NOLOAD);fn=lib.ds427_moe_epoch_info;fn.argtypes=[C.c_void_p,C.c_int];fn.restype=C.c_int
 out=(C.c_uint64*16)();rc=int(fn(out,16));v=list(map(int,out));m=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'));d=m.parents[1]/'outputs'/m.name/'cpu-active-epoch-v436';d.mkdir(parents=True,exist_ok=True);p=d/(str(os.getpid())+'-'+uuid.uuid4().hex+'.json')
 with p.open('x',encoding='utf-8')as f:json.dump(dict(schema=427,scope='raw_actual_Task427_before_validation',native_rc=rc,words=v),f);f.write('\n');f.flush();os.fsync(f.fileno())
 _RECEIPTS.append(str(p));need(rc==0,'getter idle N1 rc='+str(rc));return validate(v,completed=completed)
def generation_delta(a,b,bulk_delta):
 validate(a);validate(b,completed=True);d=[y-x for x,y in zip(a,b)]
 need(all(x>=0 for x in d[3:12]),'monotonic counters')
 need(d[3]==d[4]==d[5]==d[10]==d[11]==bulk_delta[6]and d[3]>0,'actual same bulk jobs')
 need(d[6]+d[7]>0,'generation actual fast polling')
 return d
