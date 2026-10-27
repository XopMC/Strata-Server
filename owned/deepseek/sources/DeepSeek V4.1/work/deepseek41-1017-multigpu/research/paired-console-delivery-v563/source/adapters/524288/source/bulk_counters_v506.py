"""Actual Task418 process getter, via already-loaded Task DSO only."""

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
def durable_raw(rc,v):
 p=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'));out=p.parents[1]/'outputs'/p.name/'cpu-bulk418-v457';out.mkdir(parents=True,exist_ok=True)
 target=out/(str(os.getpid())+'-'+uuid.uuid4().hex+'.json')
 with target.open('x',encoding='utf-8')as f:json.dump(dict(schema=418,scope='raw_actual_Task_getter_before_validation',native_rc=rc,words=v),f);f.write('\n');f.flush();os.fsync(f.fileno())
 _RECEIPTS.append(str(target))
from bulk_binding_v506 import check
def need(v,m):
 if not v:raise RuntimeError('425 actualTask418: '+m)
def validate(v,*,completed=False):
 need(len(v)==16 and v[:6]==[418,1,123,125,127,254]and v[13:]==[0,0,0],'actual home/worker/error/active/production mode')
 if completed:need(v[6]>0 and all(v[i]>0 for i in(7,8,9,10,11,12)),'actual both-home work occurred')
 need(v[7]==v[8]==v[9]and v[10]==v[11]==v[12]and max(v[7:13])<=v[6],'home GU/hidden/DOWN conservation')
 return v
def read(_engine_library=None,*,completed=False):
 b=check();lib=C.CDLL(b['task']['path'],mode=os.RTLD_NOW|os.RTLD_NOLOAD)
 f=lib.ds418_moe_process_info;f.argtypes=[C.c_void_p,C.c_int];f.restype=C.c_int
 out=(C.c_uint64*16)();rc=int(f(out,16));v=list(map(int,out));durable_raw(rc,v);need(rc==0,'idle N1 getter rc='+str(rc));return validate(v,completed=completed)
def generation_delta(a,b):
 validate(a);validate(b,completed=True);d=[y-x for x,y in zip(a,b)]
 need(all(x>=0 for x in d[6:13])and d[6]>0 and all(d[i]>0 for i in(7,8,9,10,11,12)),'actual generation both-home positive work')
 need(d[7]==d[8]==d[9]and d[10]==d[11]==d[12]and max(d[7:13])<=d[6],'generation phase conservation')
 return d
