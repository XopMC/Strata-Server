"""Actual456 successful-window counters; NOT device launch counters."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import ctypes as C,json,os,uuid
from pathlib import Path
FIELDS=('version','mode','workspace','successful_calls','rows','eligible_many_windows','scalar_windows','reused_successes','errors','last_M','active','reserved')
def need(v,m):
 if not v:raise RuntimeError('458 actual456: '+m)
def validate(v):
 need(len(v)==12 and v[:3]==[456,1,1]and v[8]==v[10]==v[11]==0,'latchedmode1 workspace idle/error0')
 need(v[5]+v[6]==v[3]and v[3]<=v[4]<=6*v[3]and v[7]<=v[3],'successful call/row/window conservation')
 need((v[3]==0 and v[9]==0)or(v[3]>0 and 1<=v[9]<=6),'actual lastM')
 return v
def read(lib,stage,*,absent=False,allow_absent=False):
 f=lib.ds456_verify_woa_quant_info;f.argtypes=[C.POINTER(C.c_uint64),C.c_int];f.restype=C.c_int
 out=(C.c_uint64*12)();rc=int(f(out,12));values=list(map(int,out));record=dict(schema=456,scope='actual successful host verify windows after synchronized join; NOT hardware launches',stage=stage,native_rc=rc,words=values)
 m=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'));root=m.parents[1]/'outputs'/m.name/'verify-woa-quant-v458';root.mkdir(parents=True,exist_ok=True);p=root/(str(os.getpid())+'-'+uuid.uuid4().hex+'.json')
 with p.open('x',encoding='utf-8')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 record['raw_receipt_path']=str(p)
 if absent:need(rc==2,'actual workspace must be absent/closed');return record
 if allow_absent and rc==2:return record
 need(rc==0,'getter rc='+str(rc));validate(values);return record
def delta(a,b):
 need(b.get('native_rc')==0 and a.get('native_rc')in(0,2),'actual before/after native status')
 y=validate(b['words']);absent=a.get('native_rc')==2
 if absent:
  need(a['words']==[0]*12,'absent baseline has no native counter words')
  # Only arithmetic zero baseline: no fabricated workspace/mode receipt.
  differences=[y[i]for i in range(3,9)]
 else:
  x=validate(a['words']);differences=[y[i]-x[i]for i in range(3,9)]
 d=dict(zip(FIELDS[3:9],differences))
 need(all(v>=0 for v in differences)and d['successful_calls']>0 and d['eligible_many_windows']>0,'actual generation calls/eligible M2..6 positive')
 need(d['eligible_many_windows']+d['scalar_windows']==d['successful_calls']and d['successful_calls']<=d['rows']<=6*d['successful_calls']and d['reused_successes']<=d['successful_calls']and d['errors']==0,'actual generation conservation')
 return dict(d,baseline_workspace_absent=absent,baseline_scope='workspace_absent_before_generation'if absent else'actual prior idle workspace counters',last_M=y[9])
