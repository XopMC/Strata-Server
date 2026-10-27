"""Stdlib-only source binding fragment; no device/native import on import."""
WORKERS=tuple(range(64))+tuple(range(68,127))+tuple(range(128,253))
RESERVED=(64,65,66,67)
GATHER=(0,32,68,96,128,160,192,224)
COORDINATORS=(127,254)
ROLE_CPUS=(127,253,254,255)
HEADER=(539,1,248,123,125,127,254,253,255,68,4,64,65,66,67,0)
EXPECTED=HEADER+WORKERS
FLAGS={'DS1017_UPLOAD_RESERVED_COHORT_V539':'1'}
def need(v,m):
 if not v:raise RuntimeError('cohort539: '+m)
def latch():
 import os
 for k,v in FLAGS.items():need(os.environ.get(k)in(None,v),'conflicting '+k);os.environ[k]=v
def validate_settings(s):
 need(tuple(s.workers)==WORKERS,'exact248 workers; no64..67')
 need(s.coordinator_cpu==255 and s.expert_coordinator_cpu==254,'unchanged issuer roles')
def startup(library):
 import ctypes as C
 f=library.ds539_worker_cohort_info;f.argtypes=[C.POINTER(C.c_uint64),C.c_int];f.restype=C.c_int
 out=(C.c_uint64*264)();rc=int(f(out,264));need(rc==0 and tuple(out)==EXPECTED,'actual constructor capability/CPU IDs')
 return dict(schema=539,words=list(out),scope='validated constructor/requested CPUs only; not executed work proof')
def validate_bulk(v):
 need(len(v)==16 and list(v[:6])==[418,1,123,125,127,254]and list(v[13:])==[0,0,0],'actual Task418 dynamic counts/error/active/mode')
 return v
