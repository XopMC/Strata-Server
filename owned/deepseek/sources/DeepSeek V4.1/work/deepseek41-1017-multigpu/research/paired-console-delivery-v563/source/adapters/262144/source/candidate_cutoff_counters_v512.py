"""Typed actual native Handle snapshots. No reset and no Engine pointer cast."""
import ctypes as C

def need(v,m):
 if not v:raise RuntimeError('382 actual375 state: '+m)
def read(engine):
 need(engine.handle is not None,'Engine must be alive')
 fn=engine.lib.ds375_native_engine_candidate_info
 fn.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int
 out=(C.c_uint64*24)();rc=fn(engine.handle,out,24)
 need(rc==0,'typed getter failed rc='+str(rc));return list(out)
def validate(v,*,initial=False,completed=False):
 need(len(v)==24 and v[:6]==[375,1,33280,262144,8,2048],'latched actual256 geometry/workspace')
 need(v[17:20]==[0,1,0] and v[20:]==[0]*4,'actual device0/headful/copy completion/reserved')
 need(v[13:17]==[0]*4,'device status/reserved')
 if initial:need(v[6:17]==[0]*11,'first-decode state must be empty')
 from serving_validation_v529 import candidate_state375
 candidate_state375(v,completed=completed)
 return v
