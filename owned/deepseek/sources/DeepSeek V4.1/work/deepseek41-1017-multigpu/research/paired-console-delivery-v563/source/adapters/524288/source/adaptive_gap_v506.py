"""Actual387 typed Handle control. Import-safe stdlib only; calls only explicitly."""
import ctypes as C
import os
#Exact fields generated from frozen370 Receipt370, not guessed Python word offsets.
PREFIX=['version', 'struct_bytes', 'adaptive', 'component_only']
DETAIL=['catalog', 'physical', 'active', 'spares', 'static_entries', 'pinned_bytes', 'armed', 'completed_forwards', 'observer_calls', 'observer_failures', 'service_calls', 'service_refusals', 'service_wall_ns', 'service_thread_ns', 'planner_wall_ns', 'planner_thread_ns', 'planner_group_visits', 'planner_limited', 'stage_wall_ns', 'stage_thread_ns', 'started', 'upload_completed', 'published', 'nonready_fallbacks', 'protected_ready', 'H2D_bytes', 'inverse_read_bytes', 'inverse_write_bytes', 'upload_event_ns', 'pending', 'stages_inflight', 'close_cancelled', 'positive_drains', 'closed', 'failed']
class Receipt370(C.Structure):
 _fields_=[(n,C.c_uint64)for n in PREFIX]+[('whole',C.c_uint64*45)]+[(n,C.c_uint64)for n in DETAIL]
if C.sizeof(Receipt370)!=672 or Receipt370.whole.offset!=32 or Receipt370.catalog.offset!=392:raise RuntimeError('386 exact370 receipt ABI layout')
def need(v,m):
 if not v:raise RuntimeError('386 adaptive Handle: '+m)
class GapControl386:
 def __init__(self,engine,*,generation,format_epoch,adaptive):
  need(engine.handle is not None and type(generation)is int and generation>0 and type(format_epoch)is int and format_epoch>0 and type(adaptive)is bool,'actualHandle/binding/mode')
  self.engine=engine;self.generation=generation;self.format_epoch=format_epoch;self.adaptive=adaptive
  self.armed=False;self.closed=False;self.nonce=0;self.called=0;self.actual_services=0;self.initial_skips=0;self.closed_receipt=None
  self.arm_fn=engine.lib.ds387_native_engine_adaptive_arm;self.arm_fn.argtypes=[C.c_void_p,C.c_uint64,C.c_uint64];self.arm_fn.restype=C.c_int
  self.gap_fn=engine.lib.ds387_native_engine_adaptive_gap;self.gap_fn.argtypes=[C.c_void_p,C.c_uint64,C.c_uint64,C.c_uint64,C.c_uint,C.POINTER(C.c_uint)];self.gap_fn.restype=C.c_int
  self.info_fn=engine.lib.ds387_native_engine_adaptive_receipt;self.info_fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];self.info_fn.restype=C.c_int
  self.close_fn=engine.lib.ds387_native_engine_adaptive_close;self.close_fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];self.close_fn.restype=C.c_int
 def live(self):need(not self.closed and self.engine.handle is not None,'actualEngine stillalive/notterminalclosed')
 def main(self):need(os.sched_getaffinity(0)=={255},'ownmain255 required atservice boundary')
 def decode(self,v):
  need((v.version,v.struct_bytes,v.adaptive,v.component_only)==(370,672,int(self.adaptive),0),'actualproductionOwner mode/ABI')
  need((v.catalog,v.physical,v.active,v.spares,v.static_entries)==(15360,1078,1070,8,0),'actualproduction catalog/capacity')
  return {**{n:int(getattr(v,n))for n in PREFIX+DETAIL},'whole_raw':list(v.whole),'whole_schema':'authoritative ds299 StatCount45 inside Receipt370; not duplicate GPU147 schema'}
 def snapshot(self):
  self.live();v=Receipt370();rc=self.info_fn(self.engine.handle,C.byref(v),C.sizeof(v));need(rc==0,'actualreceipt rc='+str(rc));return self.decode(v)
 def arm(self):
  self.live();self.main();need(not self.armed,'onearm only')
  rc=self.arm_fn(self.engine.handle,self.generation,self.format_epoch);need(rc==0,'arm rc='+str(rc));self.armed=True
  return self.snapshot()
 def gap(self):
  raise RuntimeError('470 legacy gap(max2) disabled; fixed-plan worker only')
 def record(self):
  need(self.called==self.actual_services==self.initial_skips==self.nonce==0,'470 forbids legacy per-proposal gap service')
  return dict(callback_calls=0,actual_service_calls=0,last_nonce=0,scope='NO proposal gap callback; uploader253 handles one fixed plan; real387 Handle ARM/receipt/close only')
 def close(self):
  if self.closed:return self.closed_receipt
  self.live();v=Receipt370();rc=self.close_fn(self.engine.handle,C.byref(v),C.sizeof(v));need(rc==0,'positiveclose rc='+str(rc));r=self.decode(v)
  need(r['closed']==1 and r['positive_drains']>=1 and not any(r[k]for k in('pending','stages_inflight','pinned_bytes')),'actualpositiveclosed/noDMA/pins')
  self.closed=True;self.closed_receipt=r;return r
