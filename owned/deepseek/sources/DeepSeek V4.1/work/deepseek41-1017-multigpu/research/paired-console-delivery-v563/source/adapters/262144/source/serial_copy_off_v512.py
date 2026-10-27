"""Actual351 serial-only path must remain OFF in matched bare wavefront."""
import ctypes as C
def words(v):
 v=list(v)
 if v!=[351]+[0]*11:raise RuntimeError('353 wavefront requires actual351 serial-copy counters allOFF/zero')
 return v
def read(library,policy,settings):
 policy.assert_current(settings)
 if dict(policy.environment).get('DS1017_SERIAL_RANK_STAGE_COPY_V351')!='0':raise RuntimeError('353 immutable serial351 flag must be0')
 fn=library.ds351_serial_stage_copy_info;fn.argtypes=[C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int;out=(C.c_uint64*12)()
 if fn(out,12):raise RuntimeError('353 actual351 serial-copy getter failed')
 value=words(out);policy.assert_current(settings);return value
