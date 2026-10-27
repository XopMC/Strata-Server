"""Explicit own-format CPU GEMV; compilation and full-model admission are separate."""
import ctypes
import os
from pathlib import Path
import threading
import numpy as np
from .cpu_mxfp4 import physical_worker_ids


class OwnCpuRows:
    def __init__(self,library,worker_cpus):
        root=Path(__file__).resolve().parents[3];library=Path(library)
        if(library.is_symlink()or not library.resolve().is_relative_to(root/'build')or not library.is_file()):
            raise ValueError('Own CPU rows require this project\'s separately built library')
        allowed=os.sched_getaffinity(0)
        if(not isinstance(worker_cpus,list)or not 1<=len(worker_cpus)<=256 or
           any(type(c)is not int or c not in allowed for c in worker_cpus)or
           len(set(worker_cpus))!=len(worker_cpus)or
           len(set(physical_worker_ids(tuple(worker_cpus))))!=len(worker_cpus)):
            raise ValueError('Own workers require explicit unique allowed physical CPUs')
        self.lock=threading.Lock();self.cpus=np.ascontiguousarray(worker_cpus,dtype=np.int32)
        self.library=ctypes.CDLL(str(library));pointer=ctypes.c_void_p
        self.create=self.library.ds41_own_pool_create;self.create.argtypes=[pointer,ctypes.c_int,pointer];self.create.restype=pointer
        self.function=self.library.ds41_own_pool_gemv
        self.function.argtypes=[pointer,pointer,pointer,pointer,*([ctypes.c_int]*4),pointer,pointer];self.function.restype=ctypes.c_int
        self.destroy=self.library.ds41_own_pool_destroy;self.destroy.argtypes=[pointer];self.destroy.restype=None
        status=ctypes.c_int();self.pool=self.create(self.cpus.ctypes.data,len(self.cpus),ctypes.byref(status))
        if status.value or not self.pool:raise ValueError('Own CPU worker creation failed: status'+str(status.value))

    def gemv(self,raw,x,bits,*,vectorized=True,work_statistics=None):
        if(raw.dtype!=np.uint8 or raw.ndim!=2 or x.dtype!=np.float32 or x.ndim!=1 or
           not 32<=len(x)<=65536 or len(x)%32 or not 1<=len(raw)<=65536 or
           type(bits)is not int or bits not in(2,3,4,6,8)or type(vectorized)is not bool or
           raw.shape[1]!=len(x)*bits//8+len(x)//32*2 or raw.nbytes>64*1024**2):
            raise ValueError('Own CPU GEMV requires bounded interleaved group32 rows and F32 activation')
        raw,x=(np.ascontiguousarray(v)for v in(raw,x));output=np.empty(len(raw),dtype=np.float32)
        done=np.zeros(len(self.cpus),dtype=np.uint64);actual=np.full(len(self.cpus),-1,dtype=np.int32)
        with self.lock:
            if not self.pool:raise ValueError('Own CPU row pool is closed')
            status=self.function(self.pool,raw.ctypes.data,x.ctypes.data,output.ctypes.data,len(raw),len(x),bits,
                int(vectorized),done.ctypes.data,actual.ctypes.data)
        if status:raise ValueError('Own CPU row GEMV failed: status'+str(status))
        if(int(done.sum())!=len(raw)or np.any(actual!=self.cpus)or not np.isfinite(output).all()):
            raise ValueError('Own CPU row output/physical-worker proof failed')
        if work_statistics is not None:
            work_statistics.append({'cpu_rows':{str(int(c)):int(n)for c,n in zip(actual,done)if n},
                'total_output_rows':int(done.sum()),'columns':len(x),'physical_workers':len(self.cpus),
                'packed_bytes':raw.nbytes,'arithmetic':'F32 decoded weights and F32 SIMD/FMA dot; different reduction order from NumPy BLAS'})
        return output

    def close(self):
        with self.lock:
            if self.pool:self.destroy(self.pool);self.pool=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
