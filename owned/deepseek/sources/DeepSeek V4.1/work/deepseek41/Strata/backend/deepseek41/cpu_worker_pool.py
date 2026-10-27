"""Lifetime-owned native CPU pool, one pinned physical thread per worker."""
import ctypes
import os
from pathlib import Path
import threading
import numpy as np
from .cpu_mxfp4 import physical_worker_ids


class NativeCpuMxFp4Pool:
    def __init__(self,library,worker_cpus):
        self.lock=threading.RLock();self.handle=None;self.cpus=tuple(worker_cpus)
        allowed=os.sched_getaffinity(0)
        if (not 1<=len(self.cpus)<=256 or len(set(self.cpus))!=len(self.cpus) or
                any(type(cpu) is not int or cpu not in allowed for cpu in self.cpus) or
                len(set(physical_worker_ids(self.cpus)))!=len(self.cpus)):
            raise ValueError('Persistent pool needs unique allowed physical CPU workers')
        library=Path(library).resolve();root=Path(__file__).resolve().parents[3]
        if not library.is_relative_to(root/'build') or not library.is_file():raise ValueError('Expected explicit project CPU pool build')
        self.library=ctypes.CDLL(str(library));pointer=ctypes.c_void_p
        create=self.library.ds41_mxfp4_pool_create;create.argtypes=[pointer,ctypes.c_int,pointer];create.restype=pointer
        self.function=self.library.ds41_mxfp4_pool_gemv
        self.function.argtypes=[pointer]*5+[ctypes.c_int]*3+[pointer]*2;self.function.restype=ctypes.c_int
        self.destroy=self.library.ds41_mxfp4_pool_destroy;self.destroy.argtypes=[pointer];self.destroy.restype=None
        cpus=np.array(self.cpus,dtype=np.int32);status=ctypes.c_int()
        handle=create(cpus.ctypes.data,len(cpus),ctypes.byref(status))
        if not handle or status.value:raise ValueError('Persistent CPU worker initialization failed: '+str(status.value))
        self.handle=handle

    def gemv(self,codes,scales,x,*,threads=None,vectorized=True,worker_cpus=None,work_statistics=None):
        with self.lock:
            if self.handle is None:raise ValueError('Persistent CPU workers already closed')
            if threads is not None and threads!=len(self.cpus):raise ValueError('Persistent worker count differs from its placement')
            if worker_cpus is not None and tuple(worker_cpus)!=self.cpus:raise ValueError('Persistent worker CPUs differ from its placement')
            if (codes.dtype!=np.uint8 or scales.dtype!=np.uint8 or x.dtype!=np.float32 or codes.ndim!=2 or x.ndim!=1 or
                len(x)%32 or len(x)<32 or codes.shape[1]*2!=len(x) or not codes.shape[0] or
                scales.shape!=(codes.shape[0],len(x)//32) or codes.nbytes>64*1024**2 or type(vectorized) is not bool):
                raise ValueError('Invalid bounded persistent CPU matrix/activation geometry')
            codes,scales,x=(np.ascontiguousarray(v)for v in (codes,scales,x))
            output=np.empty(codes.shape[0],dtype=np.float32);done=np.zeros(len(self.cpus),dtype=np.uint64);actual=np.full(len(self.cpus),-1,dtype=np.int32)
            status=self.function(self.handle,codes.ctypes.data,scales.ctypes.data,x.ctypes.data,output.ctypes.data,
                codes.shape[0],len(x),int(vectorized),done.ctypes.data,actual.ctypes.data)
            if status:raise ValueError('Persistent native CPU GEMV rejected input/device: status'+str(status))
            if int(done.sum())!=codes.shape[0] or tuple(map(int,actual))!=self.cpus or not np.isfinite(output).all():
                raise ValueError('Persistent CPU output/worker placement proof differs')
            if work_statistics is not None:
                work_statistics.append({'cpu_rows':{str(cpu):int(rows)for cpu,rows in zip(self.cpus,done)if rows},
                    'total_output_rows':int(done.sum()),'columns':len(x),'physical_workers':len(self.cpus),'worker_backend':'persistent-native'})
            return output

    def close(self):
        with self.lock:
            if self.handle is not None:
                handle=self.handle;self.handle=None;self.destroy(handle)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        if getattr(self,'handle',None) is not None:self.close()
