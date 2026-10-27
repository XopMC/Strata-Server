"""One persistent pinned worker group for 1–12 native MXFP4 projections."""
import ctypes
import os
from pathlib import Path
import threading
import numpy as np
from .cpu_mxfp4 import physical_worker_ids


class NativeCpuBatchPool:
    def __init__(self,library,worker_cpus):
        self.lock=threading.RLock();self.handle=None;self.cpus=tuple(worker_cpus)
        if(not 1<=len(self.cpus)<=256 or len(set(self.cpus))!=len(self.cpus)or
           any(type(cpu)is not int or cpu not in os.sched_getaffinity(0)for cpu in self.cpus)or
           len(set(physical_worker_ids(self.cpus)))!=len(self.cpus)):
            raise ValueError('Batch pool requires unique allowed physical workers')
        root=Path(__file__).resolve().parents[3];library=Path(library).resolve()
        if not library.is_relative_to(root/'build')or not library.is_file():raise ValueError('Expected explicit project CPU batch build')
        self.library=ctypes.CDLL(str(library));pointer=ctypes.c_void_p
        create=self.library.ds41_mxfp4_batch_create;create.argtypes=[pointer,ctypes.c_int,pointer];create.restype=pointer
        self.function=self.library.ds41_mxfp4_batch_gemv
        self.function.argtypes=[pointer]*7+[ctypes.c_int,pointer,pointer];self.function.restype=ctypes.c_int
        self.destroy=self.library.ds41_mxfp4_batch_destroy;self.destroy.argtypes=[pointer];self.destroy.restype=None
        ids=np.array(self.cpus,dtype=np.int32);status=ctypes.c_int()
        self.handle=create(ids.ctypes.data,len(ids),ctypes.byref(status))
        if not self.handle or status.value:raise ValueError('CPU batch initialization status'+str(status.value))

    def gemvs(self,projections,*,work_statistics=None):
        with self.lock:
            if self.handle is None:raise ValueError('CPU batch pool already closed')
            if not isinstance(projections,(list,tuple))or not 1<=len(projections)<=12:
                raise ValueError('Native batch requires1–12 bounded projections')
            retained=[];outputs=[]
            for codes,scales,x in projections:
                if(codes.dtype!=np.uint8 or scales.dtype!=np.uint8 or x.dtype!=np.float32 or
                   codes.ndim!=2 or x.ndim!=1 or not 32<=len(x)<=65536 or len(x)%32 or
                   not 1<=codes.shape[0]<=65536 or codes.shape[1]*2!=len(x)or
                   scales.shape!=(codes.shape[0],len(x)//32)or codes.nbytes>64*1024**2):
                    raise ValueError('Invalid native CPU batch projection geometry')
                retained.append(tuple(np.ascontiguousarray(v)for v in(codes,scales,x)))
                outputs.append(np.empty(codes.shape[0],dtype=np.float32))
            pointers=[np.array([v[index].ctypes.data for v in retained],dtype=np.uintp)for index in range(3)]
            pointers.append(np.array([v.ctypes.data for v in outputs],dtype=np.uintp))
            rows=np.array([len(v)for v in outputs],dtype=np.int32)
            columns=np.array([len(v[2])for v in retained],dtype=np.int32)
            done=np.zeros(len(self.cpus),dtype=np.uint64);actual=np.full(len(self.cpus),-1,dtype=np.int32)
            status=self.function(self.handle,*(v.ctypes.data for v in pointers),rows.ctypes.data,columns.ctypes.data,
                len(retained),done.ctypes.data,actual.ctypes.data)
            if status:raise ValueError('Native CPU batch rejected input: status'+str(status))
            if(int(done.sum())!=int(rows.sum())or tuple(map(int,actual))!=self.cpus or
               any(not np.isfinite(v).all()for v in outputs)):
                raise ValueError('Native batch output/worker proof differs')
            if work_statistics is not None:
                work_statistics.append({'cpu_rows':{str(cpu):int(n)for cpu,n in zip(self.cpus,done)if n},
                    'total_output_rows':int(rows.sum()),'projections':len(outputs),'worker_backend':'persistent-batch-native'})
            return outputs

    def close(self):
        with self.lock:
            if self.handle is not None:
                handle=self.handle;self.handle=None;self.destroy(handle)

    def gemv(self,codes,scales,x,*,threads=None,worker_cpus=None,vectorized=True,work_statistics=None):
        if vectorized is not True or(threads is not None and threads!=len(self.cpus))or(worker_cpus is not None and tuple(worker_cpus)!=self.cpus):
            raise ValueError('CPU batch single projection requires matching SIMD workers')
        return self.gemvs([(codes,scales,x)],work_statistics=work_statistics)[0]

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        if getattr(self,'handle',None)is not None:self.close()
