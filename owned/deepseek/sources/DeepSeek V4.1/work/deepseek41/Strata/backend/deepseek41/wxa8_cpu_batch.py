"""Explicit experimental integer-dot batches; A8 changes activation precision."""
import ctypes
import os
from pathlib import Path
import threading
import numpy as np
from .cpu_mxfp4 import physical_worker_ids


class Wxa8CpuBatchPool:
    def __init__(self,library,worker_cpus):
        self.lock=threading.RLock();self.handle=None;self.cpus=tuple(worker_cpus)
        if (not 1<=len(self.cpus)<=256 or len(set(self.cpus))!=len(self.cpus) or
            any(type(cpu)is not int or cpu not in os.sched_getaffinity(0)for cpu in self.cpus) or
            len(set(physical_worker_ids(self.cpus)))!=len(self.cpus)):
            raise ValueError('WXA8 needs unique allowed physical workers')
        root=Path(__file__).resolve().parents[3];library=Path(library)
        if library.is_symlink()or not library.resolve().is_relative_to(root/'build')or not library.is_file():
            raise ValueError('WXA8 needs an explicit separately built project library')
        self.library=ctypes.CDLL(str(library));pointer=ctypes.c_void_p
        create=self.library.ds41_wxa8_batch_create;create.argtypes=[pointer,ctypes.c_int,pointer];create.restype=pointer
        self.function=self.library.ds41_wxa8_batch_gemv
        self.function.argtypes=[pointer]*8+[ctypes.c_int,pointer,pointer];self.function.restype=ctypes.c_int
        self.destroy=self.library.ds41_wxa8_batch_destroy;self.destroy.argtypes=[pointer];self.destroy.restype=None
        ids=np.array(self.cpus,dtype=np.int32);status=ctypes.c_int()
        self.handle=create(ids.ctypes.data,len(ids),ctypes.byref(status))
        if not self.handle or status.value:raise ValueError('WXA8 initialization status'+str(status.value))

    def gemvs(self,projections,*,work_statistics=None):
        with self.lock:
            if self.handle is None:raise ValueError('WXA8 pool is closed')
            if not isinstance(projections,(list,tuple))or not 1<=len(projections)<=12:
                raise ValueError('WXA8 needs1–12 bounded projections')
            retained=[];outputs=[];formats=[];total=0;packed=0
            for raw,scale,x,bits in projections:
                if type(bits)is not int or bits not in (0,2,3,4,6,8):raise ValueError('Invalid WXA8 weight format')
                native=bits==0
                if (raw.dtype!=np.uint8 or raw.ndim!=2 or x.dtype!=np.float32 or x.ndim!=1 or
                    not 32<=len(x)<=65536 or len(x)%32 or not 1<=len(raw)<=65536 or
                    raw.shape[1]!=(len(x)//2 if native else len(x)*bits//8+len(x)//32*2) or
                    (native and (scale is None or scale.dtype!=np.uint8 or scale.shape!=(len(raw),len(x)//32))) or
                    (not native and scale is not None) or raw.nbytes>64*1024**2):
                    raise ValueError('Invalid bounded WXA8 geometry')
                total+=len(raw);packed+=raw.nbytes+(scale.nbytes if native else 0)
                if total>65536 or packed>768*1024**2:raise ValueError('WXA8 aggregate batch budget exceeded')
                retained.append((np.ascontiguousarray(raw),None if scale is None else np.ascontiguousarray(scale),np.ascontiguousarray(x)))
                outputs.append(np.empty(len(raw),dtype=np.float32));formats.append(bits)
            pointers=[np.array([v[index].ctypes.data if v[index]is not None else 0 for v in retained],dtype=np.uintp)
                      for index in range(3)]
            pointers.append(np.array([v.ctypes.data for v in outputs],dtype=np.uintp))
            rows=np.array([len(v)for v in outputs],dtype=np.int32)
            columns=np.array([len(v[2])for v in retained],dtype=np.int32);bits=np.array(formats,dtype=np.int32)
            done=np.zeros(len(self.cpus),dtype=np.uint64);actual=np.full(len(self.cpus),-1,dtype=np.int32)
            status=self.function(self.handle,*(v.ctypes.data for v in pointers),rows.ctypes.data,columns.ctypes.data,
                                 bits.ctypes.data,len(retained),done.ctypes.data,actual.ctypes.data)
            if status:raise ValueError('WXA8 batch rejected input: status'+str(status))
            if int(done.sum())!=int(rows.sum())or tuple(map(int,actual))!=self.cpus or any(not np.isfinite(v).all()for v in outputs):
                raise ValueError('WXA8 output/physical worker proof failed')
            if work_statistics is not None:
                work_statistics.append({'cpu_rows':{str(cpu):int(n)for cpu,n in zip(self.cpus,done)if n},
                                        'total_output_rows':int(rows.sum()),'projections':len(outputs),
                                        'activation_quantization':'per32 signed int8 absmax/127; nearest even',
                                        'source_activation_precision_changed':True})
            return outputs

    def close(self):
        with self.lock:
            if self.handle is not None:
                handle=self.handle;self.handle=None;self.destroy(handle)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
