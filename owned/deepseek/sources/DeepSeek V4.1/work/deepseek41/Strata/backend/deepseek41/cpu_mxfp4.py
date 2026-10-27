"""Bounded native CPU MXFP4 GEMV component; no CUDA or automatic compilation."""
import ctypes
import os
from functools import lru_cache
from pathlib import Path
import numpy as np

@lru_cache(maxsize=64)
def physical_worker_ids(cpus):
    result=[]
    for cpu in cpus:
        path=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        result.append(((path/'physical_package_id').read_text().strip(),(path/'core_id').read_text().strip()))
    return tuple(result)

class NativeCpuMxFp4:
    def __init__(self,library):
        library = Path(library).resolve()
        root = Path(__file__).resolve().parents[3]
        if not library.is_relative_to(root/'build') or not library.is_file():
            raise ValueError('Expected this project\'s explicit CPU component build')
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.ds41_mxfp4_gemv
        pointer = ctypes.c_void_p
        self.function.argtypes = [pointer,pointer,pointer,pointer,*([ctypes.c_int]*4)]
        self.function.restype = ctypes.c_int
        self.pinned=getattr(self.library,'ds41_mxfp4_gemv_pinned',None)
        if self.pinned is not None:
            self.pinned.argtypes=[pointer,pointer,pointer,pointer,*([ctypes.c_int]*4),pointer,pointer,pointer]
            self.pinned.restype=ctypes.c_int

    def gemv(self,codes,scales,x,*,threads=8,vectorized=True,worker_cpus=None,work_statistics=None):
        if codes.dtype != np.uint8 or scales.dtype != np.uint8 or x.dtype != np.float32:
            raise ValueError('Expected native packed uint8 codes/E8M0 and float32 activation')
        if codes.ndim != 2 or x.ndim != 1 or len(x)%32 or codes.shape[1]*2 != len(x):
            raise ValueError('Incomplete or inconsistent native matrix/activation groups')
        if scales.shape != (codes.shape[0],len(x)//32) or not codes.shape[0]:
            raise ValueError('E8M0 scale shape differs from native32-value groups')
        if codes.nbytes > 64*1024**2 or type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError('CPU component tile/physical-worker budget exceeded')
        if type(vectorized) is not bool:
            raise ValueError('Expected a boolean SIMD mode')
        codes,scales,x = (np.ascontiguousarray(v) for v in (codes,scales,x))
        output = np.empty(codes.shape[0],dtype=np.float32)
        arguments=(codes.ctypes.data,scales.ctypes.data,x.ctypes.data,output.ctypes.data,codes.shape[0],len(x),threads,int(vectorized))
        if worker_cpus is None:status=self.function(*arguments)
        else:
            allowed=os.sched_getaffinity(0)
            if (self.pinned is None or len(worker_cpus)!=threads or len(set(worker_cpus))!=threads or
                any(type(cpu) is not int or cpu not in allowed for cpu in worker_cpus)):
                raise ValueError('Pinned workers require unique explicitly allowed physical CPUs and the v2 library')
            physical=physical_worker_ids(tuple(worker_cpus))
            if len(set(physical))!=threads:raise ValueError('SMT siblings cannot count as additional physical workers')
            cpus=np.ascontiguousarray(worker_cpus,dtype=np.int32);done=np.zeros(threads,dtype=np.uint64);actual=np.full(threads,-1,dtype=np.int32)
            status=self.pinned(*arguments,cpus.ctypes.data,done.ctypes.data,actual.ctypes.data)
            if not status and (int(done.sum())!=codes.shape[0] or np.any(actual!=cpus)):
                raise ValueError('Actual CPU worker row/affinity proof differs from requested placement')
            if not status and work_statistics is not None:
                work_statistics.append({'cpu_rows':{str(int(cpu)):int(rows) for cpu,rows in zip(actual,done) if rows},
                    'total_output_rows':int(done.sum()),'columns':len(x),'physical_workers':threads})
        if status:
            raise ValueError('Native CPU GEMV rejected the input/device: status'+str(status))
        if not np.isfinite(output).all():
            raise ValueError('Native CPU GEMV produced nonfinite output')
        return output
