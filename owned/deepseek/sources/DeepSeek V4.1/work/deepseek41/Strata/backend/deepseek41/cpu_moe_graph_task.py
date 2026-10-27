"""Complete bounded CPU experts, ready for GPU stream/graph synchronization.

No CUDA API is called at import, construction, or synchronous CPU execution.
SwiGLU's C++ exp reduction is an arithmetic candidate, not a quality approval.
"""
import ctypes
from pathlib import Path
import threading
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


class CpuMoeTasks:
    def __init__(self,pool,*,a8=False,library=None):
        if type(a8)is not bool or getattr(pool,'handle',None)is None or not pool.cpus:
            raise ValueError('MoE tasks need one live admitted projection pool')
        from .cpu_batch_pool import NativeCpuBatchPool
        from .wxa8_cpu_batch import Wxa8CpuBatchPool
        if not isinstance(pool,Wxa8CpuBatchPool if a8 else NativeCpuBatchPool):
            raise ValueError('MoE task projection ABI differs from selected arithmetic')
        path=Path(library or ROOT/'build/libdeepseek41_cpu_moe_graph_task_v3.so')
        if path.is_symlink()or not path.resolve().is_relative_to(ROOT/'build')or not path.is_file():
            raise ValueError('MoE task requires a separately built project library')
        self.pool=pool;self.a8=a8;self.owner=threading.get_ident();self.tasks=[];self.handle=None
        self.library=ctypes.CDLL(str(path));P=ctypes.c_void_p;I=ctypes.c_int;U=ctypes.c_uint64
        signatures={
            'ds41_moe_engine_create':([P,ctypes.c_size_t,I,I,I],P),
            'ds41_moe_engine_destroy':([P],None),
            'ds41_moe_task_create':([P,P,P,P,I,I,I,I,I,ctypes.c_float,P,P,P,P,P],P),
            'ds41_moe_task_run':([P,P],I),'ds41_moe_task_status':([P,P],I),
            'ds41_moe_engine_work':([P,P,I],I),'ds41_moe_fp8_quant':([P,P,I],I),
            'ds41_moe_map_host':([ctypes.c_size_t,P],I),
            'ds41_moe_stream_submit':([ctypes.c_size_t,U,U],I),
            'ds41_moe_stream_wait':([ctypes.c_size_t,U],I),
            'ds41_moe_host_submit':([ctypes.c_size_t,P],I),
            'ds41_moe_host_wait':([ctypes.c_size_t,P],I)}
        for name,(args,result)in signatures.items():
            f=getattr(self.library,name);f.argtypes=args;f.restype=result
        function=ctypes.cast(pool.function,ctypes.c_void_p).value
        self.handle=self.library.ds41_moe_engine_create(pool.handle,function,int(a8),len(pool.cpus),pool.cpus[0])
        if not self.handle:raise ValueError('MoE coordinator initialization failed')

    def check(self):
        if self.handle is None or threading.get_ident()!=self.owner or self.pool.handle is None:
            raise ValueError('MoE tasks require their live serial owner and projection pool')

    def add(self,projections,x,ids,routing,output,flags,*,dim,inter,limit):
        self.check()
        if(type(dim)is not int or type(inter)is not int or not 32<=dim<=8192 or not 32<=inter<=8192 or
           dim%32 or inter%32 or not isinstance(projections,(list,tuple))or not 1<=len(projections)<=384 or
           type(limit)not in(int,float)or not np.isfinite(limit)or limit<0 or
           x.dtype!=np.uint16 or x.ndim!=2 or x.shape[1]!=dim or not 1<=len(x)<=8 or
           ids.dtype!=np.int64 or ids.ndim!=2 or ids.shape[0]!=len(x)or not 1<=ids.shape[1]<=6 or
           ids.shape[1]>len(projections)or routing.dtype!=np.float32 or routing.shape!=ids.shape or
           output.dtype!=np.uint16 or output.shape!=(*ids.shape,dim)or
           flags.dtype!=np.uint64 or flags.shape!=(2,)or flags.ctypes.data%8 or len(self.tasks)>=320 or
           any(not v.flags.c_contiguous for v in(x,ids,routing,output,flags))):
            raise ValueError('Invalid bounded MoE task buffers or geometry')
        arrays=[];weights=[];scales=[];bits=[]
        for expert in projections:
            if not isinstance(expert,(tuple,list))or len(expert)!=3:raise ValueError('MoE task needs gate/up/down in source order')
            for which,(raw,scale,b)in enumerate(expert):
                n,k=(dim,inter)if which==2 else(inter,dim)
                native=b==0
                if(type(b)is not int or b not in(0,2,3,4,6,8)or(not self.a8 and not native)or
                   raw.dtype!=np.uint8 or raw.ndim!=2 or
                   raw.shape!=(n,k//2 if native else k*b//8+k//32*2)or not raw.flags.c_contiguous or
                   (native and(scale is None or scale.dtype!=np.uint8 or scale.shape!=(n,k//32)or not scale.flags.c_contiguous))or
                   (not native and scale is not None)or raw.nbytes+(scale.nbytes if native else 0)>64*1024**2):
                    raise ValueError('MoE task projection representation differs')
                arrays.extend((raw,scale));weights.append(raw.ctypes.data);scales.append(scale.ctypes.data if native else 0);bits.append(b)
        w=np.array(weights,dtype=np.uintp);s=np.array(scales,dtype=np.uintp);b=np.array(bits,dtype=np.int32)
        flags.fill(0)
        task=self.library.ds41_moe_task_create(self.handle,w.ctypes.data,s.ctypes.data,b.ctypes.data,len(projections),
            dim,inter,ids.shape[1],len(x),float(limit),x.ctypes.data,ids.ctypes.data,routing.ctypes.data,output.ctypes.data,flags.ctypes.data)
        if not task:raise ValueError('Native complete MoE task admission failed')
        result={'handle':task,'retained':(arrays,w,s,b,x,ids,routing,output,flags),
                'x':x,'ids':ids,'routing':routing,'output':output,'flags':flags,'dim':dim,'inter':inter}
        self.tasks.append(result);return result

    def run(self,task):
        self.check()
        if not any(task is t for t in self.tasks)or task['flags'][0]!=0:
            raise ValueError('CPU MoE task is unknown or already submitted asynchronously')
        rc=self.library.ds41_moe_task_run(self.handle,task['handle'])
        if rc:raise ValueError('CPU MoE task rejected input/status'+str(rc))
        return task['output']

    def quantize(self,bf16_bits):
        self.check()
        if(bf16_bits.dtype!=np.uint16 or bf16_bits.ndim!=1 or not 32<=len(bf16_bits)<=8192 or
           len(bf16_bits)%32 or not bf16_bits.flags.c_contiguous):raise ValueError('Invalid per32 BF16 activation')
        out=np.empty(len(bf16_bits),dtype=np.float32)
        rc=self.library.ds41_moe_fp8_quant(bf16_bits.ctypes.data,out.ctypes.data,len(out))
        if rc:raise ValueError('Native per32 activation rejected/status'+str(rc))
        return out

    def statistics(self):
        self.check();reports=[]
        for task in self.tasks:
            served=ctypes.c_uint64();rc=self.library.ds41_moe_task_status(task['handle'],ctypes.byref(served))
            if rc:raise ValueError('Asynchronous MoE task failed/status'+str(rc))
            reports.append(int(served.value))
        counts=np.empty(len(self.pool.cpus),dtype=np.uint64)
        rc=self.library.ds41_moe_engine_work(self.handle,counts.ctypes.data,len(counts))
        if rc:raise ValueError('MoE coordinator physical affinity failed')
        return {'complete_task_served_counts':reports,'verified_worker_output_rows':dict(zip(map(str,self.pool.cpus),map(int,counts))),
                'source_activation_group_size':32,'routing_before_hidden_BF16':True,
                'SwiGLU_Cpp_exp_arithmetic_candidate':True,'cpu_wxa8':self.a8,'quality_gate_passed':False}

    def gpu_submit(self,task,stream,*,mode):
        self.check()
        if mode not in('memops','hostfunc')or not any(task is t for t in self.tasks):raise ValueError('Unknown GPU synchronization task/mode')
        if mode=='hostfunc':rc=self.library.ds41_moe_host_submit(stream,task['handle'])
        else:
            if 'device_flags'not in task:
                mapped=ctypes.c_uint64();rc=self.library.ds41_moe_map_host(task['flags'].ctypes.data,ctypes.byref(mapped))
                if rc:raise ValueError('GPU mapped flags are unsupported/status'+str(rc))
                task['device_flags']=int(mapped.value)
            rc=self.library.ds41_moe_stream_submit(stream,task['device_flags'],task['device_flags']+8)
        if rc:raise ValueError('GPU MoE submit failed/status'+str(rc))

    def gpu_wait(self,task,stream,*,mode):
        self.check()
        if mode=='hostfunc':rc=self.library.ds41_moe_host_wait(stream,task['handle'])
        elif mode=='memops'and'device_flags'in task:rc=self.library.ds41_moe_stream_wait(stream,task['device_flags']+8)
        else:raise ValueError('GPU wait requires its already submitted admitted mode')
        if rc:raise ValueError('GPU MoE wait failed/status'+str(rc))

    def close(self):
        if self.handle is not None:
            self.check();handle=self.handle;self.handle=None
            self.library.ds41_moe_engine_destroy(handle);self.tasks.clear()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
