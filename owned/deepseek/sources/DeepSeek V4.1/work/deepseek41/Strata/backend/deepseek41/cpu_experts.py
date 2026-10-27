"""Bounded CPU routed-expert execution and optional decode-only hybrid dispatch.

The native E2M1/E8M0 weights remain packed. Each projection consumes source
FP8-quantized activations and rounds its result to BF16 before SwiGLU. This
changes the reduction implementation and therefore requires checkpoint quality
comparison; component accuracy is not an end-to-end acceptance result.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import time
import types
import numpy as np
import torch
import torch.nn.functional as F

from .cpu_mxfp4 import NativeCpuMxFp4
from .portable_kernels import act_quant


class CpuExpertExecutor:
    def __init__(self,store,library,threads=8,max_rows=8,worker_cpus=None,persistent_workers=False,own_uniform_mode='numpy',batch_projections=False,cpu_wxa8=False):
        if type(threads) is not int or not 1 <= threads <= 256 or type(max_rows) is not int or not 1 <= max_rows <= 8:
            raise ValueError('Invalid bounded CPU expert workers/rows')
        self.store,self.threads,self.max_rows = store,threads,max_rows
        if own_uniform_mode not in('numpy','pinned-simd','pinned-simd-v2'):raise ValueError('Invalid own CPU expert arithmetic mode')
        if own_uniform_mode!='numpy'and(worker_cpus is None or len(worker_cpus)!=threads):
            raise ValueError('Own SIMD CPU workers require explicit physical placement')
        self.own_uniform_mode=own_uniform_mode;self.own_kernel=None
        if type(persistent_workers) is not bool:raise ValueError('CPU worker lifetime switch must be boolean')
        if type(batch_projections)is not bool or(batch_projections and not persistent_workers):
            raise ValueError('CPU batch projections require persistent physical workers')
        if type(cpu_wxa8)is not bool or(cpu_wxa8 and not(batch_projections and persistent_workers)):
            raise ValueError('WXA8 is an explicit batch/persistent CPU arithmetic candidate')
        self.cpu_wxa8=cpu_wxa8
        self.batch_projections=batch_projections;self.batch_calls=0
        if cpu_wxa8:
            from .wxa8_cpu_batch import Wxa8CpuBatchPool
            from .wxa8_cpu_admission import admit
            if worker_cpus is None or len(worker_cpus)!=threads:raise ValueError('WXA8 requires explicit physical workers')
            self.kernel=Wxa8CpuBatchPool(admit(store),worker_cpus)
        elif batch_projections:
            from .cpu_batch_pool import NativeCpuBatchPool
            from .cpu_batch_admission import admit
            if worker_cpus is None or len(worker_cpus)!=threads:raise ValueError('CPU batch requires explicit workers')
            self.kernel=NativeCpuBatchPool(admit(store),worker_cpus)
        elif persistent_workers:
            if worker_cpus is None or len(worker_cpus)!=threads:raise ValueError('Persistent CPU workers need explicit physical placement')
            from .cpu_worker_pool import NativeCpuMxFp4Pool
            self.kernel=NativeCpuMxFp4Pool(library,worker_cpus)
        else:self.kernel = NativeCpuMxFp4(library)
        self.persistent_workers=persistent_workers
        self.projections,self.experts,self.rows,self.packed_bytes = 0,0,0,0
        self.seconds = 0.
        self.worker_cpus=worker_cpus;self.cpu_worker_rows={};self.cpu_worker_packed_bytes={}

    def linear(self,name,x):
        if x.device.type != 'cpu' or x.dtype != torch.bfloat16 or x.ndim != 2 or not 1 <= len(x) <= self.max_rows:
            raise ValueError('CPU expert requires bounded two-dimensional CPU BF16 activations')
        info = self.store.info(name)
        if '.ffn.experts.' not in name or info['dtype'] not in ('I8','D41Q'):
            raise ValueError('CPU executor admits only native routed MXFP4 experts')
        if self.cpu_wxa8:
            quantized,activation_scale=act_quant(x.contiguous(),32,'ue8m0',torch.float8_e8m0fnu)
            activation=(quantized.float().unflatten(-1,(-1,32))*activation_scale.float().unsqueeze(-1)).flatten(-2).numpy()
            outputs=[]
            for vector in activation:
                work=[]
                outputs.append(self.kernel.gemvs([self._wxa8_projection(name,vector)],work_statistics=work)[0])
                self.batch_calls+=1
                for report in work:
                    for cpu,n in report['cpu_rows'].items():self.cpu_worker_rows[cpu]=self.cpu_worker_rows.get(cpu,0)+n
            return torch.from_numpy(np.stack(outputs)).to(torch.bfloat16)
        if info['dtype'] == 'D41Q':
            if info['shape'][1] != x.shape[1]:
                raise ValueError('Own candidate expert activation width differs')
            from .own_quant_rows import row_layout
            row_bytes=row_layout(info)
            quantized,activation_scale = act_quant(x.contiguous(),32,'ue8m0',torch.float8_e8m0fnu)
            activation = (quantized.float().unflatten(-1,(-1,32))*activation_scale.float().unsqueeze(-1)).flatten(-2).numpy()
            output = np.empty((len(x),info['shape'][0]),dtype=np.float32)
            if self.own_uniform_mode!='numpy':
                if self.own_kernel is None:
                    from .own_cpu_admission import admit,admit_v2
                    from .own_cpu_rows import OwnCpuRows
                    admission=admit_v2 if self.own_uniform_mode=='pinned-simd-v2'else admit
                    self.own_kernel=OwnCpuRows(admission(self.store),self.worker_cpus)
                read=getattr(self.store,'raw_view',self.store.raw)
                raw=np.frombuffer(read(name),dtype=np.uint8).reshape(info['shape'][0],row_bytes)
                for row,vector in enumerate(activation):
                    work=[];output[row]=self.own_kernel.gemv(raw,vector,info['bits'],work_statistics=work)
                    for report in work:
                        for cpu,rows in report['cpu_rows'].items():
                            self.cpu_worker_rows[cpu]=self.cpu_worker_rows.get(cpu,0)+rows
                            self.cpu_worker_packed_bytes[cpu]=self.cpu_worker_packed_bytes.get(cpu,0)+rows*row_bytes
                self.projections+=1;self.own_uniform_projections=getattr(self,'own_uniform_projections',0)+1
                self.packed_bytes+=info['bytes']
                return torch.from_numpy(output).to(torch.bfloat16)
            # Diagnostic fallback: only a bounded row tile is expanded. This
            # is NumPy F32 GEMV, not the native OpenMP SIMD implementation.
            row_tile = min(128,(16*1024**2)//(info['shape'][1]*4))
            if row_tile < 1:
                raise ValueError('Own CPU expert exceeds bounded decoded row scratch')
            for start in range(0,info['shape'][0],row_tile):
                end = min(start+row_tile,info['shape'][0])
                matrix = self.store.float_rows(name,np.arange(start,end))
                output[:,start:end] = activation@matrix.T
            self.projections += 1
            self.own_uniform_projections = getattr(self,'own_uniform_projections',0)+1
            self.packed_bytes += info['bytes']
            return torch.from_numpy(output).to(torch.bfloat16)
        scale_name = name.rsplit('.',1)[0]+'.scale'
        scale = self.store.info(scale_name)
        if scale['dtype'] != 'F8_E8M0' or x.shape[1] != info['shape'][1]*2:
            raise ValueError('CPU expert native activation/scale representation mismatch')
        if info['bytes']+scale['bytes'] > 64*1024**2:
            raise ValueError('One packed CPU expert projection exceeds64MiB')
        read = getattr(self.store,'raw_view',self.store.raw)
        codes = np.frombuffer(read(name),dtype=np.uint8).reshape(info['shape'])
        scales = np.frombuffer(read(scale_name),dtype=np.uint8).reshape(scale['shape'])
        quantized,activation_scale = act_quant(x.contiguous(),32,'ue8m0',torch.float8_e8m0fnu)
        activation = (quantized.float().unflatten(-1,(-1,32))*activation_scale.float().unsqueeze(-1)).flatten(-2).numpy()
        output=[]
        for row in activation:
            work=[]
            output.append(self.kernel.gemv(codes,scales,row,threads=self.threads,worker_cpus=self.worker_cpus,work_statistics=work))
            for report in work:
                for cpu,rows in report['cpu_rows'].items():
                    self.cpu_worker_rows[cpu]=self.cpu_worker_rows.get(cpu,0)+rows
                    self.cpu_worker_packed_bytes[cpu]=self.cpu_worker_packed_bytes.get(cpu,0)+rows*(len(row)//2+len(row)//32)
        output=np.stack(output)
        self.projections += 1
        self.packed_bytes += codes.nbytes+scales.nbytes
        return torch.from_numpy(output).to(torch.bfloat16)

    def _wxa8_projection(self,name,vector):
        info=self.store.info(name);read=getattr(self.store,'raw_view',self.store.raw)
        if info['dtype']=='I8':
            scale_name=name.rsplit('.',1)[0]+'.scale';scale_info=self.store.info(scale_name)
            if scale_info['dtype']!='F8_E8M0' or info['shape'][1]*2!=len(vector):
                raise ValueError('WXA8 source expert width/scale differs')
            raw=np.frombuffer(read(name),dtype=np.uint8).reshape(info['shape'])
            scale=np.frombuffer(read(scale_name),dtype=np.uint8).reshape(scale_info['shape']);bits=0
        elif info['dtype']=='D41Q':
            from .own_quant_rows import row_layout
            if info.get('protected_native_rows') or info['shape'][1]!=len(vector):
                raise ValueError('WXA8 cannot ignore protected native rows or changed activation width')
            raw=np.frombuffer(read(name),dtype=np.uint8).reshape(info['shape'][0],row_layout(info))
            scale=None;bits=info['bits']
            self.own_uniform_projections=getattr(self,'own_uniform_projections',0)+1
        else:raise ValueError('WXA8 only admits native or own encoded routed experts')
        if raw.nbytes+(0 if scale is None else scale.nbytes)>64*1024**2:
            raise ValueError('WXA8 projection exceeds64MiB')
        self.projections+=1;self.packed_bytes+=raw.nbytes+(0 if scale is None else scale.nbytes)
        return raw,scale,vector,bits

    @torch.inference_mode()
    def expert(self,prefix,x,weights,swiglu_limit):
        if weights.device.type != 'cpu' or weights.dtype != torch.float32 or weights.shape != (len(x),1):
            raise ValueError('CPU routed expert requires one float32 routing weight per row')
        if len(x)>self.max_rows:
            if x.device.type!='cpu'or x.ndim!=2 or x.dtype!=torch.bfloat16 or len(x)>4096:
                raise ValueError('CPU expert prefill needs bounded BF16 microbatches of at most4096rows')
            return torch.cat([self.expert(prefix,x[first:first+self.max_rows],weights[first:first+self.max_rows],swiglu_limit)
                for first in range(0,len(x),self.max_rows)])
        started = time.perf_counter()
        gate = self.linear(prefix+'.w1.weight',x).float()
        up = self.linear(prefix+'.w3.weight',x).float()
        if swiglu_limit > 0:
            up = up.clamp(-swiglu_limit,swiglu_limit)
            gate = gate.clamp_max(swiglu_limit)
        hidden = (F.silu(gate)*up*weights).to(torch.bfloat16)
        output = self.linear(prefix+'.w2.weight',hidden)
        self.experts += 1
        self.rows += len(x)
        self.seconds += time.perf_counter()-started
        return output

    def statistics(self):
        return {'workers':self.threads,'max_rows_per_expert':self.max_rows,'expert_calls':self.experts,
                'expert_rows':self.rows,'projection_calls':self.projections,
                'packed_weight_bytes_read':self.packed_bytes,'cpu_expert_seconds':self.seconds,
                'weight_representation':'native MXFP4/E8M0','activation_representation':'native FP8/E8M0 then per32 A8 candidate then BF16 output'if self.cpu_wxa8 else'native FP8/E8M0 then BF16 output',
                'cpu_wxa8':self.cpu_wxa8,'source_activation_precision_changed':self.cpu_wxa8,
                'own_uniform_projection_calls':getattr(self,'own_uniform_projections',0),
                'own_uniform_mode':self.own_uniform_mode,
                'own_uniform_implementation':'persistent pinned F32 SIMD/FMA/'+self.own_uniform_mode if self.own_uniform_mode!='numpy'else'bounded NumPy F32 GEMV; native OpenMP worker count does not apply to these calls',
                'worker_affinity':self.worker_cpus,'verified_worker_output_rows':self.cpu_worker_rows,
                'native_cpu_worker_backend':'persistent-native' if self.persistent_workers else 'openmp-per-projection',
                'batch_projections':self.batch_projections,'native_projection_batch_calls':self.batch_calls,
                'verified_worker_packed_bytes':self.cpu_worker_packed_bytes,
                'quality_gate_passed':False}

    def close(self):
        if self.own_kernel is not None:self.own_kernel.close();self.own_kernel=None
        if self.persistent_workers:self.kernel.close()

    @torch.inference_mode()
    def expert_group(self,jobs):
        """Two barriers for six single-token experts; source row math is retained."""
        if(not self.batch_projections or not 1<=len(jobs)<=6 or
           any(x.shape[0]!=1 or x.device.type!='cpu'or x.dtype!=torch.bfloat16 or
               w.shape!=(1,1)or w.dtype!=torch.float32 or w.device.type!='cpu'or
               self.store.info(prefix+'.w1.weight')['dtype'] not in (('I8','D41Q')if self.cpu_wxa8 else('I8',))for _,prefix,x,w,_ in jobs)):
            return{i:self.expert(prefix,x,w,limit)for i,prefix,x,w,limit in jobs}
        started=time.perf_counter();limits=[j[4]for j in jobs]
        if len(set(limits))!=1:raise ValueError('CPU batch expert group has inconsistent SwiGLU limit')
        def quantized(x):
            codes,scale=act_quant(x.contiguous(),32,'ue8m0',torch.float8_e8m0fnu)
            return(codes.float().unflatten(-1,(-1,32))*scale.float().unsqueeze(-1)).flatten(-2).numpy()
        activation=quantized(torch.cat([j[2]for j in jobs]));read=getattr(self.store,'raw_view',self.store.raw)
        def projection(prefix,which,vector):
            name=prefix+'.'+which+'.weight'
            if self.cpu_wxa8:return self._wxa8_projection(name,vector)
            scale_name=prefix+'.'+which+'.scale'
            info=self.store.info(name);scale_info=self.store.info(scale_name)
            if(info['dtype']!='I8'or scale_info['dtype']!='F8_E8M0'or info['shape'][1]*2!=len(vector)):
                raise ValueError('CPU batch requires native MXFP4/E8M0 projection')
            codes=np.frombuffer(read(name),dtype=np.uint8).reshape(info['shape'])
            scales=np.frombuffer(read(scale_name),dtype=np.uint8).reshape(scale_info['shape'])
            self.packed_bytes+=codes.nbytes+scales.nbytes;self.projections+=1
            return codes,scales,vector
        def multiply(values):
            work=[];output=self.kernel.gemvs(values,work_statistics=work);self.batch_calls+=1
            for report in work:
                for cpu,n in report['cpu_rows'].items():self.cpu_worker_rows[cpu]=self.cpu_worker_rows.get(cpu,0)+n
            # Each worker proves the actual number of output rows. Mixed-width
            # batch byte attribution is recorded globally, rather than guessed.
            return output
        first=[projection(j[1],p,activation[row])for row,j in enumerate(jobs)for p in('w1','w3')]
        projected=multiply(first)
        gate=torch.from_numpy(np.stack(projected[0::2])).to(torch.bfloat16).float()
        up=torch.from_numpy(np.stack(projected[1::2])).to(torch.bfloat16).float()
        if limits[0]>0:up=up.clamp(-limits[0],limits[0]);gate=gate.clamp_max(limits[0])
        routing=torch.cat([j[3]for j in jobs])
        hidden=quantized((F.silu(gate)*up*routing).to(torch.bfloat16))
        down=multiply([projection(j[1],'w2',hidden[row])for row,j in enumerate(jobs)])
        self.experts+=len(jobs);self.rows+=len(jobs);self.seconds+=time.perf_counter()-started
        return{j[0]:torch.from_numpy(value).to(torch.bfloat16).unsqueeze(0)for j,value in zip(jobs,down)}


def expert_cached(ops,prefix):
    """All three projection tiles must exist before an expert counts as a hit."""
    from .complete_residency import complete_expert_cached
    if complete_expert_cached(ops,prefix):return True
    for projection in ('w1','w2','w3'):
        name = prefix+'.'+projection+'.weight'
        info = ops.store.info(name)
        if info['dtype']=='D41Q':
            from .own_quant_rows import row_layout
            row_bytes=row_layout(info);width=info['shape'][1]
            if info.get('protected_native_rows'):return False
            chunk=min(ops._row_chunk(name),64*1024**2//(width*4))
            for start in range(0,info['shape'][0],chunk):
                end=min(start+chunk,info['shape'][0])
                key=('d41q-encoded-row-cuda-v1',name,start,end,info['bits'],width)
                with ops.weight_cache.lock:
                    entry=ops.weight_cache.entries.get(key)
                    if entry is None or entry.bytes!=(end-start)*row_bytes:return False
            continue
        scale_name = prefix+'.'+projection+'.scale'
        scale = ops.store.info(scale_name)
        chunk = min(ops._row_chunk(name),ops._row_chunk(scale_name))
        for start in range(0,info['shape'][0],chunk):
            end = min(start+chunk,info['shape'][0])
            key = (name,start,end,info['dtype'],tuple(info['shape']),scale_name,start,end,scale['dtype'],tuple(scale['shape']))
            size = (end-start)*(info['bytes']//info['shape'][0]+scale['bytes']//scale['shape'][0])
            # Inspect the stable cache storage contract directly. A resident
            # model can predate a new convenience method without reloading its
            # verified 475GiB source buffers or disturbing outstanding leases.
            with ops.weight_cache.lock:
                entry = ops.weight_cache.entries.get(key)
                if entry is None or entry.bytes != size:
                    return False
    return True


@contextmanager
def cpu_expert_dispatch(reference,*,mode='cpu-routed',threads=8,max_rows=1,worker_cpus=None,persistent_workers=False,own_uniform_mode='numpy',batch_projections=False,cpu_wxa8=False):
    """Patch only routed-expert decode; GPU shared experts overlap CPU misses.

    CPU jobs run in one background group with an explicit OpenMP budget. The
    main thread alone uses CUDA. Contributions are added in source expert order.
    Larger prefill groups use the unchanged GPU graph. hybrid-cache dispatches
    complete cached experts on GPU and promotes at most one selected cold
    expert per layer, chosen by decaying observed routing frequency.
    """
    if mode not in ('cpu-routed','hybrid-cache') or reference.device.type not in ('cuda','cpu') or (reference.device.type=='cpu' and mode!='cpu-routed'):
        raise ValueError('CPU routed dispatch requires the admitted CUDA reference')
    root = Path(__file__).resolve().parents[3]
    library='libdeepseek41_mxfp4_cpu_pool_v3.so' if persistent_workers else 'libdeepseek41_mxfp4_cpu_v2.so' if worker_cpus is not None else 'libdeepseek41_mxfp4_cpu.so'
    executor = CpuExpertExecutor(reference.store,root/'build'/library,threads,max_rows,worker_cpus=worker_cpus,persistent_workers=persistent_workers,own_uniform_mode=own_uniform_mode,batch_projections=batch_projections,cpu_wxa8=cpu_wxa8)
    pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='ds41-cpu-experts')
    originals = []
    frequencies = {}
    stats = {'mode':mode,'gpu_cached_expert_calls':0,'gpu_promoted_expert_calls':0,
             'prefill_gpu_fallback_calls':0,'cpu_gpu_overlap_dispatches':0}
    def forward(module,x,image_mask=None):
        shape = x.shape
        flat = x.reshape(-1,module.dim)
        if len(flat) > max_rows and reference.device.type=='cuda':
            stats['prefill_gpu_fallback_calls'] += 1
            return module._ds41_original_forward(x,image_mask)
        weights,indices = module.gate(flat,None if image_mask is None else image_mask.flatten())
        selected = sorted(set(indices.flatten().tolist()))
        jobs = []
        hot = frequencies.setdefault(module.layer_id,{})
        for expert in list(hot):
            hot[expert] *= .99
        for expert in selected:
            hot[expert] = hot.get(expert,0)+int((indices == expert).sum())
        prefixes = {i:f'layers.{module.layer_id}.ffn.experts.{i}' for i in selected}
        gpu = {i for i in selected if mode == 'hybrid-cache' and expert_cached(reference.ops,prefixes[i])}
        stats['gpu_cached_expert_calls'] += len(gpu)
        cold = set(selected)-gpu
        if mode == 'hybrid-cache' and cold:
            promoted = max(cold,key=lambda i:(hot[i],-i))
            gpu.add(promoted)
            stats['gpu_promoted_expert_calls'] += 1
        positions = {}
        for i in selected:
            rows,top = torch.where(indices == i)
            positions[i] = rows
            if i not in gpu:
                jobs.append((i,prefixes[i],flat[rows].detach().to('cpu'),
                             weights[rows,top,None].detach().float().to('cpu'),module.experts[i].swiglu_limit))
        def cpu_group():
            if batch_projections:return executor.expert_group(jobs)
            results={}
            for i,prefix,activation,weight,limit in jobs:
                pieces=[executor.expert(prefix,activation[first:first+max_rows],weight[first:first+max_rows],limit)
                    for first in range(0,len(activation),max_rows)]
                results[i]=pieces[0] if len(pieces)==1 else torch.cat(pieces)
            return results
        future = pool.submit(cpu_group) if jobs else None
        if jobs and reference.device.type=='cuda':
            stats['cpu_gpu_overlap_dispatches'] += 1
        outputs = {}
        try:
            for i in selected:
                if i in gpu:
                    rows,top = torch.where(indices == i)
                    outputs[i] = module.experts[i](flat[rows],weights[rows,top,None])
            shared = module.shared_experts(flat)
            if future is not None:
                outputs.update({i:value.to(flat.device) for i,value in future.result().items()})
            y = torch.zeros_like(flat,dtype=torch.float32)
            for i in selected:
                y[positions[i]] += outputs[i]
            y += shared
            return y.to(flat.dtype).view(shape)
        finally:
            if future is not None:
                # A GPU failure must not leave the CPU reading source/model
                # storage after the context has restored or released it.
                future.result()
    try:
        for name,module in reference.model.named_modules():
            if isinstance(module,reference.graph.MoE) and name.startswith('layers.'):
                if hasattr(module,'_ds41_original_forward'):
                    raise ValueError('Nested CPU expert dispatch is not admitted')
                originals.append((module,module.forward))
                module._ds41_original_forward = module.forward
                module.forward = types.MethodType(forward,module)
        yield lambda:{**stats,**executor.statistics()}
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        for module,original in originals:
            module.forward = original
            del module._ds41_original_forward
        if hasattr(executor,'close'):executor.close()
