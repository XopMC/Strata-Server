from .cuda_admission import cuda0_mask_admitted
"""Bounded native CUDA linear operations, using the pinned FP8/MXFP4 kernels.

Packed tiles and scales go to the masked GPU directly. No whole weight table is
expanded on CPU. Optional bounded packed residency reuses identical weight and
scale pairs. Production CPU/GPU dispatch and checkpoint quality remain pending.
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import numpy as np
import torch

from .streamed_reference import NativeOperations
from .residency import PackedTileCache

_KERNELS = {}


def decode_fp8_block_tile(codes,scales,dtype=torch.bfloat16):
    """Decode one row-aligned source FP8/E8M0 tile using bounded Torch ops."""
    if (codes.ndim != 2 or scales.ndim != 2 or codes.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float8_e8m0fnu or codes.device != scales.device
            or codes.shape[1]%32 or scales.shape != ((codes.shape[0]+31)//32,codes.shape[1]//32)):
        raise ValueError('Invalid grouped FP8 row tile and scales')
    decoded_scales = scales.float()
    if not torch.isfinite(decoded_scales).all():
        raise ValueError('Checkpoint contains a non-finite E8M0 scale')
    rows = torch.arange(codes.shape[0],device=codes.device)//32
    return (codes.float().unflatten(-1,(-1,32))*decoded_scales[rows,:,None]).flatten(-2).to(dtype)


def load_cuda_kernels(reference_directory):
    path = Path(reference_directory).resolve()/'kernel.py'
    if path not in _KERNELS:
        spec = importlib.util.spec_from_file_location('deepseek41_pinned_cuda_kernel',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _KERNELS[path] = module
    return _KERNELS[path]


class PackedCudaOperations(NativeOperations):
    def __init__(self,store,reference_directory,device='cuda:0',row_chunk=4096,observer=None,weight_cache_bytes=0,
                 row_decode_mode='cpu',activation_quantizer='sm89-fragment'):
        super().__init__(store,device,row_chunk,observer)
        if self.device.type != 'cuda' or self.device.index not in (None,0):
            raise ValueError('Native packed CUDA operations require the masked CUDA0')
        project_root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('deepseek41_packed_admission',project_root/'tools/run_guarded.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        guard.admit('gpu0')
        if not cuda0_mask_admitted():
            raise ValueError('Native packed CUDA operations require the exact physical GPU0 UUID mask')
        if type(weight_cache_bytes) is not int or weight_cache_bytes < 0:
            raise ValueError('Packed weight cache budget must be nonnegative integer bytes')
        if row_decode_mode not in ('cpu','cuda'):
            raise ValueError('Unknown native row decode mode')
        if activation_quantizer not in ('legacy','sm89-fragment'):
            raise ValueError('Unknown native activation quantizer')
        self.row_decode_mode = row_decode_mode
        self.activation_quantizer = activation_quantizer
        self.device = torch.device('cuda:0')
        if weight_cache_bytes:
            free,_ = torch.cuda.mem_get_info(self.device)
            if weight_cache_bytes > free-2*1024**3:
                raise MemoryError('Packed cache must leave at least2GiB free for model state and scratch')
        self.weight_cache = PackedTileCache(weight_cache_bytes,self.device)
        self.kernels = load_cuda_kernels(reference_directory)
        self.packed_linear_calls = 0

    def rows(self,name,indices,dtype=torch.bfloat16):
        if getattr(self,'row_decode_mode','cpu') != 'cuda':
            return super().rows(name,indices,dtype)
        info = self.store.info(name)
        indices = np.asarray(indices,dtype=np.int64).reshape(-1)
        chunk = self._row_chunk(name)
        outputs = []
        if info['dtype'] in ('BF16','F32'):
            raw_dtype = np.dtype('<u2') if info['dtype'] == 'BF16' else np.dtype('<f4')
            for first in range(0,len(indices),chunk):
                part = indices[first:first+chunk]
                values = np.frombuffer(self.store.rows(name,part),dtype=raw_dtype).copy().reshape(len(part),info['shape'][1])
                value = torch.from_numpy(values).to(self.device)
                if info['dtype'] == 'BF16':
                    value = value.view(torch.bfloat16)
                outputs.append(value.to(dtype))
                self.rows_read += len(part)
        elif (name.endswith('.attn.wo_a.weight') and info['dtype'] == 'F8_E4M3' and len(indices)
              and indices[0]%32 == 0 and np.all(np.diff(indices) == 1) and chunk >= 32):
            chunk = chunk//32*32
            for first in range(0,len(indices),chunk):
                start,end = int(indices[first]),int(indices[min(first+chunk,len(indices))-1])+1
                with self.weight_tile(name,start,end,False) as pair:
                    outputs.append(decode_fp8_block_tile(*pair,dtype))
                del pair
        else:
            return super().rows(name,indices,dtype)
        return (torch.cat(outputs) if outputs else
                torch.empty(0,info['shape'][1],device=self.device,dtype=dtype))

    @contextmanager
    def weight_tile(self,name,start,end,fp4):
        """Lease one immutable source tile and its exact native scale rows.

        A cache belongs to this operation's single verified store. Keys include
        row ranges and types; KV, activations and Engram are never admitted.
        Record the current CUDA stream even on kernel failure so eviction does
        not recycle storage while an asynchronous user still holds it.
        """
        if fp4:
            from .complete_residency import projection_view
            with projection_view(self,name,start,end) as pair:
                if pair is not None:
                    yield pair
                    return
        info = self.store.info(name)
        scale_name = name.rsplit('.',1)[0]+'.scale'
        scale_info = self.store.info(scale_name)
        scale_start,scale_end = (start,end) if fp4 else (start//32,(end+31)//32)
        expected_bytes = ((end-start)*(info['bytes']//info['shape'][0])
                          +(scale_end-scale_start)*(scale_info['bytes']//scale_info['shape'][0]))
        key = (name,start,end,info['dtype'],tuple(info['shape']),
               scale_name,scale_start,scale_end,scale_info['dtype'],tuple(scale_info['shape']))
        def load():
            return (self.packed_rows(name,np.arange(start,end),
                         torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn),
                    self.packed_rows(scale_name,np.arange(scale_start,scale_end),torch.float8_e8m0fnu))
        with self.weight_cache.lease(key,expected_bytes,load) as pair:
            try:
                yield pair
            finally:
                if self.device.type == 'cuda':
                    stream = torch.cuda.current_stream(self.device)
                    for tensor in pair:
                        tensor.record_stream(stream)

    def packed_rows(self,name,indices,dtype):
        info = self.store.info(name)
        raw = self.store.rows(name,indices)
        values = np.frombuffer(raw,dtype=np.uint8).copy().reshape(len(indices),info['shape'][1])
        self.rows_read += len(indices)
        return torch.from_numpy(values).to(self.device).view(dtype)

    def quantize_activation(self,x):
        """Keep complete32-row backing for native activation/scale tiles.

        Actual-checkpoint diagnostics reproduced partial-row query corruption
        with otherwise exact quantized codes/scales. Row-aligned backing fixed
        that case. Guarded scale stores did not overrun, so no such source
        defect is inferred; GEMM also receives complete logical rows below.
        """
        if getattr(self,'activation_quantizer','legacy') == 'sm89-fragment':
            from .sm89_quant import quantize_activation
            return quantize_activation(x)
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[-1]%32:
            raise ValueError('Packed activation quantization requires contiguous BF16 groups')
        shape = x.shape
        rows,width = x.numel()//shape[-1],shape[-1]
        flat = x.reshape(rows,width)
        padding = -rows%32
        if padding:
            flat = torch.nn.functional.pad(flat,(0,0,0,padding))
        codes,scales = self.kernels.act_quant(flat,32,'ue8m0',torch.float8_e8m0fnu)
        return (codes[:rows].reshape(shape).contiguous(),
                scales[:rows].reshape(*shape[:-1],width//32).contiguous())

    def linear(self,name,x,weight_dtype,bias_name=None):
        info = self.store.info(name)
        if info['dtype'] not in ('I8','F8_E4M3'):
            return super().linear(name,x,weight_dtype,bias_name)
        if x.device != self.device or not x.is_contiguous():
            raise ValueError('Native packed linear inputs must be contiguous on the admitted CUDA device')
        if self.observer is not None:
            self.observer(name,x.detach())
        fp4 = info['dtype'] == 'I8'
        if fp4 and '.experts.' not in name:
            raise ValueError('Only native expert MXFP4 matrices are admitted')
        scale_name = name.rsplit('.',1)[0]+'.scale'
        scale_info = self.store.info(scale_name)
        if scale_info['dtype'] != 'F8_E8M0':
            raise ValueError('Native linear scales must be E8M0')
        width = info['shape'][1]*(2 if fp4 else 1)
        if x.shape[-1] != width or width%32:
            raise ValueError('Native packed linear input/scale groups disagree')
        row_chunk = min(self._row_chunk(name),self._row_chunk(scale_name))
        if not fp4:
            row_chunk = (row_chunk//32)*32
        if row_chunk < 1:
            raise ValueError('Native packed tile does not fit a complete scale group')
        output = torch.empty(*x.shape[:-1],info['shape'][0],device=x.device,dtype=torch.bfloat16)
        input_rows = x.numel()//width
        activation_input = x.reshape(input_rows,width)
        if input_rows%32:
            activation_input = torch.nn.functional.pad(activation_input,(0,0,0,-input_rows%32))
        output_rows = output.reshape(input_rows,info['shape'][0])
        # The pinned GEMM wrapper derives its output tensor dtype from this
        # process setting; its compiled kernel writes BF16. Restore on all exits.
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            activation,scales = self.quantize_activation(activation_input)
            for start in range(0,info['shape'][0],row_chunk):
                end = min(start+row_chunk,info['shape'][0])
                with self.weight_tile(name,start,end,fp4) as pair:
                    if fp4:
                        result = self.kernels.fp4_gemm(activation,scales,*pair,
                            torch.float8_e8m0fnu,act_block_size=32)
                    else:
                        result = self.kernels.fp8_gemm(activation,scales,*pair,
                            torch.float8_e8m0fnu,block_size=32)
                    output_rows[:,start:end] = result[:input_rows]
                del pair,result
        finally:
            torch.set_default_dtype(old)
        if bias_name is not None:
            output += self.small(bias_name,output.dtype)
        self.linear_calls += 1
        self.packed_linear_calls += 1
        return output
