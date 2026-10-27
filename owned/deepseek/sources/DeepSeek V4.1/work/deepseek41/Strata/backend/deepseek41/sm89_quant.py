from .cuda_admission import cuda0_mask_admitted
"""FP8/E8M0 activation quantization without aliased shared input/output tiles.

The rounding and floor follow the pinned native representation. Inputs are
loaded directly into fragments; no shared lifetime is reused for output. This
avoids the observed nondeterministic codes in the source TileLang adapter.
"""
import os
from pathlib import Path
import importlib.util
import torch
import tilelang
import tilelang.language as T

CONFIG = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True,
          tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:True}
_ADMITTED = False


@tilelang.jit(pass_configs=CONFIG)
def quant_kernel(width):
    rows = T.symbolic('rows')
    @T.prim_func
    def kernel(x:T.Tensor[(rows,width),'bfloat16'],q:T.Tensor[(rows,width),'float8_e4m3'],
               scales:T.Tensor[(rows,width//32),'float8_e8m0fnu']):
        with T.Kernel(T.ceildiv(rows,32),width//32,threads=128) as (block_row,group):
            values = T.alloc_fragment((32,32),'float32')
            maximum = T.alloc_fragment(32,'float32')
            scale = T.alloc_fragment(32,'float32')
            quantized = T.alloc_fragment((32,32),'float8_e4m3')
            for i,j in T.Parallel(32,32):
                values[i,j] = T.if_then_else(block_row*32+i < rows,x[block_row*32+i,group*32+j],0)
            T.reduce_absmax(values,maximum,dim=1)
            for i in T.Parallel(32):
                value = T.max(maximum[i],1e-4)*(1./448.)
                bits = T.reinterpret('uint32',value)
                exponent = T.Cast('int32',(bits >> 23)&255)
                exponent += T.if_then_else((bits&8388607) != 0,1,0)
                scale[i] = T.reinterpret('float32',exponent << 23)
            for i,j in T.Parallel(32,32):
                quantized[i,j] = T.clamp(values[i,j]/scale[i],-448.,448.)
            for i in T.Parallel(32):
                if block_row*32+i < rows:
                    scales[block_row*32+i,group] = scale[i]
            T.copy(quantized,q[block_row*32,group*32])
    return kernel


def quantize_activation(x):
    global _ADMITTED
    if (x.device.type != 'cuda' or x.device.index != 0 or x.dtype != torch.bfloat16
            or not x.is_contiguous() or not x.numel() or x.shape[-1]%32):
        raise ValueError('SM89 activation quantizer requires contiguous CUDA0 BF16 groups')
    if not _ADMITTED:
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('deepseek41_quant_admission',root/'tools/run_guarded.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        guard.admit('gpu0')
        if not cuda0_mask_admitted():
            raise ValueError('Exact physical GPU0 UUID mask required')
        _ADMITTED = True
    shape,width = x.shape,x.shape[-1]
    rows = x.numel()//width
    flat = x.reshape(rows,width)
    # Keep the same complete logical backing used by the packed GEMM path.
    if rows%32:
        flat = torch.nn.functional.pad(flat,(0,0,0,-rows%32))
    codes = torch.empty_like(flat,dtype=torch.float8_e4m3fn)
    scales = torch.empty(len(flat),width//32,device=x.device,dtype=torch.float8_e8m0fnu)
    quant_kernel(width)(flat,codes,scales)
    return codes[:rows].reshape(shape).contiguous(),scales[:rows].reshape(*shape[:-1],width//32).contiguous()
