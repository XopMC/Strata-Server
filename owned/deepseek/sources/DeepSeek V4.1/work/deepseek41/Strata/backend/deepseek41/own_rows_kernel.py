from .cuda_admission import cuda0_mask_admitted
"""Exact F32/BF16 decoding of our uniform group32 rows on the masked GPU0."""
import os
from pathlib import Path
import importlib.util
import torch
import tilelang
import tilelang.language as T

CONFIG={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:True}
_ADMITTED=False


@tilelang.jit(pass_configs=CONFIG)
def decode_kernel(bits,width,dtype):
    rows=T.symbolic('rows');code_bytes=width*bits//8;row_bytes=code_bytes+width//32*2
    qmax=(1<<(bits-1))-1;mask=(1<<bits)-1
    @T.prim_func
    def kernel(raw:T.Tensor[(rows,row_bytes),'uint8'],output:T.Tensor[(rows,width),dtype]):
        with T.Kernel(T.ceildiv(rows*width,256),threads=128)as block:
            for lane in T.Parallel(256):
                position=block*256+lane
                if position<rows*width:
                    row,column=position//width,position%width
                    bit=column*bits;byte=bit//8
                    # The pair fits16bits. Signed arithmetic also avoids TVM's
                    # unsigned conditional-bound solver for code==mask.
                    pair=T.Cast('int32',raw[row,byte])|(T.Cast('int32',raw[row,byte+1])<<8)
                    code=(pair>>(bit%8))&mask
                    scale_byte=code_bytes+(column//32)*2
                    word=T.Cast('int32',raw[row,scale_byte])|(T.Cast('int32',raw[row,scale_byte+1])<<8)
                    scale=T.Cast('float32',T.reinterpret('bfloat16',T.Cast('uint16',word)))
                    # Subtract in F32: all codes/qmax are exactly representable.
                    # TVM0.1.8 can erase the signed cast while simplifying
                    # uint32 code-qmax, then rejects its negative constant.
                    value=(T.Cast('float32',code)-T.float32(qmax))*scale
                    output[row,column]=T.Cast(dtype,T.if_then_else(code==mask,T.infinity('float32'),value))
    return kernel


def decode(raw,bits,width,dtype=torch.bfloat16):
    global _ADMITTED
    from .own_quant_rows import row_layout
    if raw.ndim!=2 or type(bits)is not int or bits not in(2,3,4,6,8)or type(width)is not int or width<32 or width%32:
        raise ValueError('Invalid own CUDA row dimensions/bit precision')
    row_bytes=row_layout({'dtype':'D41Q','codec':'deepseek41-signed-uniform-bf16-group-v1',
        'bits':bits,'group_size':32,'shape':[raw.shape[0],width],
        'bytes':raw.shape[0]*(width*bits//8+width//32*2)})
    if(raw.ndim!=2 or raw.dtype!=torch.uint8 or raw.shape[-1]!=row_bytes or not raw.is_contiguous()or
       raw.device.type!='cuda'or raw.device.index!=0 or dtype not in(torch.bfloat16,torch.float32)or
       raw.shape[0]*width*4>64*1024**2):raise ValueError('Own row CUDA decode requires bounded contiguous masked GPU0 payload')
    if not _ADMITTED:
        root=Path(__file__).resolve().parents[3]
        spec=importlib.util.spec_from_file_location('own_rows_guard',root/'tools/run_guarded.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.admit('gpu0')
        if not cuda0_mask_admitted():raise ValueError('Own row decoding requires physical GPU0 only')
        _ADMITTED=True
    result=torch.empty(raw.shape[0],width,dtype=dtype,device=raw.device)
    decode_kernel(bits,width,'bfloat16'if dtype==torch.bfloat16 else'float32')(raw,result)
    return result
