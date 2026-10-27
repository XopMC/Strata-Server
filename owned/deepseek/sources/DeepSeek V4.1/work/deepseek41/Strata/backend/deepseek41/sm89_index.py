from .cuda_admission import cuda0_mask_admitted
"""SM89 index-score prototype with bounded output and no per-head score tensor.

The native BF16 dot/relu/product/sum boundaries are explicit. Component and
checkpoint numerical validation are mandatory before graph integration.
"""
import importlib.util
import os
from pathlib import Path
import torch
import tilelang
import tilelang.language as T
from .tensor_layout import canonical_tensor

CONFIG = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True,
          tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:True}
_ADMITTED = False


@tilelang.jit(pass_configs=CONFIG)
def index_score_kernel():
    batch,queries,keys = T.symbolic('batch'),T.symbolic('queries'),T.symbolic('keys')
    heads,dim,query_tile,key_tile = 32,128,2,128
    rows = query_tile*heads
    @T.prim_func
    def kernel(q:T.Tensor[(batch,queries,heads,dim),'bfloat16'],
               k:T.Tensor[(batch,keys,dim),'bfloat16'],
               w:T.Tensor[(batch,queries,heads),'bfloat16'],
               score:T.Tensor[(batch,queries,keys),'bfloat16']):
        with T.Kernel(T.ceildiv(keys,key_tile),T.ceildiv(queries,query_tile),batch,threads=256) as (kt,qt,b):
            qs = T.alloc_shared((rows,dim),'bfloat16')
            ks = T.alloc_shared((key_tile,dim),'bfloat16')
            dot = T.alloc_fragment((rows,key_tile),'float32')
            product = T.alloc_fragment((query_tile,heads,key_tile),'float32')
            reduced = T.alloc_fragment((query_tile,key_tile),'float32')
            output = T.alloc_fragment((query_tile,key_tile),'bfloat16')
            for row,d in T.Parallel(rows,dim):
                qs[row,d] = T.if_then_else(qt*query_tile+row//heads < queries,
                                          q[b,qt*query_tile+row//heads,row%heads,d],0)
            T.copy(k[b,kt*key_tile,0],ks)
            T.clear(dot)
            T.gemm(qs,ks,dot,transpose_B=True)
            for row,key in T.Parallel(rows,key_tile):
                # Source einsum rounds toBF16, multiplication also rounds to
                # BF16, then the32-head reduction accumulates inFP32.
                value = T.max(T.Cast('float32',T.Cast('bfloat16',dot[row,key])),0)
                weight = T.if_then_else(qt*query_tile+row//heads < queries,
                                       w[b,qt*query_tile+row//heads,row%heads],0)
                product[row//heads,row%heads,key] = T.Cast('float32',T.Cast('bfloat16',value*T.Cast('float32',weight)))
            T.reduce_sum(product,reduced,dim=1)
            for query,key in T.Parallel(query_tile,key_tile):
                output[query,key] = T.Cast('bfloat16',reduced[query,key])
            T.copy(output,score[b,qt*query_tile,kt*key_tile])
    return kernel


def score_index(q,k,weights,*,score_budget_bytes=64*1024**2):
    global _ADMITTED
    if (q.ndim != 4 or q.shape[-2:] != (32,128) or k.ndim != 3 or weights.ndim != 3
            or q.shape[0] != k.shape[0] or k.shape[-1] != 128 or weights.shape != q.shape[:-1]
            or q.device.type != 'cuda' or q.device.index != 0 or
            any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q,k,weights))):
        raise ValueError('Fused index requires masked CUDA0 BF1632-head/128-dimension inputs')
    batch,queries = q.shape[:2]
    keys = k.shape[1]
    if min(batch,queries,keys) < 1 or batch*queries*keys*2 > score_budget_bytes:
        raise ValueError('Fused index output exceeds bounded score budget')
    if not _ADMITTED:
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('deepseek41_index_admission',root/'tools/run_guarded.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        guard.admit('gpu0')
        if not cuda0_mask_admitted():
            raise ValueError('Physical GPU0 UUID mask required for fused index')
        _ADMITTED = True
    # Cache prefixes need canonical unused batch strides, not a second full
    # copy of the1M-position key cache. Noncontiguous real dimensions copy.
    query,key,weight = (canonical_tensor(t) for t in (q,k,weights))
    output = torch.empty((batch,queries,keys),device=q.device,dtype=q.dtype)
    index_score_kernel()(query,key,weight,output)
    return output


def score_index_with_small_fallback(q,k,weights,*,score_budget_bytes=64*1024**2):
    """Retain the source short-key behavior while staging fused long scoring.

    Short tail component tests had measurable BF16 differences. Use the native
    full-head Torch expression for <=256 keys. Long scoring remains optional
    until actual-checkpoint selection and task/NLL comparisons are accepted.
    """
    if k.shape[1] <= 256:
        if q.shape[0]*q.shape[1]*k.shape[1]*q.element_size() > score_budget_bytes:
            raise ValueError('Short index fallback output exceeds score budget')
        from .long_context import _score_tile
        return _score_tile(q,k,weights,q.shape[2])
    return score_index(q,k,weights,score_budget_bytes=score_budget_bytes)
