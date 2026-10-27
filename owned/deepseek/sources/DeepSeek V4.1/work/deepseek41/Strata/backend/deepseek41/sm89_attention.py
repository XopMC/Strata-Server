from .cuda_admission import cuda0_mask_admitted
"""SM89 sparse-attention adaptation of the pinned reference's64-position kernel.

One KV stage and16-head groups reduce shared storage for actual D=512 latents.
The64-position BF16 probability boundary is preserved. This is unvalidated until
the explicit masked-GPU component and checkpoint tests finish.
"""
import torch
import importlib.util
import os
from pathlib import Path
import tilelang
import tilelang.language as T

CONFIG = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True,
          tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:True}
_ADMITTED = False


@tilelang.jit(pass_configs=CONFIG)
def attention_kernel(heads,dim,scale):
    batch,queries,keys,count = T.symbolic('batch'),T.symbolic('queries'),T.symbolic('keys'),T.symbolic('count')
    block = 64
    @T.prim_func
    def kernel(q:T.Tensor[(batch,queries,heads,dim),'bfloat16'],
               kv:T.Tensor[(batch,keys,dim),'bfloat16'],
               output:T.Tensor[(batch,queries,heads,dim),'bfloat16'],
               sink:T.Tensor[(heads,),'float32'],
               ids:T.Tensor[(batch,queries,count),'int32']):
        with T.Kernel(queries,batch,threads=128) as (pos,b):
            qs = T.alloc_shared((heads,dim),'bfloat16')
            values = T.alloc_shared((block,dim),'bfloat16')
            probabilities = T.alloc_shared((heads,block),'bfloat16')
            out = T.alloc_shared((heads,dim),'bfloat16')
            indices = T.alloc_fragment(block,'int32')
            scores = T.alloc_fragment((heads,block),'float32')
            numerator = T.alloc_fragment((heads,dim),'float32')
            maximum = T.alloc_fragment(heads,'float32')
            previous = T.alloc_fragment(heads,'float32')
            rescale = T.alloc_fragment(heads,'float32')
            denominator = T.alloc_fragment(heads,'float32')
            block_sum = T.alloc_fragment(heads,'float32')
            T.clear(numerator)
            T.clear(denominator)
            T.fill(maximum,-1e30)
            T.copy(q[b,pos,:,:],qs)
            for tile in T.Pipelined(T.ceildiv(count,block),num_stages=1):
                for i in T.Parallel(block):
                    indices[i] = T.if_then_else(tile*block+i < count,ids[b,pos,tile*block+i],-1)
                for i,j in T.Parallel(block,dim):
                    values[i,j] = T.if_then_else(indices[i] >= 0,kv[b,indices[i],j],0)
                for i,j in T.Parallel(heads,block):
                    scores[i,j] = T.if_then_else(indices[j] >= 0,0,-T.infinity('float32'))
                T.gemm(qs,values,scores,transpose_B=True,policy=T.GemmWarpPolicy.FullRow)
                for i,j in T.Parallel(heads,block):
                    scores[i,j] *= scale
                T.copy(maximum,previous)
                T.reduce_max(scores,maximum,dim=1,clear=False)
                for i in T.Parallel(heads):
                    rescale[i] = T.exp(previous[i]-maximum[i])
                for i,j in T.Parallel(heads,block):
                    scores[i,j] = T.exp(scores[i,j]-maximum[i])
                T.reduce_sum(scores,block_sum,dim=1)
                for i in T.Parallel(heads):
                    denominator[i] = denominator[i]*rescale[i]+block_sum[i]
                T.copy(scores,probabilities)
                for i,j in T.Parallel(heads,dim):
                    numerator[i,j] *= rescale[i]
                T.gemm(probabilities,values,numerator,policy=T.GemmWarpPolicy.FullRow)
            # Stable addition of the virtual zero-valued sink KV.
            for i in T.Parallel(heads):
                previous[i] = maximum[i]
                maximum[i] = T.max(maximum[i],sink[i])
                rescale[i] = T.exp(previous[i]-maximum[i])
                denominator[i] = denominator[i]*rescale[i]+T.exp(sink[i]-maximum[i])
            for i,j in T.Parallel(heads,dim):
                numerator[i,j] = numerator[i,j]*rescale[i]/denominator[i]
            T.copy(numerator,out)
            T.copy(out,output[b,pos,:,:])
    return kernel


def sparse_attn(q,kv,sink,indices,scale):
    global _ADMITTED
    if (q.device.type != 'cuda' or q.device.index != 0 or q.dtype != torch.bfloat16 or
            kv.dtype != q.dtype or indices.dtype != torch.int32 or
            any(tensor.device != q.device for tensor in (kv,sink,indices))):
        raise ValueError('SM89 attention requires CUDA BF16 latents and INT32 sparse IDs')
    if not _ADMITTED:
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('deepseek41_attention_admission',root/'tools/run_guarded.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        guard.admit('gpu0')
        if not cuda0_mask_admitted():
            raise ValueError('Exact physical GPU0 UUID mask required for SM89 attention')
        _ADMITTED = True
    batch,queries,heads,dim = q.shape
    if (dim%16 or dim > 512 or kv.shape[0] != batch or kv.shape[-1] != dim or
            indices.shape[:2] != (batch,queries) or sink.shape != (heads,) or
            torch.any(indices < -1) or torch.any(indices >= kv.shape[1])):
        raise ValueError('Invalid bounded sparse-attention dimensions or indices')
    output = torch.empty_like(q)
    for start in range(0,heads,16):
        stop = min(start+16,heads)
        # contiguous() can preserve a noncanonical singleton-sequence stride.
        # TVM FFI validates that stride even when PyTorch ignores it for S=1.
        query = torch.empty((batch,queries,stop-start,dim),device=q.device,dtype=q.dtype)
        query.copy_(q[:,:,start:stop])
        bias = sink[start:stop].contiguous()
        if stop-start < 16:
            query = torch.nn.functional.pad(query,(0,0,0,16-(stop-start)))
            bias = torch.nn.functional.pad(bias,(0,16-(stop-start)))
        result = torch.empty_like(query)
        attention_kernel(16,dim,float(scale))(query,kv,result,bias,indices)
        output[:,:,start:stop] = result[:,:,:stop-start]
    return output
