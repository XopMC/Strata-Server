"""SM89 sparse attention dequantizes source E2M1/E4M3 only inside64-row KV tiles."""
import os
import torch
import tilelang
import tilelang.language as T
from .tensor_layout import canonical_tensor
from .native_fp4_cache import NativeFP4Prefix

CONFIG={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:True}
_ADMITTED=False


@tilelang.jit(pass_configs=CONFIG)
def attention_kernel(heads,dim,scale):
    batch,queries,windows,keys,count=T.symbolic('batch'),T.symbolic('queries'),T.symbolic('windows'),T.symbolic('keys'),T.symbolic('count')
    block=64
    @T.prim_func
    def kernel(q:T.Tensor[(batch,queries,heads,dim),'bfloat16'],
               window:T.Tensor[(batch,windows,dim),'bfloat16'],
               codes:T.Tensor[(batch,keys,dim//2),'uint8'],
               scales:T.Tensor[(batch,keys,dim//16),'float8_e4m3'],
               output:T.Tensor[(batch,queries,heads,dim),'bfloat16'],
               sink:T.Tensor[(heads,),'float32'],ids:T.Tensor[(batch,queries,count),'int32']):
        with T.Kernel(queries,batch,threads=128)as(pos,b):
            qs=T.alloc_shared((heads,dim),'bfloat16')
            values=T.alloc_shared((block,dim),'bfloat16')
            probabilities=T.alloc_shared((heads,block),'bfloat16')
            out=T.alloc_shared((heads,dim),'bfloat16')
            indices=T.alloc_fragment(block,'int32')
            scores=T.alloc_fragment((heads,block),'float32')
            numerator=T.alloc_fragment((heads,dim),'float32')
            maximum=T.alloc_fragment(heads,'float32');previous=T.alloc_fragment(heads,'float32')
            rescale=T.alloc_fragment(heads,'float32');denominator=T.alloc_fragment(heads,'float32')
            block_sum=T.alloc_fragment(heads,'float32')
            T.clear(numerator);T.clear(denominator);T.fill(maximum,-1e30)
            T.copy(q[b,pos,:,:],qs)
            for tile in T.Pipelined(T.ceildiv(count,block),num_stages=1):
                for i in T.Parallel(block):
                    indices[i]=T.if_then_else(tile*block+i<count,ids[b,pos,tile*block+i],-1)
                for i,j in T.Parallel(block,dim):
                    if indices[i]<0:
                        values[i,j]=0
                    elif indices[i]<windows:
                        values[i,j]=window[b,indices[i],j]
                    else:
                        code=codes[b,indices[i]-windows,j//2]
                        nibble=T.if_then_else(j%2==0,code&15,code>>4)
                        magnitude=nibble&7
                        value=T.if_then_else(magnitude<4,T.Cast('float32',magnitude)*.5,
                            T.if_then_else(magnitude<6,T.Cast('float32',magnitude)-2,T.if_then_else(magnitude==6,4.,6.)))
                        signed_value=T.if_then_else((nibble&8)!=0,-value,value)
                        values[i,j]=T.Cast('bfloat16',signed_value*T.Cast('float32',scales[b,indices[i]-windows,j//16]))
                for i,j in T.Parallel(heads,block):
                    scores[i,j]=T.if_then_else(indices[j]>=0,0,-T.infinity('float32'))
                T.gemm(qs,values,scores,transpose_B=True,policy=T.GemmWarpPolicy.FullRow)
                for i,j in T.Parallel(heads,block):scores[i,j]*=scale
                T.copy(maximum,previous);T.reduce_max(scores,maximum,dim=1,clear=False)
                for i in T.Parallel(heads):rescale[i]=T.exp(previous[i]-maximum[i])
                for i,j in T.Parallel(heads,block):scores[i,j]=T.exp(scores[i,j]-maximum[i])
                T.reduce_sum(scores,block_sum,dim=1)
                for i in T.Parallel(heads):denominator[i]=denominator[i]*rescale[i]+block_sum[i]
                T.copy(scores,probabilities)
                for i,j in T.Parallel(heads,dim):numerator[i,j]*=rescale[i]
                T.gemm(probabilities,values,numerator,policy=T.GemmWarpPolicy.FullRow)
            for i in T.Parallel(heads):
                previous[i]=maximum[i];maximum[i]=T.max(maximum[i],sink[i])
                rescale[i]=T.exp(previous[i]-maximum[i])
                denominator[i]=denominator[i]*rescale[i]+T.exp(sink[i]-maximum[i])
            for i,j in T.Parallel(heads,dim):numerator[i,j]=numerator[i,j]*rescale[i]/denominator[i]
            T.copy(numerator,out);T.copy(out,output[b,pos,:,:])
    return kernel


def sparse_attn(q,window,packed,sink,indices,scale):
    global _ADMITTED
    if(not isinstance(packed,NativeFP4Prefix)or q.device.type!='cuda'or q.device.index!=0 or q.dtype!=torch.bfloat16 or
       window.dtype!=q.dtype or indices.dtype!=torch.int32 or sink.dtype!=torch.float32 or
       any(value.device!=q.device for value in(window,packed,sink,indices))):
        raise ValueError('Native FP4 sparse attention requires masked GPU0 BF16 queries/window and packed KV')
    batch,queries,heads,dim=q.shape
    if(batch!=1 or dim%16 or dim>512 or packed.shape[0]!=batch or packed.shape[-1]!=dim or
       window.shape[0]!=batch or window.shape[-1]!=dim or indices.shape[:2]!=(batch,queries)or sink.shape!=(heads,)or
       torch.any(indices<-1)or torch.any(indices>=window.shape[1]+packed.length)):
        raise ValueError('Invalid bounded Native FP4 sparse attention dimensions/positions')
    if not _ADMITTED:
        import importlib.util
        from pathlib import Path
        root=Path(__file__).resolve().parents[3]
        spec=importlib.util.spec_from_file_location('native_fp4_attention_guard',root/'tools/run_guarded.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.admit('gpu0')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=guard.UUID:raise ValueError('Physical GPU0 UUID mask required')
        _ADMITTED=True
    # Length0 cannot be passed to TVM FFI. Its actual unused packed buffers are
    # still capacity-sized, so reads remain legal even at a partial first group.
    codes,scales=packed.cache.codes,packed.cache.scales
    output=torch.empty_like(q);window=canonical_tensor(window)
    for start in range(0,heads,16):
        stop=min(start+16,heads)
        query=torch.empty((batch,queries,stop-start,dim),device=q.device,dtype=q.dtype)
        query.copy_(q[:,:,start:stop]);bias=sink[start:stop].contiguous()
        if stop-start<16:
            query=torch.nn.functional.pad(query,(0,0,0,16-(stop-start)))
            bias=torch.nn.functional.pad(bias,(0,16-(stop-start)))
        result=torch.empty_like(query)
        attention_kernel(16,dim,float(scale))(query,window,codes,scales,result,bias,indices)
        output[:,:,start:stop]=result[:,:,:stop-start]
    return output
