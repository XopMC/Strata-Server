"""Native MXFP4 grouped prefill with complete layer residency in one budget.

One GPU launch handles all expert groups for one projection. FP8 activation,
FP4->FP8, per32 scales and FP32 accumulation follow the pinned GEMM. Source
expert order is retained during routed accumulation. Quality still needs tests.
"""
from contextlib import contextmanager,ExitStack
import importlib
import importlib.util
import os
from pathlib import Path
import types
import numpy as np
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from .complete_residency import projection_key

CONFIG={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:True}
_ADMITTED=False


@tilelang.jit(pass_configs=CONFIG)
def grouped_kernel(experts,N,K,max_tiles):
    rows=T.symbolic('rows')
    @T.prim_func
    def kernel(A:T.Tensor[(rows,K),'float8_e4m3'],
               B:T.Tensor[(experts,N,K),'float4_e2m1fn'],
               C:T.Tensor[(rows,N),'bfloat16'],
               SA:T.Tensor[(rows,K//32),'float8_e8m0fnu'],
               SB:T.Tensor[(experts,N,K//32),'float8_e8m0fnu'],
               offsets:T.Tensor[(experts+1,),'int32']):
        with T.Kernel(T.ceildiv(N,128),max_tiles,experts,threads=128) as (bx,by,expert):
            a=T.alloc_shared((32,32),'float8_e4m3')
            # A fractional-byte shared allocation was placed at byte3064,
            # then written with16-byte vectors (memcheck: misaligned stores).
            # Keep FP4 only in registers; shared storage is byte-addressable
            # FP8/BF16, as in the validated fragment activation quantizer.
            b4=T.alloc_fragment((128,32),'float4_e2m1fn')
            b=T.alloc_shared((128,32),'float8_e4m3')
            out=T.alloc_shared((32,128),'bfloat16')
            dot=T.alloc_fragment((32,128),'float32')
            acc=T.alloc_fragment((32,128),'float32')
            sa=T.alloc_fragment(32,'float32')
            sb=T.alloc_fragment(128,'float32')
            T.use_swizzle(panel_size=10)
            if by*32 < offsets[expert+1]-offsets[expert]:
                T.clear(dot)
                T.clear(acc)
                # The automatic pipeline with dynamic expert offsets produced
                # nondeterministic results. Validate a synchronized K loop
                # before attempting an explicit double-buffered variant.
                for k in T.serial(K//32):
                    T.copy(A[offsets[expert]+by*32,k*32],a)
                    T.copy(B[expert,bx*128,k*32],b4)
                    T.sync_threads()
                    for i,j in T.Parallel(128,32):
                        b[i,j]=T.Cast('float8_e4m3',T.Cast('float32',b4[i,j]))
                    T.sync_threads()
                    for i in T.Parallel(128):
                        sb[i]=T.Cast('float32',SB[expert,bx*128+i,k])
                    for i in T.Parallel(32):
                        sa[i]=T.Cast('float32',SA[offsets[expert]+by*32+i,k])
                    T.gemm(a,b,dot,transpose_B=True)
                    for i,j in T.Parallel(32,128):
                        acc[i,j]+=dot[i,j]*sa[i]*sb[j]
                    T.clear(dot)
                    T.sync_threads()
                T.copy(acc,out)
                T.copy(out,C[offsets[expert]+by*32,bx*128])
    return kernel


def grouped_gemm(activation,scales,codes,weight_scales,offsets,max_tiles):
    global _ADMITTED
    if (activation.device.type!='cuda' or activation.device.index!=0 or activation.ndim!=2
        or activation.dtype!=torch.float8_e4m3fn or codes.ndim!=3 or codes.dtype!=torch.float4_e2m1fn_x2
        or scales.dtype!=torch.float8_e8m0fnu or weight_scales.dtype!=torch.float8_e8m0fnu
        or offsets.dtype!=torch.int32 or offsets.shape!=(len(codes)+1,) or type(max_tiles) is not int or max_tiles<1
        or any(t.device!=activation.device or not t.is_contiguous() for t in (activation,scales,codes,weight_scales,offsets))):
        raise ValueError('Grouped GEMM requires contiguous native CUDA0 expert groups')
    rows,K=activation.shape;E,N,packed_width=codes.shape
    if K%32 or N%128 or rows%32 or packed_width*2!=K or scales.shape!=(rows,K//32) or weight_scales.shape!=(E,N,K//32):
        raise ValueError('Grouped native expert shapes disagree')
    if not _ADMITTED:
        root=Path(__file__).resolve().parents[3]
        spec=importlib.util.spec_from_file_location('deepseek41_grouped_admission',root/'tools/run_guarded.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        guard.admit('gpu0')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=guard.UUID:
            raise ValueError('Physical GPU0 UUID mask required for grouped expert kernels')
        _ADMITTED=True
    output=torch.empty(rows,N,dtype=torch.bfloat16,device=activation.device)
    grouped_kernel(E,N,K,max_tiles)(activation,codes,output,scales,weight_scales,offsets)
    return output


def routing_layout(indices,experts):
    """Stable expert grouping; every group has complete32-row backing."""
    if indices.ndim!=2 or indices.dtype!=torch.int64 or type(experts) is not int or not 1<=experts<=384:
        raise ValueError('Expected bounded native routed indices')
    if not len(indices) or indices.shape[1]>8 or bool(((indices<0)|(indices>=experts)).any()):
        raise ValueError('Invalid routed expert selection')
    # Duplicate expert selection would invalidate source accumulation ordering.
    ordered=indices.sort(-1).values
    if bool((ordered[:,1:]==ordered[:,:-1]).any()):
        raise ValueError('Router selected a duplicate expert for one token')
    counts=torch.bincount(indices.flatten(),minlength=experts).cpu().tolist()
    padded=[(count+31)//32*32 for count in counts]
    prefix=[0]
    unpadded=[0]
    for count,round_count in zip(counts,padded):
        prefix.append(prefix[-1]+round_count);unpadded.append(unpadded[-1]+count)
    permutation=indices.flatten().argsort(stable=True)
    sorted_experts=indices.flatten()[permutation]
    padding=torch.tensor([a-b for a,b in zip(prefix[:-1],unpadded[:-1])],device=indices.device,dtype=torch.int64)
    destinations=torch.arange(indices.numel(),device=indices.device)+padding[sorted_experts]
    inverse=torch.empty_like(permutation)
    inverse[permutation]=destinations
    return {'counts':counts,'offsets':torch.tensor(prefix,device=indices.device,dtype=torch.int32),
            'permutation':permutation,'destinations':destinations,'original_to_padded':inverse.reshape(indices.shape),
            'rows':prefix[-1],'max_tiles':max(padded)//32,'tokens':permutation//indices.shape[1],
            'source_expert_order':indices.argsort(dim=-1,stable=True)}


def source_order_sum(output,layout):
    positions=layout['original_to_padded'].gather(-1,layout['source_expert_order'])
    result=torch.zeros(len(positions),output.shape[-1],dtype=torch.float32,device=output.device)
    for rank in range(positions.shape[1]):
        result+=output[positions[:,rank]]
    return result


def layer_key(layer,projection,count,info,scale):
    return projection_key(layer,projection,count,info,scale)


@contextmanager
def layer_projection(ops,layer,projection,count):
    name=f'layers.{layer}.ffn.experts.0.{projection}.weight'
    info=ops.store.info(name);scale=ops.store.info(name.rsplit('.',1)[0]+'.scale')
    if info['dtype']!='I8' or scale['dtype']!='F8_E8M0':
        raise ValueError('Grouped complete layers currently admit native MXFP4/E8M0 only')
    expected=count*(info['bytes']+scale['bytes'])
    if expected>ops.weight_cache.budget_bytes:
        raise MemoryError('Complete expert projection does not fit the shared GPU budget')
    def load():
        from .staging import upload_projection
        return upload_projection(ops,layer,projection,count,info,scale)
    key=layer_key(layer,projection,count,info,scale)
    with ops.weight_cache.lease(key,expected,load) as pair:
        try:yield pair
        finally:
            for value in pair:value.record_stream(torch.cuda.current_stream(ops.device))


@contextmanager
def grouped_prefill(reference,*,minimum_tokens=512):
    # The private resident reloads this adapter between cases without recopying
    # weights. Refresh its byte-transport leaf as well, never during a forward.
    from . import staging
    importlib.reload(staging)
    if reference.device.type!='cuda' or not hasattr(reference.ops,'quantize_activation') or minimum_tokens<1:
        raise ValueError('Grouped prefill requires the existing native CUDA owner')
    originals=[]
    eligible={}
    stats={'grouped_prefill_calls':0,'grouped_projection_launches':0,'routed_token_rows':0,'fallback_calls':0}
    def forward(module,x,image_mask=None,original=None):
        shape=x.shape;flat=x.reshape(-1,module.dim)
        if len(flat)<minimum_tokens or not eligible.get(module.layer_id,False):
            stats['fallback_calls']+=1
            return original(x,image_mask)
        if reference.ops.observer is not None:
            raise ValueError('Source importance collection must use its unchanged per-expert baseline')
        weights,indices=module.gate(flat,None if image_mask is None else image_mask.flatten())
        layout=routing_layout(indices,module.n_routed_experts)
        padded=flat.new_zeros(layout['rows'],flat.shape[1])
        padded[layout['destinations']]=flat[layout['tokens']]
        quant,activation_scales=reference.ops.quantize_activation(padded)
        del padded
        with ExitStack() as stack:
            pair1=stack.enter_context(layer_projection(reference.ops,module.layer_id,'w1',module.n_routed_experts))
            pair3=stack.enter_context(layer_projection(reference.ops,module.layer_id,'w3',module.n_routed_experts))
            pair2=stack.enter_context(layer_projection(reference.ops,module.layer_id,'w2',module.n_routed_experts))
            gate=grouped_gemm(quant,activation_scales,*pair1,layout['offsets'],layout['max_tiles']).float()
            up=grouped_gemm(quant,activation_scales,*pair3,layout['offsets'],layout['max_tiles']).float()
            del quant,activation_scales
            limit=module.experts[0].swiglu_limit
            if limit>0:
                gate.clamp_max_(limit);up.clamp_(-limit,limit)
            routing_weights=torch.zeros(layout['rows'],1,device=flat.device,dtype=torch.float32)
            routing_weights[layout['destinations'],0]=weights.flatten()[layout['permutation']].float()
            hidden=(F.silu(gate)*up*routing_weights).to(torch.bfloat16)
            del gate,up,routing_weights
            hidden_quant,hidden_scales=reference.ops.quantize_activation(hidden.contiguous())
            del hidden
            output=grouped_gemm(hidden_quant,hidden_scales,*pair2,layout['offsets'],layout['max_tiles'])
            result=source_order_sum(output,layout)
        result+=module.shared_experts(flat)
        stats['grouped_prefill_calls']+=1;stats['grouped_projection_launches']+=3
        stats['routed_token_rows']+=indices.numel()
        return result.to(flat.dtype).reshape(shape)
    try:
        for name,module in reference.model.named_modules():
            if isinstance(module,reference.graph.MoE) and name.startswith('layers.'):
                eligible[module.layer_id]=all(reference.store.info(f'layers.{module.layer_id}.ffn.experts.{i}.{p}.weight')['dtype']=='I8'
                    for i in range(module.n_routed_experts) for p in ('w1','w2','w3'))
                original=module.forward
                originals.append((module,original))
                def patched(this,x,image_mask=None,original=original):return forward(this,x,image_mask,original)
                module.forward=types.MethodType(patched,module)
        yield stats
    finally:
        for module,original in originals:module.forward=original
