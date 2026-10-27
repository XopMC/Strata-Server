"""Experimental exact grouped decode for only the actually selected experts.

Reuses the isolated register-fragment grouped GEMM. Every retained compact
weight/scale pair and its source tiles share the existing lease/LRU budget.
No full384-expert layer is loaded. Never used for source calibration.
"""
from contextlib import contextmanager,ExitStack
import hashlib
import json
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from .grouped_experts import routing_layout,source_order_sum,grouped_gemm

ROOT=Path(__file__).resolve().parents[3]


def identity():
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ('selected_experts.py','grouped_experts.py')}


def compact_layout(indices,expert_count):
    if (indices.ndim!=2 or indices.dtype!=torch.int64 or not 1<=len(indices)<=8 or
            indices.shape[1]!=6 or type(expert_count)is not int or not 6<=expert_count<=384):
        raise ValueError('Selected batching admits1–8tokens with six distinct experts per token')
    if bool(((indices<0)|(indices>=expert_count)).any()):raise ValueError('Selected expert index is outside source inventory')
    experts=indices.flatten().unique(sorted=True)
    local=torch.searchsorted(experts,indices)
    layout=routing_layout(local,len(experts))
    layout['source_experts']=experts.cpu().tolist()
    return layout


@contextmanager
def selected_projection(ops,layer,projection,experts):
    if (projection not in ('w1','w2','w3')or not experts or len(experts)>48 or
            experts!=sorted(set(experts))or any(type(i)is not int or not 0<=i<384 for i in experts)):
        raise ValueError('Expected a bounded ordered selected projection')
    infos=[ops.store.info(f'layers.{layer}.ffn.experts.{i}.{projection}.weight')for i in experts]
    scales=[ops.store.info(f'layers.{layer}.ffn.experts.{i}.{projection}.scale')for i in experts]
    if (any(info['dtype']!='I8'or info['shape']!=infos[0]['shape']for info in infos)or
            any(info['dtype']!='F8_E8M0'or info['shape']!=scales[0]['shape']for info in scales)):
        raise ValueError('Selected groups require unchanged native MXFP4/E8M0 matrices')
    expected=sum(a['bytes']+b['bytes']for a,b in zip(infos,scales))
    if expected>256*1024**2 or expected>ops.weight_cache.budget_bytes:
        raise MemoryError('Selected compact projection exceeds its bounded shared cache budget')
    key=('selected-native-experts-v1',layer,projection,tuple(experts),tuple(infos[0]['shape']),tuple(scales[0]['shape']))
    if ops.weight_cache.contains(key,expected):
        def absent():raise ValueError('Single-owner compact cache entry unexpectedly disappeared')
        with ops.weight_cache.lease(key,expected,absent)as pair:
            try:yield pair
            finally:
                if ops.device.type=='cuda':
                    for value in pair:value.record_stream(torch.cuda.current_stream(ops.device))
        return
    # Source leases are obtained outside the loader: recursive cache loaders
    # are forbidden. Duplication during stack creation is also fully charged.
    with ExitStack()as stack:
        pairs=[stack.enter_context(ops.weight_tile(f'layers.{layer}.ffn.experts.{i}.{projection}.weight',
            0,info['shape'][0],True))for i,info in zip(experts,infos)]
        def load():
            return tuple(torch.stack([pair[column].view(torch.uint8)for pair in pairs]).view(pairs[0][column].dtype)
                for column in (0,1))
        with ops.weight_cache.lease(key,expected,load)as pair:
            try:yield pair
            finally:
                if ops.device.type=='cuda':
                    for value in pair:value.record_stream(torch.cuda.current_stream(ops.device))


def admit_component(reference):
    proof=json.loads((ROOT.parents[1]/'outputs/deepseek41/resident-lab/ram-isolated-selected-experts-v2.json').read_text())
    child=proof.get('child',{})
    if (proof.get('memory_errors')!=0 or child.get('implementation_sha256')!=identity()or
            child.get('source_revision')!=reference.store.ram_preload_report['source_revision']or
            len(child.get('checks',[]))!=27 or child.get('all_bitexact_native')is not True):
        raise ValueError('Selected batching requires27 exact actual-source isolated memcheck projections')


@contextmanager
def selected_decode(reference,*,component_audit=False):
    if (reference.device.type!='cuda'or reference.device.index not in (None,0)or
            reference.ops.observer is not None):raise ValueError('Selected decode requires the existing GPU0 source owner without importance collection')
    admit_component(reference)
    if not component_audit:
        proof=json.loads((ROOT.parents[1]/'outputs/deepseek41/resident-lab/ram-selected-experts-model-forced8-v2.json').read_text())
        if (proof.get('all_probe_tensors_bitexact')is not True or
                proof.get('selected_experts_implementation_sha256')!=identity()or
                proof.get('comparison',{}).get('all_greedy_decisions_match')is not True):
            raise ValueError('Selected decode requires bitexact complete actual model probes')
    originals=[];stats={'selected_decode_calls':0,'projection_launches':0,'selected_expert_groups':0,
        'routed_token_rows':0,'fallback_calls':0,'source_importance_collection':False,'cache_accounting':'all source and compact packed pairs share one existing LRU budget'}
    def forward(module,x,image_mask,original):
        shape=x.shape;flat=x.reshape(-1,module.dim)
        if not 1<=len(flat)<=8:
            stats['fallback_calls']+=1;return original(x,image_mask)
        weights,indices=module.gate(flat,None if image_mask is None else image_mask.flatten())
        layout=compact_layout(indices,module.n_routed_experts);experts=layout['source_experts']
        if any(reference.store.info(f'layers.{module.layer_id}.ffn.experts.{i}.{p}.weight')['dtype']!='I8'
                for i in experts for p in ('w1','w2','w3')):
            stats['fallback_calls']+=1;return original(x,image_mask)
        padded=flat.new_zeros(layout['rows'],flat.shape[1]);padded[layout['destinations']]=flat[layout['tokens']]
        quant,sa=reference.ops.quantize_activation(padded);del padded
        with ExitStack()as stack:
            pairs={p:stack.enter_context(selected_projection(reference.ops,module.layer_id,p,experts))for p in ('w1','w3','w2')}
            gate=grouped_gemm(quant,sa,*pairs['w1'],layout['offsets'],layout['max_tiles']).float()
            up=grouped_gemm(quant,sa,*pairs['w3'],layout['offsets'],layout['max_tiles']).float();del quant,sa
            limit=module.experts[0].swiglu_limit
            if limit>0:gate.clamp_max_(limit);up.clamp_(-limit,limit)
            routing=torch.zeros(layout['rows'],1,device=flat.device,dtype=torch.float32)
            routing[layout['destinations'],0]=weights.flatten()[layout['permutation']].float()
            hidden=(F.silu(gate)*up*routing).to(torch.bfloat16);del gate,up,routing
            activation,scales=reference.ops.quantize_activation(hidden.contiguous());del hidden
            output=grouped_gemm(activation,scales,*pairs['w2'],layout['offsets'],layout['max_tiles'])
            result=source_order_sum(output,layout)
        result+=module.shared_experts(flat)
        stats['selected_decode_calls']+=1;stats['projection_launches']+=3
        stats['selected_expert_groups']+=len(experts);stats['routed_token_rows']+=indices.numel()
        return result.to(flat.dtype).reshape(shape)
    try:
        for name,module in reference.model.named_modules():
            if isinstance(module,reference.graph.MoE)and name.startswith('layers.'):
                original=module.forward;originals.append((module,original))
                def patched(this,x,image_mask=None,original=original):return forward(this,x,image_mask,original)
                module.forward=types.MethodType(patched,module)
        yield stats
    finally:
        for module,original in originals:module.forward=original
