"""Lease expert row views from an existing complete-layer GPU allocation.

No loader, allocation, decode or new CUDA kernel. The entire parent storage
remains charged to the one cache budget until all asynchronous users finish.
"""
from contextlib import contextmanager
import re
import torch

EXPERT=re.compile(r'^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$')


def projection_key(layer,projection,count,info,scale):
    return ('native-complete-experts',layer,projection,count,tuple(info['shape']),info['dtype'],tuple(scale['shape']),scale['dtype'])


def existing_projection(ops,name):
    match=EXPERT.fullmatch(name)
    if match is None:return None
    layer,expert,projection=match.groups();layer,expert=int(layer),int(expert)
    info=ops.store.info(name)
    if info['dtype']!='I8':return None
    scale=ops.store.info(name.rsplit('.',1)[0]+'.scale')
    if scale['dtype']!='F8_E8M0':return None
    with ops.weight_cache.lock:
        index=getattr(ops.store,'index',None)
        if index is not None and f'layers.{layer}.ffn.experts.383.{projection}.weight' in index:
            # The verified full checkpoint has384experts. A direct lookup
            # avoids scanning thousands of cached row tiles on every GEMV.
            key=projection_key(layer,projection,384,info,scale)
            entry=ops.weight_cache.entries.get(key)
            candidates=[] if entry is None else [(key,entry)]
        else:candidates=ops.weight_cache.entries.items()
        for key,entry in candidates:
            if (isinstance(key,tuple) and len(key)==8 and key[:3]==('native-complete-experts',layer,projection) and
                type(key[3]) is int and expert<key[3] and key==projection_key(layer,projection,key[3],info,scale)):
                expected=key[3]*(info['bytes']+scale['bytes'])
                if entry.bytes!=expected:raise ValueError('Complete expert cache parent byte budget changed')
                return key,expert,info,scale
    return None


@contextmanager
def projection_view(ops,name,start,end):
    found=existing_projection(ops,name)
    if found is None:yield None;return
    key,expert,info,scale=found;cache=ops.weight_cache
    if type(start) is not int or type(end) is not int or not 0<=start<end<=info['shape'][0]:
        raise ValueError('Invalid complete expert row view')
    if scale['shape'][0]!=info['shape'][0]:raise ValueError('Native expert scales are not per weight row')
    leased=False;parent=None
    # Atomic lookup/lease, including old resident cache classes: do not let an
    # eviction race turn a reuse request into an unexpected whole-layer load.
    with cache.lock:
        entry=cache.entries.get(key)
        if entry is not None:
            if tuple(entry.tensors[0].shape)!=(key[3],*info['shape']) or tuple(entry.tensors[1].shape)!=(key[3],*scale['shape']):
                raise ValueError('Complete expert parent shape changed')
            entry.leases+=1;cache.entries.move_to_end(key);cache.hits+=1
            cache.saved_load_bytes+=(end-start)*(info['bytes']//info['shape'][0]+scale['bytes']//scale['shape'][0])
            parent=entry.tensors;leased=True
    if not leased:yield None;return
    try:
        pair=tuple(value[expert,start:end] for value in parent)
        if any(not value.is_contiguous() for value in pair):raise ValueError('Complete expert row view must be contiguous')
        yield pair
    finally:
        if ops.device.type=='cuda':
            stream=torch.cuda.current_stream(ops.device)
            for value in parent:value.record_stream(stream)
        with cache.lock:cache.entries[key].leases-=1


def complete_expert_cached(ops,prefix):
    return all(existing_projection(ops,prefix+'.'+projection+'.weight') is not None for projection in ('w1','w2','w3'))
