"""Bounded byte-exact pinned staging for already packed native expert layers."""
import time
import numpy as np
import torch


def staging_plan(weight_bytes,scale_bytes,count,*,budget_bytes=64*1024**2):
    if any(type(value) is not int or value<1 for value in (weight_bytes,scale_bytes,count,budget_bytes)):
        raise ValueError('Invalid packed staging byte geometry')
    row_bytes=weight_bytes+scale_bytes
    lanes=2
    per_lane=budget_bytes//lanes
    batch=min(count,per_lane//row_bytes)
    if batch<1:raise MemoryError('One packed expert exceeds a staging lane')
    return {'lanes':lanes,'batch_experts':batch,'lane_bytes':batch*row_bytes,
        'total_pinned_bytes':lanes*batch*row_bytes,'budget_bytes':budget_bytes,
        'batches':(count+batch-1)//batch}


def upload_projection(ops,layer,projection,count,info,scale):
    plan=staging_plan(info['bytes'],scale['bytes'],count)
    codes=torch.empty(count,*info['shape'],device=ops.device,dtype=torch.uint8)
    scales=torch.empty(count,*scale['shape'],device=ops.device,dtype=torch.uint8)
    # Layerwise runs under torch.device('cuda:0'); pinned staging must still
    # explicitly allocate HOST memory rather than inherit that device context.
    slabs=[torch.empty(plan['lane_bytes'],dtype=torch.uint8,device='cpu',pin_memory=True) for _ in range(plan['lanes'])]
    events=[None]*plan['lanes'];started=time.perf_counter();read=getattr(ops.store,'raw_view',ops.store.raw)
    try:
        for batch,first in enumerate(range(0,count,plan['batch_experts'])):
            lane=batch%plan['lanes'];number=min(plan['batch_experts'],count-first)
            if events[lane] is not None:events[lane].synchronize()
            slab=slabs[lane];host=slab.numpy()
            weight_size=number*info['bytes'];scale_size=number*scale['bytes']
            for offset,expert in enumerate(range(first,first+number)):
                prefix=f'layers.{layer}.ffn.experts.{expert}.{projection}'
                for suffix,expected,begin in (('weight',info,offset*info['bytes']),
                                               ('scale',scale,weight_size+offset*scale['bytes'])):
                    name=prefix+'.'+suffix;actual=ops.store.info(name)
                    if any(actual[key]!=expected[key] for key in ('shape','dtype','bytes')):
                        raise ValueError('Source complete expert layer has mixed representations')
                    value=np.frombuffer(read(name),dtype=np.uint8)
                    if len(value)!=expected['bytes']:raise ValueError('Packed expert payload byte size changed')
                    host[begin:begin+len(value)]=value
            codes[first:first+number].copy_(slab[:weight_size].view(number,*info['shape']),non_blocking=True)
            scales[first:first+number].copy_(slab[weight_size:weight_size+scale_size].view(number,*scale['shape']),non_blocking=True)
            events[lane]=torch.cuda.Event();events[lane].record(torch.cuda.current_stream(ops.device))
    finally:
        # No pinned slab can be reused/released while a DMA still consumes it.
        for event in events:
            if event is not None:event.synchronize()
    stats=getattr(ops,'packed_staging_statistics',None)
    if stats is None:stats={'projection_uploads':0,'payload_bytes':0,'host_pinned_peak_bytes':0,'upload_seconds':0.};ops.packed_staging_statistics=stats
    stats['projection_uploads']+=1;stats['payload_bytes']+=count*(info['bytes']+scale['bytes'])
    stats['host_pinned_peak_bytes']=max(stats['host_pinned_peak_bytes'],plan['total_pinned_bytes'])
    stats['upload_seconds']+=time.perf_counter()-started
    return codes.view(torch.float4_e2m1fn_x2),scales.view(torch.float8_e8m0fnu)
