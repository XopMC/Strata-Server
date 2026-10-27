"""Reuse immutable native BF16/F32 dense row tiles in the same GPU budget."""
from contextlib import contextmanager
import numpy as np
import torch


@contextmanager
def dense_native_cache(reference,*,include_vision=False):
    ops = reference.ops
    original = ops.rows
    if not hasattr(ops,'weight_cache'):
        raise ValueError('Dense residency requires the existing bounded native cache')
    stats = {'calls':0,'cacheable_row_tiles':0,'gpu_tensor_bytes_requested':0}
    def rows(name,indices,dtype=torch.bfloat16):
        info = ops.store.info(name)
        prefixes = ('head.','vision.','aligner.') if include_vision else ('head.',)
        if info['dtype'] not in ('BF16','F32') or not name.endswith('.weight') or not name.startswith(prefixes):
            return original(name,indices,dtype)
        values = np.asarray(indices,dtype=np.int64).reshape(-1)
        if not len(values) or np.any(values < 0) or np.any(values >= info['shape'][0]) or np.any(np.diff(values) != 1):
            return original(name,indices,dtype)
        width = info['shape'][1]
        gpu_row_bytes = width*torch.empty((),dtype=dtype,device='meta').element_size()
        chunk = min(ops._row_chunk(name),max(1,(64*1024**2)//gpu_row_bytes))
        output = []
        stats['calls'] += 1
        for first in range(0,len(values),chunk):
            part = values[first:first+chunk]
            start,end = int(part[0]),int(part[-1])+1
            key = ('native-dense',name,start,end,info['dtype'],tuple(info['shape']),str(dtype))
            expected = len(part)*gpu_row_bytes
            def load(part=part):
                weight = original(name,part,dtype)
                return weight,torch.empty(0,dtype=torch.uint8,device=ops.device)
            with ops.weight_cache.lease(key,expected,load) as pair:
                output.append(pair[0])
                if ops.device.type == 'cuda':
                    pair[0].record_stream(torch.cuda.current_stream(ops.device))
            stats['cacheable_row_tiles'] += 1
            stats['gpu_tensor_bytes_requested'] += expected
        return output[0] if len(output) == 1 else torch.cat(output)
    ops.rows = rows
    try:
        yield stats
    finally:
        ops.rows = original
