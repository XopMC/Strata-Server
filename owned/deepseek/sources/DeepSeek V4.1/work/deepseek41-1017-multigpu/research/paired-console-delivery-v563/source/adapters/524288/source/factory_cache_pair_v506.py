"""Reuse unchanged compiled native functions without repeated JIT lookup work."""
from contextlib import contextmanager
from functools import lru_cache
import os


@contextmanager
def native_launch_cache(reference):
    from stage_devices_pair_v506 import validate_visible_pair
    validate_visible_pair()
    if reference.device.type!='cuda' or reference.device.index not in (None,0):
        raise ValueError('Pair native launch cache still requires logical CUDA0 tensors')
    module=reference.ops.kernels
    if getattr(module,'_ds41_factory_cache_active',False):raise ValueError('Nested native factory caching is not supported')
    originals={};cached={};statistics={'factories':{},'retains_weight_tensors':False,'changes_compiled_kernels':False}
    try:
        module._ds41_factory_cache_active=True
        for name in ('fp8_gemm_kernel','fp4_gemm_kernel'):
            original=getattr(module,name)
            originals[name]=original
            cache=lru_cache(maxsize=128)(original);cached[name]=cache
            setattr(module,name,cache)
        yield statistics
    finally:
        for name,original in originals.items():
            cache=cached[name];info=cache.cache_info()
            statistics['factories'][name]={'hits':info.hits,'misses':info.misses,'retained_functions':info.currsize,'maximum':info.maxsize}
            setattr(module,name,original);cache.cache_clear()
        module._ds41_factory_cache_active=False
