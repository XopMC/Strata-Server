"""Observe and release unused process-local CUDA pools between native requests.

CUDA graph memory nodes use a driver pool that is separate from the Torch
allocator. Destroying the graph does not return its unused physical pages.
No device reset, visibility change, pool-policy change or live allocation free.
The caller must synchronize all request work before trimming.
"""
import ctypes as C
from native_cuda_loader_pair_v506 import load_isolated


def _functions(path):
    lib = load_isolated(path)
    definitions = {
        'cudaGetDevice': ([C.c_void_p], C.c_int),
        'cudaDeviceGetGraphMemAttribute': ([C.c_int, C.c_int, C.c_void_p], C.c_int),
        'cudaDeviceGraphMemTrim': ([C.c_int], C.c_int),
        'cudaDeviceGetDefaultMemPool': ([C.c_void_p, C.c_int], C.c_int),
        'cudaMemPoolGetAttribute': ([C.c_void_p, C.c_int, C.c_void_p], C.c_int),
        'cudaMemPoolTrimTo': ([C.c_void_p, C.c_size_t], C.c_int),
    }
    for name, (arguments, result) in definitions.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = arguments, result
    device = C.c_int(-1)
    _check(lib.cudaGetDevice(C.byref(device)), 'cudaGetDevice')
    if device.value != 0:
        raise ValueError('Request memory management admits only visible GPU0')
    return lib


def _check(code, operation):
    if code:
        raise RuntimeError(operation + ' returned CUDA status ' + str(code))


def pool_snapshot(path):
    lib = _functions(path)
    graph = {}
    for name, attribute in (('used_bytes', 0), ('reserved_bytes', 2)):
        value = C.c_uint64()
        _check(lib.cudaDeviceGetGraphMemAttribute(0, attribute, C.byref(value)), name)
        graph[name] = value.value
    pool = C.c_void_p()
    _check(lib.cudaDeviceGetDefaultMemPool(C.byref(pool), 0), 'default memory pool')
    default = {}
    for name, attribute in (('used_bytes', 7), ('reserved_bytes', 5)):
        value = C.c_uint64()
        _check(lib.cudaMemPoolGetAttribute(pool, attribute, C.byref(value)), name)
        default[name] = value.value
    return {'driver_graph_pool': graph, 'default_async_pool': default}


def trim_unused(path, torch):
    lib = _functions(path)
    torch.cuda.synchronize(0)
    before = pool_snapshot(path)
    applied = []
    graph = before['driver_graph_pool']
    if graph['reserved_bytes'] > graph['used_bytes']:
        _check(lib.cudaDeviceGraphMemTrim(0), 'cudaDeviceGraphMemTrim')
        applied.append('unused_driver_graph_pool')
    default = before['default_async_pool']
    if default['reserved_bytes'] > default['used_bytes']:
        pool = C.c_void_p()
        _check(lib.cudaDeviceGetDefaultMemPool(C.byref(pool), 0), 'default memory pool')
        _check(lib.cudaMemPoolTrimTo(pool, 0), 'cudaMemPoolTrimTo')
        applied.append('unused_default_async_pool')
    return {'before': before, 'after': pool_snapshot(path), 'trimmed': applied}
