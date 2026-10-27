"""Address GPU0 via lazy NVML/PCI lookup, without CUDA or other GPU handles."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import ctypes as C
from datetime import datetime
import json
import os
from pathlib import Path

from index_selection_v512 import current_pair
_SELECTED = current_pair()[0]
UUID = _SELECTED["uuid"]
PCI = _SELECTED["pci_bus_id"]
ROOT = Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41'))


class Memory(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]


class Process(C.Structure):
    _fields_ = [('pid', C.c_uint), ('usedGpuMemory', C.c_ulonglong),
               ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]


class Value(C.Union):
    _fields_ = [('dVal', C.c_double), ('uiVal', C.c_uint), ('ulVal', C.c_ulong),
               ('ullVal', C.c_ulonglong), ('sllVal', C.c_longlong)]


class Field(C.Structure):
    _fields_ = [('fieldId', C.c_uint), ('scopeId', C.c_uint), ('timestamp', C.c_longlong),
               ('latencyUsec', C.c_longlong), ('valueType', C.c_int),
               ('nvmlReturn', C.c_int), ('value', Value)]


def query_gpu0():
    lib = C.CDLL('libnvidia-ml.so.1')
    lib.nvmlErrorString.argtypes = [C.c_int]
    lib.nvmlErrorString.restype = C.c_char_p

    def checked(name, argtypes, *arguments):
        function = getattr(lib, name)
        function.argtypes, function.restype = argtypes, C.c_int
        code = function(*arguments)
        if code:
            raise RuntimeError(name + ': ' + lib.nvmlErrorString(code).decode())

    checked('nvmlInit_v2', [])
    try:
        handle = C.c_void_p()
        checked('nvmlDeviceGetHandleByPciBusId_v2', [C.c_char_p, C.POINTER(C.c_void_p)],
                PCI.encode(), C.byref(handle))
        uuid = C.create_string_buffer(96)
        checked('nvmlDeviceGetUUID', [C.c_void_p, C.c_void_p, C.c_uint], handle, uuid, len(uuid))
        if uuid.value.decode() != UUID:
            raise RuntimeError('The GPU0 PCI address now belongs to a different UUID')
        memory = Memory()
        checked('nvmlDeviceGetMemoryInfo', [C.c_void_p, C.POINTER(Memory)], handle, C.byref(memory))
        processes = (Process * 64)()
        count = C.c_uint(len(processes))
        checked('nvmlDeviceGetComputeRunningProcesses_v3',
                [C.c_void_p, C.POINTER(C.c_uint), C.POINTER(Process)],
                handle, C.byref(count), processes)
        recovery = Field(fieldId=230)
        checked('nvmlDeviceGetFieldValues', [C.c_void_p, C.c_int, C.POINTER(Field)],
                handle, 1, C.byref(recovery))
        if recovery.nvmlReturn or recovery.valueType != 1:
            raise RuntimeError('GPU0 recovery action cannot be verified')
        if count.value > len(processes):
            raise RuntimeError('GPU0 compute-owner list exceeds the checked buffer')
        return dict(gpu_uuid=UUID, pci_bus=PCI, compute_pids=[p.pid for p in processes[:count.value]],
                    VRAM_total_bytes=memory.total, VRAM_free_bytes=memory.free, VRAM_used_bytes=memory.used,
                    recovery_action=recovery.value.uiVal, CUDA_initialized=False,
                    other_GPU_handles_requested=False, query='lazy NVML + exact PCI handle; UUID verified')
    finally:
        checked('nvmlShutdown', [])


def require_idle_healthy(report):
    if report.get('gpu_uuid') != UUID or report.get('pci_bus') != PCI:
        raise ValueError('Exact physical GPU0 identity is required')
    if report.get('recovery_action') != 0:
        raise RuntimeError('GPU0 recovery action is ' + str(report.get('recovery_action'))
                           + '; action2 means node reboot, no new benchmark admitted')
    if not isinstance(report.get('compute_pids'), list) or report['compute_pids']:
        raise RuntimeError('GPU0 is occupied or its compute owners are unknown')
    if report.get('VRAM_free_bytes', 0) <= 0:
        raise RuntimeError('GPU0 free memory cannot be verified')
    return report


def assert_gpu0_idle():
    recovery = ROOT / 'state/GPU-driver-recovery.json'
    if recovery.exists():
        record = json.loads(recovery.read_text())
        if record.get('GPU_jobs_paused'):
            raise RuntimeError('GPU tests are paused during the user-requested driver/card recovery')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(r['uuid'] for r in current_pair()) or os.environ.get('DEEPSEEK41_GPU_STAGE', '0') != '0':
        raise ValueError('This idle check admits only the explicit GPU0-only configuration')
    return require_idle_healthy(query_gpu0())


if __name__ == '__main__':
    from warm_run_journal import durable_save
    try:
        report = query_gpu0()
        report['phase'] = 'GPU0_ready' if not report['recovery_action'] and not report['compute_pids'] else 'GPU0_not_admitted'
    except (RuntimeError, OSError) as error:
        report = dict(phase='GPU0_query_failed', gpu_uuid=UUID, pci_bus=PCI, error=str(error), CUDA_initialized=False)
    report['time'] = datetime.now().astimezone().isoformat()
    durable_save(ROOT / 'state/GPU0-health.json', report)
    print(json.dumps(report))
