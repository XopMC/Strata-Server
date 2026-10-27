"""Per-fresh-child index0/1 selection; no persistent hardware allowlist."""
import os,sys
from index_devices_v512 import resolve_current_pair
_state=None
_pid=None

def current_pair():
 global _state,_pid
 if _pid!=os.getpid():
  if any(n in sys.modules for n in ('torch','cupy','pycuda.driver')):
   raise RuntimeError('Resolve current NVIDIA indices0/1 before CUDA-capable imports')
  _state=resolve_current_pair();_pid=os.getpid()
 return tuple(dict(r) for r in _state)

def uuids():return tuple(r['uuid'] for r in current_pair())
def mask():return ','.join(uuids())

def verify_cuda_pair(torch):
 expected=uuids()
 if torch.cuda.device_count()!=2:raise RuntimeError('Exactly two visible CUDA devices required')
 for i,want in enumerate(expected):
  got=str(getattr(torch.cuda.get_device_properties(i),'uuid','')).lower().removeprefix('gpu-')
  if got!=want.lower().removeprefix('gpu-'):
   raise RuntimeError('CUDA logical order differs from freshly resolved NVIDIA indices0/1')
 return expected
