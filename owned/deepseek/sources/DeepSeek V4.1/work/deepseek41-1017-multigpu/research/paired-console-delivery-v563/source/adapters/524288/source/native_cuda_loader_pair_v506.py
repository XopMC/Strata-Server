"""Private pair-only deepbind loader; it does not initialize CUDA itself."""
import ctypes
import os
import threading
from pathlib import Path
from stage_devices_pair_v506 import validate_visible_pair

_lock = threading.Lock()
_handles = {}

def load_isolated(path):
    validate_visible_pair()
    if not hasattr(os, "RTLD_DEEPBIND"):
        raise RuntimeError("Pair native loader requires Linux RTLD_DEEPBIND")
    path = str(Path(path).resolve(strict=True))
    with _lock:
        if path not in _handles:
            _handles[path] = ctypes.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_DEEPBIND)
        return _handles[path]
