"""Keep the exact CPU-parent bank class; never import/copy an alternate owner."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import os
from pathlib import Path
import sys
import threading

WORK = Path(_ds_publication_source('DeepSeek V4.1/work'))
_USED_PID = None
_claim_lock = threading.Lock()


def require_owner_module():
    module = sys.modules.get('one_bank_store')
    if module is None or not isinstance(getattr(module, 'OneBankStore', None), type):
        raise RuntimeError('CPU bank parent must import canonical one_bank_store before pair runtime/fork')
    if Path(module.D) != WORK/'deepseek41' or Path(module.Q) != WORK/'deepseek41-1017-nvfp4':
        raise RuntimeError('Bank module must retain its canonical producer roots, not copied snapshot roots')
    return module


def claim_runtime(store):
    global _USED_PID
    module = require_owner_module()
    if not isinstance(store, module.OneBankStore):
        raise TypeError('Pair runtime must borrow the exact inherited parent OneBankStore class')
    report = store.validate_selected_owner(deep=True)
    if report.get('owner_pid') == os.getpid() or not report.get('runtime_admissible'):
        raise RuntimeError('Prepare CPU-only bank in parent, then fork one fresh GPU child')
    with _claim_lock:
        if _USED_PID == os.getpid():
            raise RuntimeError('One pair runtime per fresh GPU child; old named CPU cache has no unregister')
        _USED_PID = os.getpid()  # failed construction also consumes this child
    return module
