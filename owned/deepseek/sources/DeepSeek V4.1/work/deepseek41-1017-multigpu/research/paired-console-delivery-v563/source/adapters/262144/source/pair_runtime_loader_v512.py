"""CPU-only module loader. Parent bank/module must already exist before fork."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]
SOURCE = PACKET/'source'
WORK = Path(_ds_publication_source('DeepSeek V4.1/work'))
D = WORK/'deepseek41'
FROZEN_SOURCE = WORK/'deepseek41-1017-nvfp4/delivery/single-gpu-frozen-v1/source-snapshot'


def load_context_runtime(context):
    if type(context) is not int or context != 262144:
        raise ValueError('Pair V10 admits only frozen 256k/512k profiles')
    from index_selection_v512 import current_pair
    current_pair()  # NVIDIA index0/1 resolved BEFORE any numerical module import.
    # Front-load private companions. Do not alias/replace global production modules.
    for path in reversed((SOURCE, FROZEN_SOURCE, D/'tools', D/'Strata')):
        value = str(path)
        if value in sys.path:sys.path.remove(value)
        sys.path.insert(0, value)
    from owner_contract_v512 import require_owner_module
    owner = require_owner_module()
    # Verify bounded source closure before importing numerical modules; no DSO/model reads.
    pins = json.loads((PACKET/'DEPENDENCY_PINS.json').read_text())
    for row in pins['files']:
        path = Path(row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Pair V10 source dependency changed: ' + str(path))
    for already in ('backend.deepseek41.packed_cuda','backend.deepseek41.sm89_quant','backend.deepseek41.sm89_attention'):
        if already in sys.modules:raise RuntimeError('Fresh child required before numerical helper import: '+already)
    # Child-only source shadows. No alternate OneBankStore module/class.
    import importlib
    package=importlib.import_module('backend.deepseek41')
    for suffix,file in (('cuda_admission','cuda_admission_index_v512.py'),('runtime','runtime_base_index_v512.py')):
        fullname='backend.deepseek41.'+suffix
        existing=sys.modules.get(fullname)
        target=SOURCE/file
        if existing is not None and Path(existing.__file__).resolve()!=target:
            raise RuntimeError('Fresh index child required; historical backend module already imported: '+fullname)
        if existing is None:
            spec0=importlib.util.spec_from_file_location(fullname,target)
            shadow=importlib.util.module_from_spec(spec0);sys.modules[fullname]=shadow
            try:spec0.loader.exec_module(shadow)
            except BaseException:sys.modules.pop(fullname,None);raise
            setattr(package,suffix,shadow)
    name = f'backend.deepseek41._strata_pair_runtime_{context}_v512'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != SOURCE/f'runtime_{context}_pair_v512.py':
            raise RuntimeError('Pair runtime module-name collision')
        return module
    spec = importlib.util.spec_from_file_location(name, SOURCE/f'runtime_{context}_pair_v512.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if sys.modules.get('one_bank_store') is not owner:
            raise RuntimeError('Pair import replaced the inherited bank module')
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
