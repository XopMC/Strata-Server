"""Narrow metadata-only bridge for an already-running frozen test controller.
The V1 harness reads an undefined global library_sha after successful inference.
A fresh child owns one immutable runtime/DSO; bind that field to its verified SHA.
No controller file, parent process, model operation, or other caller is modified.
"""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import hashlib,sys
from pathlib import Path
CONTROLLER=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu/research/pair-controller-v1/controller_v1.py'))
EXPECTED='85ef71be4c99db50d448c260f05742abf34b347d0d8f28ff36231a0cda4bc403'
def bind_frozen_controller_metadata(library_path, library_sha256):
    module=sys.modules.get('__main__')
    if module is None or Path(getattr(module,'__file__','')).resolve()!=CONTROLLER:return
    if hashlib.sha256(CONTROLLER.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Controller compatibility requires exact frozen V1 source')
    if hashlib.sha256(Path(library_path).read_bytes()).hexdigest()!=library_sha256:raise RuntimeError('Controller DSO metadata must match actual DSO')
    previous=getattr(module,'library_sha',library_sha256)
    if previous!=library_sha256:raise RuntimeError('One immutable runtime DSO per fresh controller child')
    module.library_sha=library_sha256
