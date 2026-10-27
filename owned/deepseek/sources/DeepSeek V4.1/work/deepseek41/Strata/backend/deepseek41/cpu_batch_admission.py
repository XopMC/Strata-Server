"""Component admission; complete candidate quality remains mandatory."""
import hashlib
import json
from pathlib import Path


def admit(store):
    root=Path(__file__).resolve().parents[3]
    proof=json.loads((root.parents[1]/'outputs/deepseek41/cpu-batch-source-components-v2.json').read_text())
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(
        root/'Strata/backend/deepseek41/cpu_mxfp4.cpp',root/'Strata/backend/deepseek41/cpu_mxfp4_batch.cpp',
        root/'Strata/backend/deepseek41/cpu_batch_pool.py',root/'build/libdeepseek41_mxfp4_cpu_batch_v1.so')}
    if(proof.get('phase')!='actual_source_cpu_batch_components_bitexact'or
       proof.get('implementation_sha256')!=expected or proof.get('CUDA_initialized')is not False or
       proof.get('source_revision')!=store.ram_preload_report['source_revision']or
       not proof.get('both_sockets_computed_actual_rows')or len(proof.get('checks',[]))!=6 or
       any(not c.get('all_outputs_bitexact')or not all(c.get('socket_rows',{}).values())for c in proof['checks'])):
        raise ValueError('CPU batch dispatch lacks matching actual source component proof')
    return root/'build/libdeepseek41_mxfp4_cpu_batch_v1.so'
