"""Hash-bound CPU component admission; full model quality is a separate gate."""
import hashlib
import json
import math
from pathlib import Path


def admit(store):
    root=Path(__file__).resolve().parents[3]
    proof=json.loads((root.parents[1]/'outputs/deepseek41/resident-lab/ram-isolated-wxa8-cpu-v2.json').read_text())
    child=proof.get('child',{})
    files=[root/'Strata/backend/deepseek41'/name for name in
        ('cpu_wxa8_batch.cpp','cpu_mxfp4.cpp','own_uniform_cpu_v2.cpp','wxa8_cpu_batch.py')]
    files += [root/'build/libdeepseek41_wxa8_cpu_batch_v1.so',root/'build/wxa8_cpu_boundary.cpp']
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
    checks=child.get('checks',[])
    if (proof.get('phase')!='serial_CPU_WXA8_component_complete' or
        child.get('implementation_sha256')!=expected or child.get('CUDA_initialized')is not False or
        child.get('source_revision')!=store.ram_preload_report['source_revision'] or
        child.get('boundary_sanitizer_checks')!=81 or len(checks)!=18 or
        child.get('both_sockets_computed_actual_rows')is not True or
        {(c.get('source_tensor'),c.get('bits'))for c in checks}!={
            (name,bits)for name in('layers.0.ffn.experts.0.w1.weight','layers.20.ffn.experts.17.w2.weight',
                                 'layers.39.ffn.experts.383.w3.weight')for bits in(0,2,3,4,6,8)} or
        any(c.get('replays_bitexact')is not True or set(c.get('cpu_rows_by_socket',{}))!={'0','1'} or
            any(type(v)is not int or v<=0 for v in c['cpu_rows_by_socket'].values()) or
            any(type(c.get(key))not in(int,float) or not math.isfinite(c[key]) or not 0<=c[key]<=limit
                for key,limit in(('independent_A8_relative_F64_error',2e-6),
                                  ('original_FP8_activation_relative_F64_error',.015)))for c in checks)):
        raise ValueError('WXA8 CPU lacks unchanged18 actual source component proofs')
    return root/'build/libdeepseek41_wxa8_cpu_batch_v1.so'
