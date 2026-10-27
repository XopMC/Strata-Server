"""Admit the separately tested own CPU component without changing source math."""
import hashlib
import json
from pathlib import Path


def admit(store):
    root=Path(__file__).resolve().parents[3]
    path=root.parents[1]/'outputs/deepseek41/own-uniform-cpu-source-components-v1.json'
    proof=json.loads(path.read_text())
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(
        root/'Strata/backend/deepseek41/own_uniform_cpu.cpp',root/'Strata/backend/deepseek41/own_cpu_rows.py',
        root/'build/libdeepseek41_own_uniform_cpu_v1.so',root/'build/own_uniform_cpu_boundary.cpp')}
    if(proof.get('phase')!='actual_source_own_CPU_SIMD_component_validated'or
       proof.get('source_revision')!=store.ram_preload_report['source_revision']or
       proof.get('CUDA_initialized')is not False or proof.get('implementation_sha256')!=expected or
       len(proof.get('checks',[]))!=15 or proof.get('boundary_sanitizer_checks_passed')!=75 or
       not proof.get('both_sockets_computed_actual_rows')or
       any(not c.get('replay_bits_exact')or not c.get('relative_F64_errors')or
           any(type(e)not in(float,int)or not 0<=e<=2e-6 for e in c['relative_F64_errors'])for c in proof['checks'])):
        raise ValueError('Own SIMD CPU rows require matching actual-source and boundary sanitizer proof')
    return root/'build/libdeepseek41_own_uniform_cpu_v1.so'


def admit_v2(store):
    """Faster integer unpacking must preserve the independently admitted v1 dot."""
    baseline=admit(store)
    root=Path(__file__).resolve().parents[3]
    proof=json.loads((root.parents[1]/'outputs/deepseek41/own-uniform-cpu-source-components-v2.json').read_text())
    files=[root/'Strata/backend/deepseek41/own_uniform_cpu_v2.cpp',
           root/'Strata/backend/deepseek41/own_cpu_rows.py',
           root/'build/libdeepseek41_own_uniform_cpu_v2.so',
           root/'build/own_uniform_cpu_boundary_v2.cpp']
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
    cells={(name,bits)for name in ('layers.0.ffn.experts.0.w1.weight',
                                  'layers.20.ffn.experts.17.w2.weight',
                                  'layers.39.ffn.experts.383.w3.weight')for bits in (2,3,4,6,8)}
    checks=proof.get('checks',[])
    if (proof.get('phase')!='actual_source_own_CPU_SIMD_v2_component_validated' or
        proof.get('source_revision')!=store.ram_preload_report['source_revision'] or
        proof.get('implementation_sha256')!=expected or
        proof.get('baseline_library_sha256')!=hashlib.sha256(baseline.read_bytes()).hexdigest() or
        proof.get('CUDA_initialized')is not False or proof.get('boundary_sanitizer_checks_passed')!=75 or
        proof.get('both_sockets_computed_actual_rows')is not True or
        len(checks)!=15 or {(c.get('source_tensor'),c.get('bits'))for c in checks}!=cells or
        any(c.get('all_outputs_bitexact_v1')is not True or c.get('replay_bits_exact')is not True or
            len(c.get('relative_F64_errors',[]))!=2 or
            any(type(e)not in (float,int)or not 0<=e<=2e-6 for e in c['relative_F64_errors'])for c in checks)):
        raise ValueError('Own SIMD V2 needs matching source bitexact and sanitizer proofs')
    return root/'build/libdeepseek41_own_uniform_cpu_v2.so'
