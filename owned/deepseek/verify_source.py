#!/usr/bin/env python3
"""Finite CPU-only source/manifest checks; imports no exported runtime module."""
import ast
import hashlib
import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    build = json.loads((root / 'build-manifest.json').read_text())
    sources = json.loads((root / 'source-manifest.json').read_text())['files']
    lifecycle = json.loads((root / 'lifecycle-manifest.json').read_text())
    recovery = json.loads((root / 'recovery-index.json').read_text())
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    check(not build['unresolved'], 'unresolved native input')
    check(not build['source_problems'], 'native source/header issue')
    outputs = [u['output'] for u in build['units'].values()]
    check(len(set(outputs)) == len(outputs), 'colliding outputs')
    for relative, row in {**sources, **lifecycle['files']}.items():
        path = root / relative
        check(path.is_file() and not path.is_symlink(), 'missing/nonregular export: ' + relative)
        check(hashlib.sha256(path.read_bytes()).hexdigest() == row['exported_sha256'], 'changed export: ' + relative)
        check(path.suffix not in ('.a', '.o', '.so', '.pyc', '.gguf', '.safetensors'), 'binary/payload export')
        check(relative.isascii(), 'non-ASCII export filename')
        text = path.read_text()
        check('/home/' not in text and '/media/' not in text, 'machine source dependency in ' + relative)
        check(not re.search(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}', text), 'persistent hardware identity in ' + relative)
        if path.suffix == '.py':
            ast.parse(text)
    for sha, unit in build['units'].items():
        check(len(sha) == 64, 'object SHA')
        check(unit['argv'][0] in ('@CXX@', '@NVCC@'), 'unbound compiler')
        check(unit['argv'][unit['argv'].index('-o') + 1] == unit['output'], 'output binding')
        for relative in unit['sources']:
            check(relative in sources, 'source outside exported closure')
        for arg in unit['argv']:
            check('/home/' not in arg and '/media/' not in arg, 'private compile path')
            check(not (arg.startswith('@SOURCE_ROOT@') and arg.endswith(('.o', '.a', '.so'))), 'frozen binary compile input')
        if any(s.endswith('native_experts_cohort_v539.cpp') for s in unit['sources']):
            for flag in ('-fno-fast-math', '-ffp-contract=off', '-frounding-math', '-march=znver4'):
                check(flag in unit['argv'], 'lost CPU numerical flag ' + flag)
    for archive in build['archives'].values():
        names = [m['name'] for m in archive['members']]
        check(len(names) == len(set(names)), 'duplicate archive member needs occurrence-aware extraction')
        for member in archive['members']:
            check(member['object'] in build['units'], 'unresolved archive member')
    for item in recovery['objects']:
        if item['member'] is not None:
            check(item['occurrence'] == 1, 'unqualified archive occurrence')
    # Exact sequence of old own object/archive/DSO inputs, with repetitions intact.
    by_input = {r['input']: r['sha256'] for r in recovery['objects'] if r['member'] is None}
    for context, old in recovery['links'].items():
        expected = []
        for item in old['ordered_inputs']:
            if 'system_library' in item:
                expected.append('@CUDA_LIBRARY_DIR@/' + item['system_library'])
            elif item['input'] in build['archives']:
                expected.append(build['archives'][item['input']]['output'])
            else:
                expected.append(build['units'][by_input[item['input']]]['output'])
        engine = build['engines'][context]
        argv = engine['argv']
        oi = argv.index('-o') + 1
        actual = [arg for i, arg in enumerate(argv) if i != oi and not arg.startswith('-') and arg.endswith(('.o', '.a', '.so'))]
        check(actual == expected, 'changed link order ' + context)
        check('-Wl,-soname,libds1017_pair_runtime_v1.so' in argv, 'changed main C ABI SONAME')
        check('-fuse-ld=mold' in argv and '-Wl,-z,defs' in argv, 'link admission flags')
    check(set(build['engines']) == {'262144', '524288'}, 'context set')
    delta = json.loads((root / 'diagnostic-admission-delta.json').read_text())
    check(len(delta['files']) == 4 and not delta['numerical_leaves_changed'] and
          not delta['C_ABI_signatures_changed'] and not delta['production_native_peer_admission_changed'],
          'diagnostic-only admission derivative scope')
    check(sorted(d['expected_visible_count'] for d in delta['files']) == [1, 1, 2, 2], 'admitted cohort widths')
    check(sum(d['scope'] == 'diagnostic_only' for d in delta['files']) == 3 and
          sum(d['scope'] == 'byte_inverse_device_binding' for d in delta['files']) == 1, 'admission port scope')
    for refused in lifecycle['unresolved']:
        check(not (root / 'sources' / refused['source']).exists(), 'refused old hardware-bound module was exported')
    check(lifecycle['status'] == 'PUBLICATION_IMPORTED_FACTORY_EXPLICITLY_DISABLED_ROOT_REBIND_REQUIRED' and lifecycle.get('publication_factory_enabled') is False, 'fake lifecycle admission')
    factory_delta = json.loads((root / 'factory-disabled-delta.json').read_text())
    check(not factory_delta['numerical_or_C_ABI_changes'] and len(factory_delta['files']) == 3, 'disabled factory scope')
    for row in factory_delta['files']:
        tree = ast.parse((root / row['path']).read_text())
        values = [n.value for n in tree.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == 'PUBLICATION_FACTORY_ENABLED' for t in n.targets)]
        check(len(values) == 1 and isinstance(values[0], ast.Constant) and values[0].value is False, 'factory gate must stay false')
        for name in row['guarded_entrypoints']:
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            gate = fn.body[0]
            check(isinstance(gate, ast.If) and isinstance(gate.test, ast.UnaryOp) and
                  isinstance(gate.test.op, ast.Not) and isinstance(gate.test.operand, ast.Name) and
                  gate.test.operand.id == 'PUBLICATION_FACTORY_ENABLED' and
                  isinstance(gate.body[0], ast.Raise), 'entrypoint must refuse before original body: ' + name)
    topology = json.loads((root / 'current-numa-topology-delta.json').read_text())
    check(len(topology['files']) == 4 and not topology['numerical_leaves_changed'] and
          not topology['class_layout_or_C_ABI_changed'] and
          not topology['stream_copy_credit_or_thread_lifecycle_changed'] and
          not topology['production_native_peer_admission_changed'] and
          topology['serving_factory_enabled'] is False, 'NUMA topology-only derivative scope')
    check(sorted(row['context'] for row in topology['files']) == [262144, 262144, 524288, 524288],
          'both selected contexts must resolve current NUMA topology')
    headers = []
    for row in topology['files']:
        text = (root / row['path']).read_text()
        check(hashlib.sha256(text.encode()).hexdigest() == row['after_sha256'], 'NUMA source hash')
        check(sources[row['path']]['pre_current_numa_port_exported_sha256'] == row['before_sha256'] and
              row['full_file_reverse_scope_proof'] is True, 'NUMA parent/scope pin')
        check(not re.search(r'/sys/bus/pci/devices/[0-9a-fA-F]{4,8}:', text), 'saved PCI locality path')
        check('cudaSetDevice' not in text, 'locality lookup must not change current device')
        if row['path'].endswith('.hpp'):
            headers.append(text)
            check('#include <cuda_runtime_api.h>' in text and '#include <cstdio>' in text,
                  'self-contained CUDA/PCI resolver declarations')
            for guard in ('cudaGetDevice(&device) != cudaSuccess',
                          'expected_device >= 0 && device != expected_device',
                          'cudaDeviceGetPCIBusId(pci, sizeof(pci), device) != cudaSuccess',
                          'bus > 255 || slot > 31 || function > 7',
                          'static_cast<size_t>(length) >= sizeof(canonical)',
                          'if (!(in >> node) || node < 0 || (in >> extra))'):
                check(guard in text, 'missing fresh-device/PCI/sysfs failure guard: ' + guard)
            check('std::string("/sys/bus/pci/devices/") + canonical + "/numa_node"' in text,
                  'sysfs path must be formed from runtime PCI query')
            body = text[text.index('class NativeStageCopyPoolV2 {'):]
            lookup = ('        const int node = stage_copy_current_numa_node_v2(-1, "GPU NUMA node unavailable for stage-copy pool");\n'
                      '        const int main_cpu = sched_getcpu();\n')
        else:
            body = text
            lookup = '    const int node=stage_copy_current_numa_node_v2(device,"325 actual GPU NUMA node unavailable");\n'
            check('if(pool){if(bound!=device)' in text, 'rank pool must retain ordinal lifetime guard')
        check(body.count(lookup) == 1, 'one current-owner NUMA lookup per pool creation')
        check(hashlib.sha256(body.replace(lookup, '@CURRENT_CUDA_NUMA_LOOKUP@\n', 1).encode()).hexdigest() ==
              row['body_excluding_topology_lookup_sha256'], 'changed copy/worker/rank body')
    check(len(headers) == 2 and headers[0] == headers[1], 'context NUMA/pool headers must match')
    report = dict(status='PASS', checks=checks, native_units=len(build['units']),
        native_archives=len(build['archives']), native_source_notice_files=len(sources),
        lifecycle_source_files=len(lifecycle['files']), lifecycle_explicit_refusals=lifecycle['unresolved'],
        publication_diagnostic_ports=3, byte_inverse_identity_ports=1, selected_sources_persistent_UUIDs=False,
        current_cuda_numa_ports=4, topology_runtime_or_affinity_execution=False,
        serving_factory_explicitly_disabled=True,
        compiler_executed=False, cuda_initialized=False, model_or_weights_read=False,
        clean_source_build_pending=True, numerical_runtime_parity_pending=True)
    (root / 'STATIC_CHECKS.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DeepSeek source checks PASS:', checks, '; no compiler/CUDA/model execution')


if __name__ == '__main__':
    main()
