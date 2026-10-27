#!/usr/bin/env python3
"""Read-only historical provenance recovery; writes only this staging directory.

No compiler, CUDA initialization, DSO loading, weights, or system mutation.
The only binary-reader tool is ar, for member names/bytes/SHA provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import os
import ast
import io
import tokenize
import difflib
from collections import defaultdict


def digest(data):
    return hashlib.sha256(data).hexdigest()


def listing(root, *globs):
    argv = ['rg', '--files', '--no-ignore', str(root)]
    for glob in globs:
        argv += ['-g', glob]
    result = subprocess.run(argv, text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return [Path(s) for s in result.stdout.splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True)
    args = ap.parse_args()
    ws = args.workspace.resolve()
    out = Path(__file__).resolve().parent
    base = ws / 'DeepSeek V4.1/work/deepseek41'
    multi = ws / 'DeepSeek V4.1/work/deepseek41-1017-multigpu'
    single = ws / 'DeepSeek V4.1/work/deepseek41-1017-nvfp4'
    roots = [base, multi, single]
    receipt = multi / 'state/pass-upload-async-native-root-build-v558.json'
    actual = json.loads(receipt.read_text())
    commands = defaultdict(list)
    metadata = {}

    def rel(path):
        return str(Path(path).relative_to(ws))

    def pathof(s, cwd):
        p = Path(s)
        return p if p.is_absolute() else cwd / p

    def tokens(text):
        # Migrated Make metadata has unquoted paths containing spaces. Replace
        # the known workspace prefix BEFORE shlex, never execute its shell text.
        protected = text.replace(str(ws), '@WORKSPACE@').replace('DeepSeek V4.1', 'DeepSeek@SPACE@V4.1')
        return [s.replace('@WORKSPACE@', str(ws)).replace('@SPACE@', ' ') for s in shlex.split(protected)]

    def add(argv, cwd, evidence, score):
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            return
        # taskset's -c is affinity; locate the compiler first.
        start = next((i for i, s in enumerate(argv)
                      if Path(s).name in ('nvcc', 'clang++-18', 'clang++', 'g++', 'c++', 'gcc', 'clang-18')), None)
        if start is None:
            return
        argv = argv[start:]
        if '-o' not in argv or argv.index('-o') + 1 >= len(argv):
            return
        target = pathof(argv[argv.index('-o') + 1], cwd)
        if not target.suffix in ('.o', '.so'):
            return
        source = [x for x in argv[1:] if x.endswith(('.cpp', '.cc', '.c', '.cu')) and not x.startswith('-')]
        if not source and target.suffix != '.so':
            return
        if source and all(not Path(s).is_absolute() for s in source) and \
                not all((cwd / s).is_file() for s in source) and all((ws / s).is_file() for s in source):
            # The initial packer receipt records workspace-relative argv and log
            # names. Its unique existing source path fixes the omitted cwd.
            cwd = ws
            target = pathof(argv[argv.index('-o') + 1], cwd)
        commands[str(target)].append(dict(argv=argv, cwd=str(cwd), evidence=rel(evidence),
                                           score=score, sources=[str(pathof(s, cwd)) for s in source]))

    def walk(node, cwd, evidence, score):
        if isinstance(node, dict):
            cwd = Path(node.get('cwd', cwd)) if isinstance(node.get('cwd', cwd), (str, Path)) else cwd
            argv = node.get('argv', node.get('args', node.get('cmd', node.get('command'))))
            if isinstance(argv, str) and '$' not in argv and '\n' not in argv:
                try:
                    argv = tokens(argv)
                except ValueError:
                    argv = None
            if isinstance(argv, list):
                add(argv, cwd, evidence, score + (10 if node.get('exit_code') == 0 else 0))
            for key, value in node.items():
                if key not in ('stdout', 'stderr', 'output', 'log', 'argv', 'args'):
                    walk(value, cwd, evidence, score)
        elif isinstance(node, list):
            if node and all(isinstance(x, str) for x in node):
                add(node, cwd, evidence, score)
                return
            for value in node:
                walk(value, cwd, evidence, score)

    json_count = 0
    for root in roots + [ws / 'DeepSeek V4.1/outputs/deepseek41/analysis']:
        for file in listing(root, '*.json'):
            if file.stat().st_size > 16 * 1024 * 1024:
                continue
            try:
                data = json.loads(file.read_text())
            except (ValueError, UnicodeError, OSError):
                continue
            json_count += 1
            score = 100 if file == receipt else 0
            walk(data, base, file, score)
    # CMake Unix Makefiles are source build metadata, not executable commands here.
    make_count = 0
    for root in roots:
        for file in listing(root, 'build.make'):
            flagsfile = file.with_name('flags.make')
            if not flagsfile.exists():
                continue
            lines = file.read_text().splitlines()
            variables = dict(re.findall(r'^(\w+) = (.*)$', flagsfile.read_text(), re.M))
            binary = re.search(r'^CMAKE_BINARY_DIR = (.*)$', file.read_text(), re.M)
            if not binary:
                continue
            cwd = Path(binary[1])
            for line in lines:
                if line.startswith('\t') and ' -c ' in line and ' -o ' in line and '.o' in line:
                    expanded = re.sub(r'\$\((\w+)\)', lambda m: variables.get(m[1], m[0]), line.strip())
                    if '$(' in expanded:
                        continue
                    try:
                        add(tokens(expanded), cwd, file, -10)
                        make_count += 1
                    except ValueError:
                        pass
    walk(actual, base, receipt, 100)
    print('Indexed JSON', json_count, 'Make compile rules', make_count, 'object outputs', len(commands), flush=True)

    links = [c for c in actual['steps'] if c['label'].startswith('link558-')]
    wanted = []
    link_models = {}
    for link in links:
        context = link['label'].split('-')[-1]
        argv = link['argv']
        output_index = argv.index('-o') + 1
        order = []
        for i, s in enumerate(argv):
            if i == output_index or s.startswith('-') or not s.endswith(('.o', '.a', '.so')):
                continue
            path = pathof(s, Path(link['cwd']))
            if not str(path).startswith(str(ws)):
                order.append(dict(system_library=path.name))
                continue
            key = rel(path)
            order.append(dict(input=key))
            if path.suffix == '.a':
                members = subprocess.check_output(['ar', 't', str(path)], text=True).splitlines()
                if len(members) != len(set(members)):
                    raise RuntimeError('duplicate members require indexed extraction: ' + key)
                for member in members:
                    data = subprocess.check_output(['ar', 'p', str(path), member])
                    wanted.append(dict(context=context, input=key, member=member, occurrence=1,
                                       sha256=digest(data), bytes=len(data)))
            else:
                wanted.append(dict(context=context, input=key, member=None,
                                   sha256=digest(path.read_bytes()), bytes=path.stat().st_size))
        link_models[context] = dict(evidence=rel(receipt), ordered_inputs=order)

    # Actual factory563 additionally loads CPU418 and Task489. Cold565 uses the
    # existing lossless v4->output16 packer; include its source target, not its DSO.
    delivery = json.loads((multi / 'research/paired-console-delivery-v563/DELIVERY_BINDINGS.json').read_text())
    profile = json.loads((multi / 'research/paired-console-server-v565/PROFILE_TEMPLATES.json').read_text())
    extra = [delivery['contexts']['262144'][name] for name in ('CPU_projection', 'Task_library')]
    extra.append(dict(path=profile['262144']['packer'], sha256=profile['262144']['packer_sha256']))
    for row in extra:
        path = Path(row['path'])
        data = path.read_bytes()
        if digest(data) != row['sha256']:
            raise RuntimeError('Selected factory dependency SHA mismatch: ' + rel(path))
        wanted.append(dict(context='factory', input=rel(path), member=None, sha256=digest(data), bytes=len(data)))

    needed_hashes = {w['sha256'] for w in wanted}
    # A linked owned DSO is another source target, never a binary exception.
    # Recover and hash its direct object inputs from the recorded link command.
    extra_links = {}
    queue = [w for w in wanted if w['member'] is None and w['input'].endswith('.so')]
    visited_dso = set()
    while queue:
        item = queue.pop()
        if item['sha256'] in visited_dso:
            continue
        visited_dso.add(item['sha256'])
        rules = commands.get(str(ws / item['input']), [])
        if not rules:
            continue
        rule = max(rules, key=lambda r: r['score'])
        extra_links[item['sha256']] = rule
        output_index = rule['argv'].index('-o') + 1
        for i, token in enumerate(rule['argv']):
            if i == output_index or token.startswith('-') or not token.endswith(('.o', '.so', '.a')):
                continue
            path = pathof(token, Path(rule['cwd']))
            if not str(path).startswith(str(ws)) or not path.is_file():
                continue
            if path.suffix == '.a':
                raise RuntimeError('Additional DSO archive requires explicit recovery: ' + str(path))
            data = path.read_bytes()
            sha = digest(data)
            wanted.append(dict(context='shared', input=rel(path), member=None, sha256=sha, bytes=len(data)))
            needed_hashes.add(sha)
            if path.suffix == '.so':
                queue.append(wanted[-1])
    candidates = defaultdict(list)
    # SHA matching is required; a basename match by itself supplies NO evidence.
    for root in roots:
        for obj in listing(root, '*.o', '*.so'):
            try:
                sha = digest(obj.read_bytes())
            except OSError:
                continue
            if sha in needed_hashes:
                candidates[sha].append(obj)
    units = {}
    unresolved = []
    attempted = set()
    for w in wanted:
        if w['sha256'] in attempted:
            continue
        attempted.add(w['sha256'])
        possible = []
        for obj in candidates[w['sha256']]:
            for command in commands.get(str(obj), []):
                if all(Path(s).is_file() for s in command['sources']):
                    possible.append((command['score'], str(obj), command))
        if not possible:
            unresolved.append(dict(**w, reason='no SHA-matched object with recoverable compiler command',
                                   matching_objects=[rel(p) for p in candidates[w['sha256']]]))
            continue
        possible.sort(key=lambda v: (-v[0], v[1], v[2]['evidence']))
        _, obj, command = possible[0]
        units[w['sha256']] = dict(object_sha256=w['sha256'], original_object=rel(obj), **command)
    report = dict(schema=1, status='SOURCE_RECOVERY_IN_PROGRESS',
                  native_revision=558, factory_revision=563, cold_factory_revision=565,
                  actual_receipt_sha256=digest(receipt.read_bytes()),
                  objects=wanted, units=list(units.values()), links=link_models,
                  unresolved=unresolved,
                  constraints=dict(build_executed=False, gpu_executed=False, binary_exported=False))
    export_native(ws, out, report, links, tokens)
    export_lifecycle(ws, out, multi, base, single)
    # Historical command paths are provenance only. The executable build manifest
    # below refers exclusively to this source package and the installed toolkit.
    historical = json.dumps(report, indent=2, ensure_ascii=True)
    historical = historical.replace(json.dumps(str(ws), ensure_ascii=True)[1:-1], '${HISTORICAL_WORKSPACE}')
    (out / 'recovery-index.json').write_text(historical + '\n')
    print('Unique object/DSO hashes', len(needed_hashes), 'resolved', len(units), 'unresolved', len(unresolved))
    for u in unresolved:
        print('UNRESOLVED', u['member'] or Path(u['input']).name,
              u['sha256'][:12], 'matching objects', len(u['matching_objects']))


def export_native(ws, out, report, links, tokens):
    units = {u['object_sha256']: u for u in report['units']}
    source_records = {}
    problems = []
    traversed = set()
    include_pattern = re.compile(r'(^\s*#\s*include\s*[<"])([^>"\n]+)([>"])', re.M)
    diagnostic_deltas = []

    def staged(path):
        path = Path(os.path.normpath(path))
        relative = path.relative_to(ws)
        if not str(relative).isascii():
            raise RuntimeError('Non-ASCII source path requires an explicit map: ' + str(relative))
        return Path('sources') / relative

    def inside(path):
        try:
            Path(path).relative_to(ws)
            return True
        except ValueError:
            return False

    def expand(argv, cwd):
        result = []
        i = 0
        while i < len(argv):
            if argv[i] == '--options-file':
                file = Path(argv[i + 1])
                if not file.is_absolute():
                    file = cwd / file
                result += tokens(file.read_text())
                i += 2
            else:
                result.append(argv[i])
                i += 1
        return result

    def include_dirs(argv, cwd):
        dirs = []
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg in ('-I', '-isystem', '-iquote'):
                i += 1
                p = Path(argv[i])
            elif arg.startswith('-I') and len(arg) > 2:
                p = Path(arg[2:])
            else:
                i += 1
                continue
            p = Path(os.path.normpath(p if p.is_absolute() else cwd / p))
            dirs.append(p)
            i += 1
        return dirs

    def copy_source(path, dirs):
        path = Path(os.path.normpath(path))
        key = (str(path), tuple(map(str, dirs)))
        if key in traversed:
            return
        traversed.add(key)
        if not inside(path) or not path.is_file():
            problems.append(dict(kind='missing source/header', path=str(path)))
            return
        relative = staged(path)
        original = path.read_bytes()
        try:
            text = original.decode('utf-8')
        except UnicodeError:
            problems.append(dict(kind='non-text source refused', path=str(path)))
            return
        replacements = []
        for match in include_pattern.finditer(text):
            name = match[2]
            requested = Path(name)
            candidates = ([requested] if requested.is_absolute() else
                          ([path.parent / requested] if match[1].endswith('"') else []) +
                          [d / requested for d in dirs])
            target = next((Path(os.path.normpath(p)) for p in candidates if p.is_file()), None)
            if target is None:
                if match[1].endswith('"'):
                    problems.append(dict(kind='unresolved quoted include', source=str(path), include=name))
                continue  # Unresolved angle headers belong to compiler/toolkit/system.
            if inside(target):
                copy_source(target, dirs)
                if requested.is_absolute():
                    replacement = os.path.relpath(out / staged(target), (out / relative).parent)
                    replacements.append((match.start(2), match.end(2), replacement))
        for start, end, replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]
        # Comments may contain historical paths. They are not build dependencies,
        # but do not publish the machine-specific workspace root in source text.
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if str(ws) in line:
                if line.lstrip().startswith(('//', '*')):
                    lines[i] = line.replace(str(ws), '${HISTORICAL_WORKSPACE}')
                else:
                    problems.append(dict(kind='remaining operational workspace path', source=str(path), line=i + 1))
        text = ''.join(lines)
        before_diagnostic = text
        substitutions = []
        if str(path.relative_to(ws)).endswith('native-ds41/src/ds41/native_bf16_many_v33.cu'):
            identity_guard = re.search(r'if\(!mask \|\| std::strcmp\(mask,"[^"]+"\) \|\|', text)
            if not identity_guard:
                raise RuntimeError('Diagnostic BF16 original identity guard changed')
            substitutions = [
                ('    const char* mask=std::getenv("CUDA_VISIBLE_DEVICES");\n', ''),
                (identity_guard[0],
                 'if(!strata::ds41::publication::metadata_snapshot(1) ||'),
                ('if(cudaGetDevice(&device)!=cudaSuccess || device!=0) return 2;',
                 'if(cudaGetDevice(&device)!=cudaSuccess || device!=0 || !strata::ds41::publication::runtime_snapshot(1)) return 2;')]
        elif str(path.relative_to(ws)).endswith('native-ds41/src/ds41/wo_a_many_v116.cu'):
            identity_guard = re.search(r'if \(!mask \|\| std::strcmp\(mask,"[^"]+"\)', text)
            if not identity_guard:
                raise RuntimeError('Diagnostic WOA original identity guard changed')
            substitutions = [
                ('    const char* mask = std::getenv("CUDA_VISIBLE_DEVICES");\n', ''),
                (identity_guard[0],
                 'if (!strata::ds41::publication::metadata_snapshot(1)'),
                ('if (cudaGetDevice(&device) != cudaSuccess || device != 0) return 2;',
                 'if (cudaGetDevice(&device) != cudaSuccess || device != 0 || !strata::ds41::publication::runtime_snapshot(1)) return 2;')]
        elif path.name == 'headless_resource_v320.inc':
            identity_guard = re.search(r'const char\*pair=std::getenv\("DS41_NATIVE_PAIR_MASK"\),\*mask=.*?"320 exact pair visibility policy BEFORE CUDA"\);', text, re.S)
            if not identity_guard:
                raise RuntimeError('Diagnostic320 original metadata guard changed')
            old = identity_guard[0]
            new = ('const char*pair=std::getenv("DS41_NATIVE_PAIR_MASK"),*stage=std::getenv("DEEPSEEK41_GPU_STAGE");\n'
                   '        fail(pair&&!std::strcmp(pair,"1")&&stage&&!std::strcmp(stage,"0")&&strata::ds41::publication::metadata_snapshot(2),"320 fresh pair snapshot BEFORE CUDA");')
            uuid_lines = [s for s in text.splitlines(keepends=True) if 'const unsigned char uuids[2][16]' in s or 'memcmp(prop.uuid.bytes,uuids[d],16)' in s]
            if len(uuid_lines) != 2:
                raise RuntimeError('Diagnostic320 original identity block changed')
            substitutions = [(old, new), (''.join(uuid_lines),
                '        fail(strata::ds41::publication::runtime_snapshot(2),"320 fresh pair CUDA UUID/order mismatch");\n')]
        elif str(path.relative_to(ws)).endswith('pair-packed-inverse-v141-r2/source/pair_inverse_v141.cu'):
            identity_array = re.search(r'constexpr unsigned char UUIDs\[2\]\[16\]=\{.*?\};\n', text, re.S)
            if not identity_array:
                raise RuntimeError('Byte-inverse141 original identity array changed')
            substitutions = [(identity_array[0], ''),
                ('CUuuid uuid{};if(cuDeviceGetUuid(&uuid,current)!=CUDA_SUCCESS||std::memcmp(uuid.bytes,UUIDs[expected],16))return DS141_DEVICE;',
                 'strata::ds41::publication::DiagnosticSnapshot fresh{};if(!strata::ds41::publication::diagnostic_snapshot(2,fresh))return DS141_DEVICE;\n'
                 ' CUuuid uuid{};if(cuDeviceGetUuid(&uuid,current)!=CUDA_SUCCESS||std::memcmp(uuid.bytes,fresh[expected].data(),16))return DS141_DEVICE;')]
        if substitutions:
            for old, new in substitutions:
                if text.count(old) != 1:
                    raise RuntimeError('Diagnostic-only source pin changed: ' + str(path))
                text = text.replace(old, new, 1)
            helper = out / 'include/strata_ds/fresh_diagnostic_visibility.hpp'
            include = '#include "' + os.path.relpath(helper, (out / relative).parent) + '"\n'
            text = include + text
            delta = ''.join(difflib.unified_diff(before_diagnostic.splitlines(keepends=True), text.splitlines(keepends=True),
                    fromfile=str(relative), tofile=str(relative)))
            delta = ''.join('- <historical identity-only diagnostic guard removed>\n'
                            if line.startswith('-') and ('GPU-' in line or 'uuids[2][16]' in line or 'UUIDs[2][16]' in line or 'prop.pciDomainID' in line or re.search(r'\{0x[0-9a-f]+,0x', line))
                            else line for line in delta.splitlines(keepends=True))
            diagnostic_deltas.append(dict(source=str(relative), original_sha256=digest(before_diagnostic.encode()),
                exported_sha256=digest(text.encode()), expected_visible_count=2 if path.name in ('headless_resource_v320.inc', 'pair_inverse_v141.cu') else 1,
                scope='byte_inverse_device_binding' if path.name == 'pair_inverse_v141.cu' else 'diagnostic_only',
                delta=delta))
        encoded = text.encode()
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        source_records[str(relative)] = dict(original=str(path.relative_to(ws)),
            original_sha256=digest(original), exported_sha256=digest(encoded),
            bytes=len(encoded), absolute_include_rewrites=len(replacements), diagnostic_snapshot_port=bool(substitutions))

    expanded = {}
    for sha, unit in units.items():
        cwd = Path(unit['cwd'])
        argv = expand(unit['argv'], cwd)
        dirs = include_dirs(argv, cwd)
        expanded[sha] = argv
        for source in unit['sources']:
            copy_source(Path(source), dirs)
        for i, flag in enumerate(argv[:-1]):
            if flag == '-include':
                p = Path(argv[i + 1])
                copy_source(p if p.is_absolute() else cwd / p, dirs)

    # Retain notices along every copied source's ancestry, including vendored
    # CUTLASS/exllamav3 leaves. Never copy an entire third-party checkout.
    for record in list(source_records.values()):
        current = (ws / record['original']).parent
        while current != ws and inside(current):
            for name in ('LICENSE', 'LICENSE.txt', 'LICENSE.ggml', 'LICENSE-ggml.txt', 'COPYING', 'NOTICE'):
                notice = current / name
                if notice.is_file() and notice.stat().st_size < 1024 * 1024:
                    destination = out / staged(notice)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    data = notice.read_bytes()
                    destination.write_bytes(data)
                    source_records[str(staged(notice))] = dict(original=str(notice.relative_to(ws)),
                        original_sha256=digest(data), exported_sha256=digest(data), bytes=len(data), notice=True)
            current = current.parent
    helper = out / 'include/strata_ds/fresh_diagnostic_visibility.hpp'
    helper_data = helper.read_bytes()
    source_records[str(helper.relative_to(out))] = dict(original='PUBLICATION_ADDED_FRESH_DIAGNOSTIC_VALIDATOR',
        original_sha256=None, exported_sha256=digest(helper_data), bytes=len(helper_data))
    # The same include can be visited in several TU contexts. Its actual delta
    # and original full source SHA are identical; retain one record per source.
    diagnostic_deltas = list({d['source']: d for d in diagnostic_deltas}.values())
    if len(diagnostic_deltas) != 4:
        raise RuntimeError('Expected exactly three diagnostic and one byte-inverse admission ports')
    (out / 'diagnostic-admission-delta.json').write_text(json.dumps(dict(schema=1, numerical_leaves_changed=False,
        C_ABI_signatures_changed=False, production_native_peer_admission_changed=False,
        files=diagnostic_deltas), indent=2) + '\n')

    object_paths = {}
    for obj in report['objects']:
        if obj['member'] is None:
            object_paths[str(ws / obj['input'])] = obj['sha256']
    outputs = {}
    for sha, unit in units.items():
        suffix = Path(unit['original_object']).suffix
        outputs[sha] = ('@BINARY_ROOT@/support/' + Path(unit['original_object']).name if suffix == '.so'
                        else '@BINARY_ROOT@/objects/u_' + sha[:20] + '.o')
    archives = {}
    for obj in report['objects']:
        if obj['member'] is not None:
            archive = archives.setdefault(obj['input'], dict(
                output='@BINARY_ROOT@/archives/a_' + digest(obj['input'].encode())[:16] + '.a', members=[]))
            pair = dict(name=obj['member'], object=obj['sha256'])
            if pair not in archive['members']:
                archive['members'].append(pair)

    def rewrite(arg, cwd):
        if arg.startswith('-ccbin='):
            return '-ccbin=@CXX@'
        if arg.startswith('-I') and len(arg) > 2:
            return '-I' + rewrite(arg[2:], cwd)
        if arg.startswith('-L') and len(arg) > 2:
            return '-L' + rewrite(arg[2:], cwd)
        if arg.startswith('-Wl,-rpath,'):
            if str(ws) in arg:
                return '-Wl,-rpath,$ORIGIN'
            if '/cuda' in arg:
                return '-Wl,-rpath,@CUDA_LIBRARY_DIR@'
        if arg in ('/usr/bin/clang++-18', 'clang++-18', '/usr/bin/c++', 'c++', 'clang++'):
            return '@CXX@'
        if Path(arg).name == 'nvcc':
            return '@NVCC@'
        if arg.startswith('/usr/local/cuda'):
            if '/stubs' in arg:
                return '@CUDA_LIBRARY_DIR@/stubs'
            if '/include' in arg:
                return '@CUDA_ROOT@/include'
            if Path(arg).suffix in ('.a', '.so'):
                return '@CUDA_LIBRARY_DIR@/' + Path(arg).name
            return '@CUDA_LIBRARY_DIR@'
        if arg.startswith(str(ws)):
            path = Path(arg)
            key = str(path.relative_to(ws))
            if key in archives:
                return archives[key]['output']
            if str(path) in object_paths:
                return outputs[object_paths[str(path)]]
            return '@SOURCE_ROOT@/' + str(staged(path))
        if not arg.startswith('-') and (cwd / arg).exists() and inside(cwd / arg):
            return rewrite(str(cwd / arg), cwd)
        if not arg.startswith('-') and arg.endswith(('.o', '.a', '.so')):
            path = cwd / arg
            if str(path) in object_paths or str(path.relative_to(ws)) in archives:
                return rewrite(str(path), cwd)
        return arg

    manifest_units = {}
    for sha, unit in units.items():
        argv = expanded[sha]
        cwd = Path(unit['cwd'])
        cleaned = []
        i = 0
        while i < len(argv):
            if argv[i] in ('-MD', '-MMD'):
                i += 1
            elif argv[i] in ('-MT', '-MF', '-MQ'):
                i += 2
            elif argv[i] == '-o':
                cleaned += ['-o', outputs[sha]]
                i += 2
            else:
                cleaned.append(rewrite(argv[i], cwd))
                i += 1
        if cleaned[0] == '@NVCC@' and not any(s.startswith('-ccbin=') for s in cleaned):
            cleaned.insert(1, '-ccbin=@CXX@')
        if outputs[sha].endswith('.so'):
            # A support DSO without SONAME records its absolute build filename in
            # the parent DT_NEEDED. Give it a basename without changing symbols.
            if cleaned[0] == '@NVCC@':
                if not any('soname' in s for s in cleaned):
                    cleaned += ['-Xlinker', '-soname', '-Xlinker', Path(outputs[sha]).name]
                cleaned.append('-Xcompiler=-fuse-ld=mold')
            else:
                if not any(s.startswith('-Wl,-soname,') for s in cleaned):
                    cleaned.append('-Wl,-soname,' + Path(outputs[sha]).name)
                if '-fuse-ld=mold' not in cleaned:
                    cleaned.append('-fuse-ld=mold')
        manifest_units[sha] = dict(output=outputs[sha], argv=cleaned,
            historical_compiler=unit['argv'][0], evidence=unit['evidence'],
            sources=[str(staged(Path(p))) for p in unit['sources']],
            dependencies=[str(staged(Path(p))) for p, _ in sorted(traversed)
                          if (out / staged(Path(p))).is_file()] + ['include/strata_ds/fresh_diagnostic_visibility.hpp'])

    engines = {}
    for link in links:
        context = link['label'].split('-')[-1]
        argv = link['argv']
        compiler = next(i for i, s in enumerate(argv) if Path(s).name == 'clang++-18')
        argv = argv[compiler:]
        cwd = Path(link['cwd'])
        result = []
        i = 0
        while i < len(argv):
            if argv[i] == '-o':
                result += ['-o', '@BINARY_ROOT@/' + context + '/libds1017_pair_runtime_v1.so']
                i += 2
            else:
                arg = rewrite(argv[i], cwd)
                if arg == '-Wl,-rpath,$ORIGIN':
                    arg = '-Wl,-rpath,$ORIGIN/../support'
                result.append(arg)
                i += 1
        engines[context] = dict(argv=result, output='@BINARY_ROOT@/' + context + '/libds1017_pair_runtime_v1.so')
    build = dict(schema=1, units=manifest_units, archives=archives, engines=engines,
        flags_policy='Preserved per TU; dependency-output flags removed; compiler/toolkit paths rebound to selected Clang18/CUDA; no fast-math override. Support SONAME basenames and ORIGIN RPATH remove build-path runtime dependencies; all host links use mold.',
        unresolved=report['unresolved'], source_problems=problems)
    (out / 'build-manifest.json').write_text(json.dumps(build, indent=2) + '\n')
    (out / 'source-manifest.json').write_text(json.dumps(dict(files=source_records), indent=2) + '\n')
    print('Exported native source/notice files', len(source_records), 'source problems', len(problems))


def export_lifecycle(ws, out, multi, base, single):
    """Export executable source, never the private binding receipts or profiles.

    Absolute source literals become paths into the exported source tree. The
    historical content/seal/receipt guards remain; no new ready credit is forged.
    A future Root binder must re-qualify and bind the clean build, not old DSOs.
    """
    delivery = multi / 'research/paired-console-delivery-v563'
    cold = multi / 'research/paired-console-server-v565'
    manifest = json.loads((delivery / 'SOURCE_MANIFEST.json').read_text())
    queue = [Path(r['path']) for r in manifest['files'] if r['path'].endswith('.py')
             and '/source/' in r['path'] and not Path(r['path']).name.startswith(('test_', 'prepare_', 'root_bind', 'source_check'))]
    queue.append(delivery / 'serving_runtime_v529.py')
    queue += [p for p in sorted(cold.glob('*.py')) if not p.name.startswith(('test_', 'root_', 'check_'))]
    queue.append(single / 'research/old-complete-runtime-1017-adapter-v1/one_bank_store.py')
    seen = set()
    records = {}
    unresolved = []
    dynamic = []
    while queue:
        file = Path(os.path.normpath(queue.pop()))
        if file in seen:
            continue
        seen.add(file)
        if not file.is_file() or not file.is_relative_to(ws):
            unresolved.append(dict(kind='missing Python source', path=str(file).replace(str(ws), '${HISTORICAL_WORKSPACE}')))
            continue
        raw = file.read_bytes()
        text = raw.decode('utf-8')
        tree = ast.parse(text)
        # Follow direct imports only where the original file/package exists. Do
        # not substitute a same-basename module from an unrelated research path.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.replace('.', '/')
                candidates = [file.parent / (module + '.py')]
                if node.module.startswith(('backend.', 'serve.')):
                    candidates.append(base / 'Strata' / (module + '.py'))
                if node.level:
                    candidates.append(base / 'Strata/backend/deepseek41' / (module + '.py'))
                target = next((p for p in candidates if p.is_file()), None)
                if target:
                    queue.append(target)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    p = file.parent / (name.name.replace('.', '/') + '.py')
                    if p.is_file():
                        queue.append(p)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr in ('import_module', 'spec_from_file_location'):
                dynamic.append(dict(source=str(file.relative_to(ws)), line=node.lineno,
                                    call=ast.get_source_segment(text, node)))
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        edits = []
        needs_helper = False
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.STRING:
                continue
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(value, str):
                continue
            if value.startswith(str(ws)):
                relative = str(Path(value).relative_to(ws))
                replacement = '_ds_publication_source(' + repr(relative) + ')'
                needs_helper = True
                p = Path(value)
                if p.suffix == '.py' and p.is_file():
                    queue.append(p)
            elif value.startswith('/media/'):
                # No model payload or install-specific external model path ships.
                replacement = '_ds_publication_model_dir()'
                needs_helper = True
            elif str(ws) in value:
                replacement = repr(value.replace(str(ws), '${HISTORICAL_WORKSPACE}'))
            else:
                continue
            edits.append((offsets[token.start[0] - 1] + token.start[1],
                          offsets[token.end[0] - 1] + token.end[1], replacement))
        for start, end, replacement in reversed(edits):
            text = text[:start] + replacement + text[end:]
        if needs_helper:
            depth = len(file.relative_to(ws).parts) - 1
            helper = ('\ndef _ds_publication_source(relative):\n'
                      ' from pathlib import Path as _PublicationPath\n'
                      f' return str(_PublicationPath(__file__).resolve().parents[{depth}] / relative)\n'
                      '\ndef _ds_publication_model_dir():\n'
                      ' import os as _publication_os\n'
                      ' return _publication_os.environ["STRATA_MODEL_DIR"]\n\n')
            parsed = ast.parse(text)
            insertion = 0
            for node in parsed.body:
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) or \
                        (isinstance(node, ast.ImportFrom) and node.module == '__future__'):
                    insertion = node.end_lineno
                else:
                    break
            lines = text.splitlines(keepends=True)
            lines.insert(insertion, helper)
            text = ''.join(lines)
        # Never export an obsolete fixed hardware allowlist as an executable path.
        fixed = re.findall(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}', text)
        if fixed:
            unresolved.append(dict(kind='fixed hardware identity source refused', source=str(file.relative_to(ws))))
            continue
        text = text.replace(str(ws), '${HISTORICAL_WORKSPACE}')
        ast.parse(text)
        destination = out / 'sources' / file.relative_to(ws)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        records[str(destination.relative_to(out))] = dict(original=str(file.relative_to(ws)),
            original_sha256=digest(raw), exported_sha256=digest(text.encode()),
            path_literal_rewrites=len(edits), source_helper=needs_helper)
    report = dict(schema=1, status='SOURCE_ONLY_ROOT_REBIND_REQUIRED_NOT_A_RUNNABLE_DELIVERY',
        factory_revision=563, cold_revision=565, files=records, unresolved=unresolved,
        dynamic_imports=dynamic,
        not_exported=['private DELIVERY_BINDINGS/SEALED/DEPENDENCY_PINS/runtime profiles',
                      'model manifests/payloads, API secrets, logs, benchmark prompts, old DSOs'],
        blockers=['Fresh clean-build source/content bindings and native idle/API qualification required.',
                  'Historical guards reference many comparison receipts; port to common publication binder without deleting numerical/lifetime guards.',
                  'Dynamic private-source loaders must bind the exported module graph, not legacy machine paths.',
                  'Cold565 remains pending independent actual cold-start qualification.'])
    (out / 'lifecycle-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Exported lifecycle Python files', len(records), 'explicit source refusals', len(unresolved))


if __name__ == '__main__':
    main()
