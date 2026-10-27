#!/usr/bin/env python3
"""Build runner for CMake; never used by the source exporter.

Every compiler invocation is an argv array. No shell expansion. It consumes only
exported sources, newly built objects and the installed toolchain/toolkit.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--cuda-root', required=True)
    ap.add_argument('--cuda-library-dir', required=True)
    ap.add_argument('--cxx', required=True)
    ap.add_argument('--nvcc', required=True)
    ap.add_argument('--ar', required=True)
    ap.add_argument('kind', choices=('unit', 'archive', 'engine'))
    ap.add_argument('key')
    args = ap.parse_args()
    manifest = json.loads((args.source / 'build-manifest.json').read_text())
    if manifest['unresolved'] or manifest['source_problems']:
        raise SystemExit('Source recovery has unresolved inputs; inspect build-manifest.json')
    substitutions = {'@SOURCE_ROOT@': str(args.source.resolve()),
        '@BINARY_ROOT@': str(args.binary.resolve()), '@CUDA_ROOT@': args.cuda_root,
        '@CUDA_LIBRARY_DIR@': args.cuda_library_dir, '@CXX@': args.cxx,
        '@NVCC@': args.nvcc}

    def expand(value):
        for old, new in substitutions.items():
            value = value.replace(old, new)
        if '@' in value:
            raise ValueError('Unexpanded build token: ' + value)
        return value

    args.binary.mkdir(parents=True, exist_ok=True)
    if args.kind == 'archive':
        archive = manifest['archives'][args.key]
        output = Path(expand(archive['output']))
        output.parent.mkdir(parents=True, exist_ok=True)
        member_dir = output.with_suffix('.members')
        member_dir.mkdir(parents=True, exist_ok=True)
        members = []
        for member in archive['members']:
            source = Path(expand(manifest['units'][member['object']]['output']))
            target = member_dir / member['name']
            if Path(member['name']).name != member['name']:
                raise ValueError('Invalid member name')
            shutil.copyfile(source, target)
            members.append(str(target))
        output.unlink(missing_ok=True)
        subprocess.run([args.ar, 'qc', str(output), *members], check=True)
        subprocess.run([args.ar, 's', str(output)], check=True)
        return
    node = manifest['units'][args.key] if args.kind == 'unit' else manifest['engines'][args.key]
    output = Path(expand(node['output']))
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [expand(s) for s in node['argv']]
    if args.kind == 'engine' and '-fuse-ld=mold' not in argv:
        raise ValueError('Required mold linker missing')
    cwd = args.binary / 'working'
    cwd.mkdir(parents=True, exist_ok=True)
    subprocess.run(argv, cwd=cwd, check=True)


if __name__ == '__main__':
    main()
