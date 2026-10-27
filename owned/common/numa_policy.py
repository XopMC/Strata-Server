"""Resolve legacy all-socket interleave defaults from fresh Linux NUMA scope.

CPU-only, read-only discovery: no placement, migration or system policy changes.
Only the exact compatibility tokens --interleave=0,1 and --interleave=all
are rewritten. Other explicit policies are caller-owned and remain unchanged.
"""
from pathlib import Path
import re
from typing import Sequence

class NumaPolicyError(RuntimeError):
    """Fresh topology cannot safely admit the legacy interleave default."""

def _parse_list(value: str, label: str) -> set[int]:
    value=value.strip()
    if not value:
        raise NumaPolicyError(f'{label} is empty')
    result=set()
    for item in value.split(','):
        if not re.fullmatch(r'\d+(?:-\d+)?',item):
            raise NumaPolicyError(f'{label} has an invalid/ambiguous node list')
        parts=item.split('-'); low=int(parts[0]); high=int(parts[-1])
        if high<low or high>1048575:
            raise NumaPolicyError(f'{label} has an invalid node range')
        nodes=set(range(low,high+1))
        if result.intersection(nodes):
            raise NumaPolicyError(f'{label} repeats a node')
        result.update(nodes)
    return result

def _read(path: Path, label: str) -> str:
    try:return path.read_text()
    except (OSError,UnicodeError) as error:
        raise NumaPolicyError(f'Cannot read {label}: {path}') from error

def discover_interleave_nodes(*, node_root: Path=Path('/sys/devices/system/node'),
                              status_path: Path=Path('/proc/self/status')) -> list[int]:
    """Online memory-bearing nodes allowed to THIS process, discovered each call."""
    online=_parse_list(_read(node_root/'online','online NUMA nodes'),'online NUMA nodes')
    status=_read(status_path,'process memory-node scope')
    matches=re.findall(r'^Mems_allowed_list:\s*([^\n]*)$',status,re.MULTILINE)
    if len(matches)!=1:
        raise NumaPolicyError('Exactly one Mems_allowed_list field is required')
    allowed=_parse_list(matches[0],'Mems_allowed_list')
    candidates=online.intersection(allowed)
    if not candidates:
        raise NumaPolicyError('No online NUMA node intersects process memory-node scope')
    result=[]
    for node in sorted(candidates):
        data=_read(node_root/f'node{node}'/'meminfo',f'node{node} memory inventory')
        entries=re.findall(r'^Node\s+(\d+)\s+MemTotal:\s*(\d+)\s+kB\s*$',data,re.MULTILINE)
        if len(entries)!=1 or int(entries[0][0])!=node:
            raise NumaPolicyError(f'node{node} has missing/ambiguous MemTotal inventory')
        if int(entries[0][1])>0:result.append(node)
    if not result:
        raise NumaPolicyError('No allowed online memory-bearing NUMA nodes')
    # A concurrent topology/cpuset change would invalidate this admission snapshot.
    online_after=_parse_list(_read(node_root/'online','online NUMA nodes'),'online NUMA nodes')
    status_after=_read(status_path,'process memory-node scope')
    after=re.findall(r'^Mems_allowed_list:\s*([^\n]*)$',status_after,re.MULTILINE)
    if online_after!=online or len(after)!=1 or _parse_list(after[0],'Mems_allowed_list')!=allowed:
        raise NumaPolicyError('NUMA topology or allowed memory-node scope changed during discovery')
    return result

def resolve_interleave_args(argv: Sequence[str], *, node_root: Path=Path('/sys/devices/system/node'),
                            status_path: Path=Path('/proc/self/status')) -> list[str]:
    """Expand only exact legacy/all interleave tokens; preserve every other arg.

    Injectable filesystem roots are for read-only topology fixtures. No cached
    topology is retained: each applicable launch resolves its current scope.
    """
    if isinstance(argv,(str,bytes)) or any(not isinstance(x,str) for x in argv):
        raise NumaPolicyError('argv must be a sequence of string arguments')
    values=list(argv)
    targets=[i for i,arg in enumerate(values) if arg in ('--interleave=0,1','--interleave=all')]
    if not targets:return values
    policies=[arg for arg in values if arg=='--interleave' or arg.startswith('--interleave=') or
              arg in ('--membind','--preferred','--preferred-many','--localalloc','--weighted-interleave') or
              arg.startswith(('--membind=','--preferred=','--preferred-many=','--weighted-interleave='))]
    if len(targets)!=1 or len(policies)!=1:
        raise NumaPolicyError('Conflicting/ambiguous memory policies; no interleave rewrite admitted')
    nodes=discover_interleave_nodes(node_root=Path(node_root),status_path=Path(status_path))
    values[targets[0]]='--interleave='+','.join(map(str,nodes))
    return values
