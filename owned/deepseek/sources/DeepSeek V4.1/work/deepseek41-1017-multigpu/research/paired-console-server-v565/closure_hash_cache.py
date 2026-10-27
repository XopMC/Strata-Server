"""Source-only proposal: process-local file digests, never bank/protocol PASS caching.

No I/O on import. Every lookup opens/checks a real FD and refreshes path/link
metadata. The caller must still compare admission evidence and perform ALL bank,
owner, epoch, RO-VMA, close/drain, source-policy and fresh-fork guards.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re
import stat

INSTALLED_ROOTS = (
    Path('/usr/local/cuda-13.4/targets/x86_64-linux/lib'),
    Path('/usr/lib/x86_64-linux-gnu'),
    Path('/lib/x86_64-linux-gnu'),
)


class VerificationError(ValueError):
    pass


def need(value, message):
    if not value:
        raise VerificationError(message)


@dataclass(frozen=True)
class Signature:
    device: int
    inode: int
    bytes: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    nlink: int

    @classmethod
    def of(cls, value):
        return cls(value.st_dev, value.st_ino, value.st_size,
                   value.st_mtime_ns, value.st_ctime_ns,
                   value.st_mode, value.st_nlink)


@dataclass(frozen=True)
class VerifiedFile:
    evidence: dict
    cache_hit: bool


class FileHashCache:
    """Bounded digest LRU; no retained FDs, file bytes, parsed source or bank data.

    installed_roots is a TRUSTED construction-time policy, not a user input or
    auto-discovered exception. Production integration must retain INSTALLED_ROOTS.
    Tests may supply a private temporary directory. Instances are single-thread
    only, matching the CPU parent's fork contract. A fork clears copied entries
    before the next lookup; this does not authorize use in a CUDA child.
    """
    def __init__(self, *, max_entries=4096, installed_roots=INSTALLED_ROOTS):
        need(type(max_entries) is int and 1 <= max_entries <= 65536,
             'bounded cache capacity')
        self.max_entries = max_entries
        self.installed_roots = tuple(Path(p) for p in installed_roots)
        need(self.installed_roots and all(p.is_absolute() for p in self.installed_roots),
             'explicit absolute installed-library roots')
        self._entries = OrderedDict()
        self._pid = os.getpid()
        self.hash_reads = 0
        self.cache_hits = 0

    def clear(self):
        self._entries.clear()

    def _process(self):
        if self._pid != os.getpid():
            self.clear()
            self._pid = os.getpid()
            self.hash_reads = self.cache_hits = 0

    def _allowed(self, path):
        return any(path.is_relative_to(root) for root in self.installed_roots)

    @staticmethod
    def _links(path):
        # Collect every symlink component plus the components of its targets.
        # resolve() supplies final path semantics; this only records fresh link
        # identities. Bounded traversal handles leaf SONAME and ancestor aliases.
        queue = [path]
        visited = set()
        evidence = {}
        while queue:
            current = queue.pop()
            for component in reversed((current, *current.parents)):
                key = str(component)
                if key in visited:
                    continue
                visited.add(key)
                value = component.lstat()
                if stat.S_ISLNK(value.st_mode):
                    need(len(evidence) < 16, 'installed link chain exceeds16')
                    target = os.readlink(component)
                    evidence[key] = dict(path=key, link=target,
                                         stat=asdict(Signature.of(value)))
                    redirected = Path(target) if os.path.isabs(target) else component.parent / target
                    queue.append(redirected)
        return [evidence[key] for key in sorted(evidence)]

    def _snapshot(self, path, installed):
        if installed:
            need(self._allowed(path), 'explicit installed dependency outside allowed roots')
            resolved = path.resolve(strict=True)
            need(self._allowed(resolved), 'installed dependency target escapes allowed roots')
            links = self._links(path)
        else:
            resolved = path.resolve(strict=True)
            need(resolved == path and not path.is_symlink(),
                 'project pin must be canonical with no symlink components')
            links = []
        value = resolved.lstat()
        need(stat.S_ISREG(value.st_mode), 'pin target is not a regular file')
        return resolved, Signature.of(value), links

    @staticmethod
    def _row(row):
        need(isinstance(row, dict) and {'path', 'sha256', 'bytes'} <= set(row),
             'explicit path/SHA256/bytes row required')
        need(isinstance(row['path'], str), 'path must be text')
        path = Path(row['path'])
        need(path.is_absolute() and '..' not in path.parts, 'absolute canonical pin path required')
        need(type(row['bytes']) is int and row['bytes'] >= 0, 'literal nonnegative pinned bytes')
        need(isinstance(row['sha256'], str) and
             re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None, 'literal SHA256 required')
        return path

    def verify_project(self, row, *, expected_evidence=None):
        return self._verify(row, installed=False, expected_evidence=expected_evidence)

    def verify_installed_dependency(self, row, *, expected_evidence=None):
        # Only caller's explicit DEPENDENCY_PINS rows may reach this method.
        return self._verify(row, installed=True, expected_evidence=expected_evidence)

    def _verify(self, row, *, installed, expected_evidence):
        self._process()
        path = self._row(row)
        resolved, before, links = self._snapshot(path, installed)
        need(before.bytes == row['bytes'], 'pinned size changed')
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open(resolved, flags)
        signature = None
        try:
            with os.fdopen(fd, 'rb') as stream:
                fd = None
                signature = Signature.of(os.fstat(stream.fileno()))
                need(signature == before and stat.S_ISREG(signature.mode),
                     'file changed between path check and open')
                hit = signature in self._entries
                if hit:
                    digest = self._entries[signature]
                else:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    self.hash_reads += 1
                after_fd = Signature.of(os.fstat(stream.fileno()))
                resolved_after, after_path, links_after = self._snapshot(path, installed)
                need(signature == after_fd == after_path and resolved_after == resolved
                     and links_after == links, 'file/link changed while verifying')
                need(digest == row['sha256'], 'pinned content SHA256 mismatch')
                evidence = dict(path=str(path), resolved_target=str(resolved),
                                bytes=signature.bytes, sha256=digest,
                                link_chain=links, target_stat=asdict(signature))
                if expected_evidence is not None:
                    need(evidence == expected_evidence, 'admitted path/link/stat evidence changed')
                self._entries[signature] = digest
                self._entries.move_to_end(signature)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
                if hit:
                    self.cache_hits += 1
                return VerifiedFile(evidence, hit)
        except BaseException:
            if signature is not None:
                self._entries.pop(signature, None)
            raise
        finally:
            if fd is not None:
                os.close(fd)
