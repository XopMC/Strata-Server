"""Bounded packed weight/scale residency, separate from KV and Engram state.

No device is selected or initialized here. The admitted runtime supplies a
loader for already verified packed tiles. Leases protect live users; async
streams must additionally record their tensor lifetimes in the CUDA adapter.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
import torch


@dataclass
class _Entry:
    tensors: tuple
    bytes: int
    leases: int = 0


class PackedTileCache:
    def __init__(self,budget_bytes,device):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError('Packed cache budget must be nonnegative bytes')
        self.budget_bytes,self.device = budget_bytes,torch.device(device)
        self.entries = OrderedDict()
        self.lock = RLock()
        self.in_loader = False
        self.bytes,self.peak_bytes = 0,0
        self.hits,self.misses,self.evictions,self.bypasses,self.load_failures = 0,0,0,0,0
        self.loaded_bytes,self.saved_load_bytes = 0,0

    def _load(self,loader,expected_bytes):
        try:
            self.in_loader = True
            try:
                tensors = loader()
            finally:
                self.in_loader = False
            if (not isinstance(tensors,tuple) or len(tensors) != 2
                    or any(not isinstance(t,torch.Tensor) or t.device != self.device or not t.is_contiguous() for t in tensors)):
                raise ValueError('Packed residency loader must return contiguous weight/scale tensors on its device')
            actual = sum(t.numel()*t.element_size() for t in tensors)
            if actual != expected_bytes:
                raise ValueError('Packed residency size differs from its declared budget')
        except Exception:
            self.load_failures += 1
            raise
        self.loaded_bytes += expected_bytes
        return tensors

    @contextmanager
    def lease(self,key,expected_bytes,loader):
        if type(expected_bytes) is not int or expected_bytes < 1:
            raise ValueError('Packed tile needs a positive known byte size')
        hash(key)
        cached = False
        with self.lock:
            if self.in_loader:
                raise ValueError('Packed loader cannot recursively lease its cache')
            entry = self.entries.get(key)
            if entry is not None:
                if entry.bytes != expected_bytes:
                    raise ValueError('Packed cache key was reused with a different tensor size')
                entry.leases += 1
                self.entries.move_to_end(key)
                self.hits += 1
                self.saved_load_bytes += expected_bytes
                tensors,cached = entry.tensors,True
            else:
                self.misses += 1
                if expected_bytes > self.budget_bytes:
                    self.bypasses += 1
                    tensors = self._load(loader,expected_bytes)
                else:
                    # Decide capacity before loading. Live leased tiles cannot
                    # be evicted to make an optimistic allocation fit.
                    needed = self.bytes+expected_bytes-self.budget_bytes
                    victims,freed = [],0
                    for candidate,value in self.entries.items():
                        if needed <= freed:
                            break
                        if not value.leases:
                            victims.append(candidate)
                            freed += value.bytes
                    if needed > freed:
                        raise MemoryError('Packed cache budget is occupied by live leases')
                    for candidate in victims:
                        victim = self.entries.pop(candidate)
                        self.bytes -= victim.bytes
                        self.evictions += 1
                        del victim
                    tensors = self._load(loader,expected_bytes)
                    self.entries[key] = _Entry(tensors,expected_bytes,1)
                    self.bytes += expected_bytes
                    self.peak_bytes = max(self.peak_bytes,self.bytes)
                    cached = True
        try:
            yield tensors
        finally:
            if cached:
                with self.lock:
                    self.entries[key].leases -= 1

    def contains(self,key,expected_bytes=None):
        """Inspect residency without loading, leasing or changing LRU order."""
        with self.lock:
            entry = self.entries.get(key)
            return entry is not None and (expected_bytes is None or entry.bytes == expected_bytes)

    def clear(self):
        with self.lock:
            if any(entry.leases for entry in self.entries.values()):
                raise ValueError('Cannot clear packed cache with live leases')
            self.entries.clear()
            self.bytes = 0

    def statistics(self):
        with self.lock:
            return {'budget_bytes':self.budget_bytes,'resident_bytes':self.bytes,'peak_resident_bytes':self.peak_bytes,
                    'entries':len(self.entries),'live_leases':sum(e.leases for e in self.entries.values()),
                    'hits':self.hits,'misses':self.misses,'evictions':self.evictions,'bypasses':self.bypasses,
                    'load_failures':self.load_failures,'loaded_bytes':self.loaded_bytes,
                    'saved_load_bytes':self.saved_load_bytes,
                    'accounting':'packed tensor payload only; CUDA allocator/scratch measured separately',
                    'policy':'LRU packed weight+scale pairs; does not contain KV/Engram/activations'}
