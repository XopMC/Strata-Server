"""Bounded row reads; the two large Engram tables are never materialized."""
import json
import os
from pathlib import Path, PurePosixPath
import struct
import hashlib
import time

import numpy as np

from .native import bf16, e4m3fn, e8m0, fp8_block_rows, fp8_row_groups, mxfp4

REVISION = "23da584e06e57ec43c896d286b44b822c9bb0ad6"
HEADER_LIMIT = 16 * 1024 * 1024


def available_memory_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise ValueError('Cannot determine host memory reserve')


class NativeStore:
    def __init__(self, source, verified_status, read_limit_bytes=64 * 1024 * 1024):
        self.source = Path(source).resolve()
        status = json.loads(Path(verified_status).read_text())
        if status.get("phase") != "complete" or status.get("revision") != REVISION:
            raise ValueError("Native weights are unavailable until the pinned download is verified")
        if Path(status["destination"]).resolve() != self.source:
            raise ValueError("Verification is for a different source directory")
        self.verified = {f["name"]: f for f in status["verified_files"]}
        if len([n for n in self.verified if n.endswith(".safetensors")]) != 48:
            raise ValueError("Verification does not cover all 48 weight shards")
        self.index = json.loads((self.source / "model.safetensors.index.json").read_text())["weight_map"]
        self.read_limit_bytes = read_limit_bytes
        self._shards = {}
        self.payload_read_calls = 0
        self.payload_bytes_read = 0
        self.ram_shards = {}
        self.ram_preload_report = None
        self.ram_read_calls = self.ram_bytes_read = 0

    def _shard(self, filename):
        if filename in self._shards:
            return self._shards[filename]
        relative = PurePosixPath(filename)
        path = self.source / filename
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
            raise ValueError("Unsafe shard path")
        if filename not in self.verified or path.stat().st_size != self.verified[filename]["size"]:
            raise ValueError("Source shard changed after verification")
        descriptor = os.open(path, os.O_RDONLY)
        try:
            raw = os.pread(descriptor, 8, 0)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors length")
            length = struct.unpack("<Q", raw)[0]
            if not 2 <= length <= HEADER_LIMIT:
                raise ValueError("Invalid safetensors header length")
            raw = os.pread(descriptor, length, 8)
            if len(raw) != length:
                raise ValueError("Truncated safetensors header")
            header = json.loads(raw)
            result = (descriptor, header, length + 8, path.stat().st_size)
            self._shards[filename] = result
            return result
        except Exception:
            os.close(descriptor)
            raise

    def info(self, name):
        filename = self.index[name]
        descriptor, header, payload, size = self._shard(filename)
        tensor = header[name]
        start, end = tensor["data_offsets"]
        if start < 0 or end < start or payload + end > size:
            raise ValueError("Tensor payload range is invalid")
        return {**tensor, "shard": filename, "offset": payload + start, "bytes": end - start}

    def raw(self, name):
        info = self.info(name)
        if info["bytes"] > self.read_limit_bytes:
            raise ValueError("Tensor exceeds bounded read limit; use row gathering")
        descriptor = self._shard(info["shard"])[0]
        raw = self._read(descriptor,info['bytes'],info['offset'])
        if len(raw) != info["bytes"]:
            raise ValueError("Truncated tensor read")
        return raw

    def raw_view(self,name):
        """Bounded immutable packed view for synchronous CPU kernels.

        A RAM view retains its owned array through memoryview/NumPy. It must
        never be used as writable scratch. Disk fallback owns ordinary bytes.
        Accounting counts consumed source bytes even without a host copy.
        """
        info = self.info(name)
        if info['bytes'] > self.read_limit_bytes:
            raise ValueError('Tensor exceeds bounded packed view limit')
        descriptor = self._shard(info['shard'])[0]
        resident = self.ram_shards.get(descriptor)
        if resident is None:
            return memoryview(self.raw(name))
        offset,length = info['offset'],info['bytes']
        result = memoryview(resident)[offset:offset+length].toreadonly()
        if result.nbytes != length:
            raise ValueError('Truncated resident packed view')
        self.payload_read_calls += 1
        self.payload_bytes_read += length
        self.ram_read_calls += 1
        self.ram_bytes_read += length
        return result

    def _read(self,descriptor,length,offset):
        resident = getattr(self,'ram_shards',{}).get(descriptor)
        if resident is None:
            raw = os.pread(descriptor,length,offset)
        else:
            raw = resident[offset:offset+length].tobytes()
            self.ram_read_calls += 1
            self.ram_bytes_read += len(raw)
        self.payload_read_calls += 1
        self.payload_bytes_read += len(raw)
        return raw

    def row_span_view(self,name,start,end):
        """One bounded consecutive row span, immutable and zero-copy in RAM."""
        info=self.info(name);shape=info['shape']
        if (len(shape)!=2 or shape[0]<1 or info['bytes']%shape[0] or
                type(start) is not int or type(end) is not int or not 0<=start<=end<=shape[0]):
            raise ValueError('Invalid consecutive packed row span')
        row_bytes=info['bytes']//shape[0];length=(end-start)*row_bytes
        if length>self.read_limit_bytes:raise ValueError('Packed row span exceeds bounded read limit')
        if not length:return memoryview(b'')
        descriptor=self._shard(info['shard'])[0];offset=info['offset']+start*row_bytes
        resident=getattr(self,'ram_shards',{}).get(descriptor)
        if resident is None:
            result=memoryview(self._read(descriptor,length,offset))
        else:
            result=memoryview(resident)[offset:offset+length].toreadonly()
            self.payload_read_calls+=1;self.payload_bytes_read+=length
            self.ram_read_calls+=1;self.ram_bytes_read+=length
        if result.nbytes!=length:raise ValueError('Truncated consecutive packed row span')
        return result

    def preload_packed_ram(self,*,budget_bytes,reserve_bytes=80*1024**3,on_progress=None):
        """Own exact packed shard bytes in RAM, without dequantization/disk copies.

        Admission and memory checks precede allocation. Each copied shard must
        pass its pinned SHA256 before the RAM map is published. Failure drops
        every new buffer; ordinary disk reads remain available. NUMA placement
        follows the launcher's memory policy and must be measured separately.
        """
        import importlib.util
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('deepseek41_ram_admission',root/'tools/run_guarded.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        guard.admit('cpu')
        if (type(budget_bytes) is not int or budget_bytes < 1
                or type(reserve_bytes) is not int or reserve_bytes < 1):
            raise ValueError('Resident packed RAM budgets must be positive integer bytes')
        if self.ram_shards:
            raise ValueError('Packed weights are already resident')
        files = sorted(n for n in self.verified if n.endswith('.safetensors'))
        total = sum(self.verified[n]['size'] for n in files)
        if len(files) != 48 or total > budget_bytes:
            raise MemoryError('All48 packed shards must fit the declared RAM budget')
        scratch = min(total,self.read_limit_bytes)
        if available_memory_bytes() < total+reserve_bytes+scratch:
            raise MemoryError('Packed RAM load would violate the host memory reserve')
        buffers,proof = {},[]
        storage = view = target = None
        copied,started,last_check = 0,time.perf_counter(),0.0
        try:
            for number,name in enumerate(files):
                descriptor,_,_,size = self._shard(name)
                if size != self.verified[name]['size']:
                    raise ValueError('Source size changed before packed RAM load')
                storage = np.empty(size,dtype=np.uint8)
                view = memoryview(storage)
                digest = hashlib.sha256()
                offset = 0
                while offset < size:
                    stop = min(offset+self.read_limit_bytes,size)
                    target = view[offset:stop]
                    received = 0
                    while received < len(target):
                        read = os.preadv(descriptor,[target[received:]],offset+received)
                        if read == 0:
                            raise ValueError('Truncated source during packed RAM load')
                        received += read
                    digest.update(target)
                    copied += len(target)
                    offset = stop
                    now = time.perf_counter()
                    if now-last_check >= 1:
                        last_check = now
                        free = available_memory_bytes()
                        if free < reserve_bytes:
                            raise MemoryError('Host memory reserve exhausted during packed RAM load')
                        if on_progress is not None:
                            on_progress({'phase':'packed_ram_copy','bytes_copied':copied,'total_bytes':total,
                                'shard':number+1,'shards':48,'available_memory_bytes':free,
                                'elapsed_seconds':now-started})
                if digest.hexdigest() != self.verified[name]['digest']:
                    raise ValueError('Packed RAM SHA256 differs from the pinned source: '+name)
                buffers[descriptor] = storage
                proof.append({'name':name,'bytes':size,'sha256':digest.hexdigest(),
                              'buffer_address':int(storage.ctypes.data)})
                target.release()
                view.release()
                storage = view = target = None
            self.ram_shards = buffers
            self.ram_preload_report = {'phase':'all48_packed_shards_sha256_verified_in_ram','source_revision':REVISION,
                'resident_bytes':total,'budget_bytes':budget_bytes,'reserve_bytes':reserve_bytes,
                'copy_seconds':time.perf_counter()-started,'shards':proof,
                'dequantized':False,'disk_copy_created':False,'numa_placement_verified':False}
            return self.ram_preload_report
        except BaseException:
            if target is not None:
                target.release()
            if view is not None:
                view.release()
            storage = view = target = None
            buffers.clear()
            raise

    def rows(self, name, indices):
        info = self.info(name)
        shape = info["shape"]
        if len(shape) != 2 or shape[0] == 0 or info["bytes"] % shape[0]:
            raise ValueError("Row gathering requires a regular nonempty matrix")
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= shape[0]):
            raise ValueError("Row index outside tensor bounds")
        row_bytes = info["bytes"] // shape[0]
        if len(indices) * row_bytes > self.read_limit_bytes:
            raise ValueError("Requested rows exceed bounded read limit")
        descriptor = self._shard(info["shard"])[0]
        # Dense row tiles are contiguous. One bounded pread per run avoids
        # thousands of tiny FUSE/NTFS reads without reordering sparse gathers.
        boundaries = [0,*list(np.flatnonzero(np.diff(indices) != 1)+1),len(indices)]
        values = []
        for start,stop in zip(boundaries,boundaries[1:]):
            if start == stop:
                continue
            length = (stop-start)*row_bytes
            raw = self._read(descriptor,length,info['offset']+int(indices[start])*row_bytes)
            if len(raw) != length:
                raise ValueError('Truncated row read')
            values.append(raw)
        return b"".join(values)

    def float_rows(self, name, indices):
        info = self.info(name)
        indices = np.asarray(indices, dtype=np.int64)
        raw = self.rows(name, indices)
        shape = (len(indices), info["shape"][1])
        if info["dtype"] == "BF16":
            return bf16(raw).reshape(shape)
        if info["dtype"] == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape)
        codes = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        if info["dtype"] == "F8_E8M0":
            return e8m0(codes)
        scale_name = name.rsplit(".", 1)[0] + ".scale"
        if info["dtype"] == "I8" and ".experts." in name:
            scale_info = self.info(scale_name)
            scales = np.frombuffer(self.rows(scale_name, indices), dtype=np.uint8)
            scales = scales.reshape(len(indices), scale_info["shape"][1])
            return mxfp4(codes, scales)
        if info["dtype"] == "F8_E4M3":
            if scale_name not in self.index:
                return e4m3fn(codes)
            scale_info = self.info(scale_name)
            if ".engram.embed." in name:
                scales = np.frombuffer(self.rows(scale_name, indices), dtype=np.uint8)
                return fp8_row_groups(codes, scales.reshape(len(indices), scale_info["shape"][1]))
            scales = np.frombuffer(self.raw(scale_name), dtype=np.uint8).reshape(scale_info["shape"])
            return fp8_block_rows(codes, scales, indices)
        raise ValueError(f"Unsupported native tensor representation: {info['dtype']} {name}")

    def close(self):
        self.ram_shards.clear()
        for descriptor, *_ in self._shards.values():
            os.close(descriptor)
        self._shards.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
