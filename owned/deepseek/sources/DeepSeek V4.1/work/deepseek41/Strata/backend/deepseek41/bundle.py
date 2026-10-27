"""Self-contained packed model container, bounded streaming writer and reader.

All native tensors can be preserved exactly. Our candidate rows store codes
and BF16 scales together; original scale names resolve to internal aliases.
The manifest references only files inside the bundle. Its integrity is not a
quality or performance acceptance. No source removal is performed here.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np

from .native import bf16,e4m3fn,e8m0,mxfp4,fp8_row_groups,fp8_block_rows
from .mixed_quant import QuantizedRows,FORMAT as QUANT_FORMAT

FORMAT = 'deepseek41-packed-bundle-v1'
REVISION = '23da584e06e57ec43c896d286b44b822c9bb0ad6'
REQUIRED_METADATA = ('config.json','runtime-config.json','tokenizer.json','tokenizer_config.json')


def safe_file(root,name):
    if not isinstance(name,str) or Path(name).name != name or name in ('.','..'):
        raise ValueError('Bundle file must be an internal plain filename')
    path = root/name
    if path.is_symlink():
        raise ValueError('Bundle symlinks are refused')
    return path


class BundleWriter:
    def __init__(self,directory,recipe,*,shard_bytes=8*1024**3,reserve_bytes=100*1024**3,
                 resume=False,checkpoint_bytes=8*1024**3):
        self.root = Path(directory).resolve()
        if type(resume) is not bool or type(checkpoint_bytes) is not int or checkpoint_bytes<1:
            raise ValueError('Invalid resumable writer admission')
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise ValueError('Candidate directory must be empty; existing evidence cannot be overwritten')
        if type(shard_bytes) is not int or shard_bytes < 1 or type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError('Invalid candidate shard/disk budgets')
        self.root.mkdir(parents=True,exist_ok=True)
        self.shard_bytes,self.reserve_bytes = shard_bytes,reserve_bytes
        self.recipe = recipe
        self.tensors,self.files = {},{}
        self.stream = self.digest = None
        self.file_name,self.file_bytes,self.number = None,0,0
        self.finished = False
        self.pending=None
        self.checkpoint_bytes,self.bytes_since_checkpoint=checkpoint_bytes,0
        self.identity={'format':FORMAT,'source_revision':REVISION,'shard_bytes':shard_bytes,
            'reserve_bytes':reserve_bytes,'initial_recipe_sha256':hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        if resume:self.restore()
        else:
            self.save_json('candidate.build-owner.json',self.identity)
            self.checkpoint()

    def save_json(self,name,value):
        temporary=safe_file(self.root,name+'.tmp')
        with temporary.open('w') as stream:
            json.dump(value,stream,separators=(',',':'));stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temporary.replace(safe_file(self.root,name))

    def restore(self):
        if (self.root/'model.bundle.json').exists():raise ValueError('Completed candidate cannot be resumed or overwritten')
        if json.loads(safe_file(self.root,'candidate.build-owner.json').read_text())!=self.identity:
            raise ValueError('Interrupted candidate belongs to a different recipe/source/budget')
        state_path=safe_file(self.root,'candidate.build-state.json')
        if state_path.stat().st_size>64*1024**2:raise ValueError('Interrupted candidate state exceeds64MiB')
        state=json.loads(state_path.read_text())
        if state.get('identity')!=self.identity:raise ValueError('Interrupted candidate checkpoint identity differs')
        committed=state['files'];next_number=state['number']
        if type(next_number) is not int or next_number<0:raise ValueError('Invalid interrupted shard counter')
        intervals={}
        for name,record in committed.items():
            safe_file(self.root,name)
            match=re.fullmatch(r'deepseek41-(\d{5})\.bin',name)
            if (type(record['bytes']) is not int or record['bytes']<1 or
                not re.fullmatch(r'[0-9a-f]{64}',record['sha256']) or
                not ((record['kind']=='weights' and match and int(match[1])<next_number) or
                     (record['kind']=='metadata' and name in REQUIRED_METADATA))):
                raise ValueError('Invalid committed candidate file descriptor')
            if record['kind']=='weights':intervals[name]=[]
        entries=list(state['tensors'].values())
        pending=state['pending']
        if pending is not None:
            if (type(pending.get('name')) is not str or pending['name'] in state['tensors'] or
                type(pending.get('offset')) is not int or not 0<=pending['offset']<=pending['descriptor']['bytes']):
                raise ValueError('Invalid interrupted candidate tensor offset')
            entries.append({'segments':pending['segments'],'stored_bytes':pending['offset']})
        for record in entries:
            if record.get('codec')=='internal-row-scales':continue
            cursor=0
            for segment in record['segments']:
                name=segment['file'];start=segment['offset'];size=segment['bytes']
                if (name not in intervals or any(type(v) is not int for v in (start,size,segment['tensor_offset'])) or
                    segment['tensor_offset']!=cursor or start<0 or size<1 or start+size>committed[name]['bytes']):
                    raise ValueError('Interrupted candidate tensor segments changed')
                intervals[name].append((start,start+size));cursor+=size
            if cursor!=record['stored_bytes']:raise ValueError('Interrupted candidate tensor accounting differs')
        for name,regions in intervals.items():
            cursor=0
            for first,last in sorted(regions):
                if first!=cursor:raise ValueError('Interrupted candidate has unowned or overlapping bytes')
                cursor=last
            if cursor!=committed[name]['bytes']:raise ValueError('Interrupted candidate has unowned trailing bytes')
        discard=[]
        for path in self.root.iterdir():
            safe_file(self.root,path.name)
            if not path.is_file():raise ValueError('Unexpected directory in interrupted candidate')
            if path.name in committed:
                record=committed[path.name];digest=hashlib.sha256()
                if path.stat().st_size!=record['bytes']:raise ValueError('Interrupted candidate committed size changed')
                with path.open('rb') as stream:
                    for data in iter(lambda:stream.read(8*1024**2),b''):digest.update(data)
                if digest.hexdigest()!=record['sha256']:raise ValueError('Interrupted candidate committed SHA256 changed')
            elif path.name not in ('candidate.build-owner.json','candidate.build-state.json'):
                match=re.fullmatch(r'deepseek41-(\d{5})\.bin',path.name)
                if ((match and int(match[1])>=next_number) or path.name in REQUIRED_METADATA or
                    path.name in ('candidate.build-state.json.tmp','model.bundle.json.tmp')):discard.append(path)
                else:raise ValueError('Unknown file in interrupted candidate; refuse deletion: '+path.name)
        if set(committed)-{p.name for p in self.root.iterdir()}:raise ValueError('Interrupted candidate committed file missing')
        self.recipe,self.tensors,self.files=state['recipe'],state['tensors'],committed
        self.pending,self.number=state['pending'],next_number
        # Only files owned by this exact incomplete candidate, never source,
        # other models, or unknown entries. All committed files were verified.
        for path in discard:path.unlink()

    def checkpoint(self):
        self.close_shard()
        self.save_json('candidate.build-state.json',{'identity':self.identity,'recipe':self.recipe,
            'tensors':self.tensors,'files':self.files,'pending':self.pending,'number':self.number})
        self.bytes_since_checkpoint=0

    def resume_offset(self,name,descriptor):
        if self.pending is None:return 0
        if self.pending['name']!=name or self.pending['descriptor']!=descriptor:
            raise ValueError('Resume must continue the exact interrupted tensor descriptor')
        return self.pending['offset']

    def reserve(self,additional=0):
        disk = os.statvfs(self.root)
        if disk.f_bavail*disk.f_frsize < self.reserve_bytes+additional:
            raise OSError('Candidate would violate the required disk reserve')

    def close_shard(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            if hasattr(os,'posix_fadvise'):
                os.posix_fadvise(self.stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            self.stream.close()
            self.files[self.file_name] = {'bytes':self.file_bytes,'sha256':self.digest.hexdigest(),'kind':'weights'}
            self.stream = None

    def open_shard(self):
        self.reserve()
        self.file_name = f'deepseek41-{self.number:05}.bin'
        self.number += 1
        self.file_bytes = 0
        self.stream = safe_file(self.root,self.file_name).open('xb',buffering=0)
        self.digest = hashlib.sha256()

    def write_tensor(self,name,descriptor,blocks):
        if self.finished or name in self.tensors or not isinstance(name,str) or not name:
            raise ValueError('Candidate tensor is repeated or writer is finished')
        length = descriptor['bytes']
        if type(length) is not int or length < 1:
            raise ValueError('Candidate tensor must declare its exact positive encoded bytes')
        offset=self.resume_offset(name,descriptor)
        segments=[] if self.pending is None else self.pending['segments']
        self.pending={'name':name,'descriptor':descriptor,'segments':segments,'offset':offset}
        for block in blocks:
            view = memoryview(block).cast('B')
            if view.nbytes > 64*1024**2 or offset+view.nbytes > length:
                raise ValueError('Candidate stream exceeds tensor/block budget')
            consumed = 0
            while consumed < view.nbytes:
                if self.stream is None:
                    self.open_shard()
                chunk_size = min(view.nbytes-consumed,self.shard_bytes-self.file_bytes)
                self.reserve(chunk_size)
                chunk = view[consumed:consumed+chunk_size]
                written = self.stream.write(chunk)
                if written != chunk_size:
                    raise OSError('Incomplete candidate shard write')
                self.digest.update(chunk)
                if segments and segments[-1]['file'] == self.file_name:
                    segments[-1]['bytes'] += chunk_size
                else:
                    segments.append({'file':self.file_name,'offset':self.file_bytes,'tensor_offset':offset,'bytes':chunk_size})
                self.file_bytes += chunk_size
                offset += chunk_size
                consumed += chunk_size
                self.bytes_since_checkpoint+=chunk_size
                if self.file_bytes == self.shard_bytes:
                    self.close_shard()
            # Commit only at an entire input-block boundary. Own-format input
            # blocks contain complete rows, so quantization resumes by row.
            self.pending['offset']=offset
            if self.bytes_since_checkpoint>=self.checkpoint_bytes:self.checkpoint()
        if offset != length:
            raise ValueError('Incomplete tensor stream; candidate cannot be finalized')
        self.tensors[name] = {**descriptor,'segments':segments,'stored_bytes':length}
        self.pending=None

    def add_scale_alias(self,name,parent):
        if name in self.tensors or parent not in self.tensors or self.finished:
            raise ValueError('Candidate scale alias must name a completed internal tensor')
        tensor = self.tensors[parent]
        if tensor.get('codec') != QUANT_FORMAT:
            raise ValueError('Scale alias parent must use our candidate row format')
        rows,width = tensor['shape']
        self.tensors[name] = {'dtype':'BF16','shape':[rows,width//32],'bytes':rows*(width//32)*2,
            'stored_bytes':0,'codec':'internal-row-scales','parent':parent}

    def protect_native_rows(self,name,row_ids,weight_name,scale_name):
        """Attach sparse source-exact Engram rows, stored inside this bundle."""
        if self.finished or name not in self.tensors or self.tensors[name].get('codec')!=QUANT_FORMAT:
            raise ValueError('Protected rows require a completed candidate tensor')
        names=(row_ids,weight_name,scale_name)
        if len(set(names))!=3 or any(n not in self.tensors or not self.tensors[n].get('internal') for n in names):
            raise ValueError('Protected rows must reference three completed internal records')
        self.tensors[name]['protected_native_rows']={'ids':row_ids,'weight':weight_name,'scale':scale_name}

    def add_metadata(self,name,data):
        if name not in REQUIRED_METADATA or name in self.files or self.finished or len(data) > 32*1024**2:
            raise ValueError('Expected one bounded required model metadata file')
        self.reserve(len(data))
        path = safe_file(self.root,name)
        with path.open('xb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        self.files[name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'kind':'metadata'}

    def finish(self,expected_names):
        public={name for name,record in self.tensors.items() if not record.get('internal')}
        referenced={name for record in self.tensors.values() for name in record.get('protected_native_rows',{}).values()}
        internal={name for name,record in self.tensors.items() if record.get('internal')}
        if (self.finished or set(expected_names) != public or referenced != internal
            or not set(REQUIRED_METADATA).issubset(self.files)):
            raise ValueError('Candidate inventory or standalone tokenizer/config files are incomplete')
        self.close_shard()
        manifest = {'format':FORMAT,'source_revision':REVISION,'model_id':'deepseek-v4.1-flash-abliterated',
            'recipe':self.recipe,'files':self.files,'tensors':self.tensors,'quality_gate_passed':False,
            'payload_bytes':sum(tensor['stored_bytes'] for tensor in self.tensors.values())}
        temporary = self.root/'model.bundle.json.tmp'
        temporary.write_text(json.dumps(manifest,separators=(',',':'))+'\n')
        with temporary.open('rb') as source:
            os.fsync(source.fileno())
        temporary.replace(self.root/'model.bundle.json')
        self.finished = True
        for name in ('candidate.build-owner.json','candidate.build-state.json'):
            safe_file(self.root,name).unlink()
        return manifest

    def close(self):
        self.close_shard()


def candidate_row_bytes(quant):
    """Interleave code/scale rows without a second whole-table copy."""
    if not isinstance(quant,QuantizedRows):
        raise ValueError('Expected our validated candidate rows')
    codes = quant.packed
    scales = quant.scales.astype('<u2',copy=False).view(np.uint8).reshape(len(codes),-1)
    return np.concatenate((codes,scales),axis=1).tobytes()


class BundleStore:
    def __init__(self,directory,*,verify=True,read_limit_bytes=64*1024**2):
        if type(read_limit_bytes) is not int or not 1 <= read_limit_bytes <= 64*1024**2:
            raise ValueError('Candidate read budget must be1byte–64MiB')
        self.source = Path(directory).resolve()
        path = safe_file(self.source,'model.bundle.json')
        if path.stat().st_size > 64*1024**2:
            raise ValueError('Candidate manifest exceeds64MiB')
        self.manifest = json.loads(path.read_text())
        if self.manifest.get('format') != FORMAT or self.manifest.get('source_revision') != REVISION:
            raise ValueError('Unexpected candidate format/source revision')
        self.index = self.manifest['tensors']
        self.read_limit_bytes = read_limit_bytes
        self.descriptors,self.ram_shards,self.protected_ids = {},{},{}
        self.payload_read_calls = self.payload_bytes_read = self.ram_read_calls = self.ram_bytes_read = 0
        self.ram_preload_report = None
        self.verified = {}
        try:
            for name,record in self.manifest['files'].items():
                file = safe_file(self.source,name)
                if file.stat().st_size != record['bytes']:
                    raise ValueError('Candidate file size changed: '+name)
                if verify:
                    digest = hashlib.sha256()
                    with file.open('rb') as stream:
                        for chunk in iter(lambda:stream.read(self.read_limit_bytes),b''):
                            digest.update(chunk)
                    if digest.hexdigest() != record['sha256']:
                        raise ValueError('Candidate SHA256 mismatch: '+name)
                self.verified[name] = {'name':name,'size':record['bytes'],'digest':record['sha256'],'kind':record['kind']}
                if record['kind'] == 'weights':
                    self.descriptors[name] = os.open(file,os.O_RDONLY)
            if not set(REQUIRED_METADATA).issubset(self.verified):
                raise ValueError('Candidate lacks standalone model metadata')
            intervals = {name:[] for name in self.descriptors}
            for name,record in self.index.items():
                if record.get('codec') == 'internal-row-scales':
                    if record['parent'] not in self.index or self.index[record['parent']].get('codec') != QUANT_FORMAT:
                        raise ValueError('Scale alias escapes its candidate tensor')
                    continue
                offset = 0
                for segment in record['segments']:
                    file = self.verified.get(segment['file'])
                    if (file is None or file['kind'] != 'weights' or segment['tensor_offset'] != offset
                        or segment['offset'] < 0 or segment['bytes'] < 1 or segment['offset']+segment['bytes'] > file['size']):
                        raise ValueError('Invalid candidate tensor segment')
                    offset += segment['bytes']
                    intervals[segment['file']].append((segment['offset'],segment['offset']+segment['bytes']))
                if offset != record['bytes'] or record['stored_bytes'] != offset:
                    raise ValueError('Incomplete candidate tensor segment inventory')
            for file,regions in intervals.items():
                cursor = 0
                for first,last in sorted(regions):
                    if first != cursor:
                        raise ValueError('Candidate file has unowned or overlapping payload bytes')
                    cursor = last
                if cursor != self.verified[file]['size']:
                    raise ValueError('Candidate file has unowned trailing payload bytes')
            if sum(record['stored_bytes'] for record in self.index.values()) != self.manifest['payload_bytes']:
                raise ValueError('Candidate payload accounting mismatch')
            referenced=set()
            for name,record in self.index.items():
                protection=record.get('protected_native_rows')
                if not protection:continue
                if record.get('codec')!=QUANT_FORMAT or '.engram.embed.weight' not in name or set(protection)!={'ids','weight','scale'}:
                    raise ValueError('Only own-format Engram tables admit sparse native protection')
                ids,weight,scale=(self.index.get(protection[k],{}) for k in ('ids','weight','scale'))
                if (any(not child.get('internal') for child in (ids,weight,scale)) or ids.get('dtype')!='I64'
                    or len(ids.get('shape',[]))!=1 or ids['shape'][0]<1 or ids['bytes']!=ids['shape'][0]*8
                    or ids['bytes']>64*1024**2 or weight.get('dtype')!='F8_E4M3' or scale.get('dtype')!='F8_E8M0'
                    or weight.get('shape')!=[ids['shape'][0],record['shape'][1]]
                    or scale.get('shape')!=[ids['shape'][0],record['shape'][1]//32]
                    or weight.get('native_scale_name')!=protection['scale']):
                    raise ValueError('Protected native Engram rows/scales/IDs have inconsistent shapes')
                referenced.update(protection.values())
            if referenced!={name for name,record in self.index.items() if record.get('internal')}:
                raise ValueError('Bundle contains unreferenced internal weight payload')
        except BaseException:
            self.close()
            raise

    def info(self,name):
        return self.index[name]

    def _range(self,name,offset,length):
        record = self.info(name)
        if offset < 0 or length < 0 or length > self.read_limit_bytes or offset+length > record['bytes']:
            raise ValueError('Candidate read exceeds bounded tensor range')
        parts = []
        for segment in record['segments']:
            first = max(offset,segment['tensor_offset'])
            last = min(offset+length,segment['tensor_offset']+segment['bytes'])
            if first >= last:
                continue
            file,inside,count = segment['file'],segment['offset']+first-segment['tensor_offset'],last-first
            resident = self.ram_shards.get(file)
            if resident is None:
                raw = os.pread(self.descriptors[file],count,inside)
            else:
                raw = resident[inside:inside+count].tobytes()
                self.ram_read_calls += 1
                self.ram_bytes_read += len(raw)
            if len(raw) != count:
                raise ValueError('Truncated candidate tensor read')
            parts.append(raw)
            self.payload_read_calls += 1
            self.payload_bytes_read += count
        return b''.join(parts)

    def rows(self,name,indices):
        record = self.info(name)
        shape = record['shape']
        values = np.asarray(indices,dtype=np.int64)
        if len(shape) != 2 or values.ndim != 1 or np.any(values < 0) or np.any(values >= shape[0]):
            raise ValueError('Candidate row index outside tensor')
        row_bytes = record['bytes']//shape[0]
        if len(values)*row_bytes > self.read_limit_bytes:
            raise ValueError('Candidate rows exceed bounded read budget')
        if record.get('codec') == 'internal-row-scales':
            parent = self.info(record['parent'])
            raw = np.frombuffer(self.rows(record['parent'],values),dtype=np.uint8).reshape(len(values),parent['bytes']//parent['shape'][0])
            return raw[:,parent['shape'][1]*parent['bits']//8:].copy().tobytes()
        boundaries = [0,*list(np.flatnonzero(np.diff(values) != 1)+1),len(values)]
        return b''.join(self._range(name,int(values[start])*row_bytes,(end-start)*row_bytes)
            for start,end in zip(boundaries,boundaries[1:]) if start != end)

    def raw(self,name):
        record = self.info(name)
        if record.get('codec') == 'internal-row-scales':
            if record['bytes'] > self.read_limit_bytes:
                raise ValueError('Candidate scale alias exceeds bounded read limit')
            return self.rows(name,np.arange(record['shape'][0]))
        return self._range(name,0,record['bytes'])

    def raw_view(self,name):
        record = self.info(name)
        if record.get('codec') == 'internal-row-scales' or len(record['segments']) != 1 or record['bytes'] > self.read_limit_bytes:
            return memoryview(self.raw(name))
        segment = record['segments'][0]
        resident = self.ram_shards.get(segment['file'])
        if resident is None:
            return memoryview(self.raw(name))
        self.payload_read_calls += 1
        self.payload_bytes_read += record['bytes']
        self.ram_read_calls += 1
        self.ram_bytes_read += record['bytes']
        return memoryview(resident)[segment['offset']:segment['offset']+record['bytes']].toreadonly()

    def row_span_view(self,name,start,end):
        record=self.info(name);shape=record['shape']
        if (len(shape)!=2 or shape[0]<1 or record['bytes']%shape[0] or
                type(start) is not int or type(end) is not int or not 0<=start<=end<=shape[0]):
            raise ValueError('Invalid candidate packed row span')
        length=(end-start)*(record['bytes']//shape[0]);offset=start*(record['bytes']//shape[0])
        if length>self.read_limit_bytes:raise ValueError('Candidate packed row span exceeds bounded read limit')
        if not length:return memoryview(b'')
        if record.get('codec')=='internal-row-scales':return memoryview(self.rows(name,np.arange(start,end)))
        for segment in record['segments']:
            if segment['tensor_offset']<=offset and offset+length<=segment['tensor_offset']+segment['bytes']:
                resident=self.ram_shards.get(segment['file'])
                if resident is not None:
                    begin=segment['offset']+offset-segment['tensor_offset']
                    value=memoryview(resident)[begin:begin+length].toreadonly()
                    if value.nbytes!=length:raise ValueError('Truncated candidate packed RAM span')
                    self.payload_read_calls+=1;self.payload_bytes_read+=length
                    self.ram_read_calls+=1;self.ram_bytes_read+=length
                    return value
        return memoryview(self._range(name,offset,length))

    def float_rows(self,name,indices):
        record = self.info(name)
        indices = np.asarray(indices,dtype=np.int64)
        raw = self.rows(name,indices)
        if record.get('codec') == QUANT_FORMAT:
            width,bits = record['shape'][1],record['bits']
            if len(indices)*width*4>64*1024**2:
                raise ValueError('Decoded candidate rows exceed64MiB')
            data = np.frombuffer(raw,dtype=np.uint8).reshape(len(indices),record['bytes']//record['shape'][0])
            code_bytes = width*bits//8
            scales = data[:,code_bytes:].copy().view('<u2')
            output=QuantizedRows(data[:,:code_bytes].copy(),scales,bits,width,32,0.,0.).decode()
            protection=record.get('protected_native_rows')
            if protection and len(indices):
                ids=self.protected_ids.get(name)
                if ids is None:
                    ids=np.frombuffer(self.raw(protection['ids']),dtype='<i8').copy()
                    if np.any(ids<0) or np.any(ids>=record['shape'][0]) or np.any(np.diff(ids)<=0):
                        raise ValueError('Protected native row IDs must be unique sorted source positions')
                    self.protected_ids[name]=ids
                positions=np.searchsorted(ids,indices)
                selected=(positions<len(ids)) & (ids[np.minimum(positions,len(ids)-1)]==indices)
                if np.any(selected):
                    output[selected]=self.float_rows(protection['weight'],positions[selected])
            return output
        shape = (len(indices),record['shape'][1])
        if record['dtype'] == 'BF16':
            return bf16(raw).reshape(shape)
        if record['dtype'] == 'F32':
            return np.frombuffer(raw,dtype='<f4').reshape(shape)
        codes = np.frombuffer(raw,dtype=np.uint8).reshape(shape)
        if record['dtype'] == 'F8_E8M0':
            return e8m0(codes)
        scale_name = record.get('native_scale_name',name.rsplit('.',1)[0]+'.scale')
        if record['dtype'] == 'I8' and '.experts.' in name:
            scale = self.info(scale_name)
            scales = np.frombuffer(self.rows(scale_name,indices),dtype=np.uint8).reshape(len(indices),scale['shape'][1])
            return mxfp4(codes,scales)
        if record['dtype'] == 'F8_E4M3':
            if scale_name not in self.index:
                return e4m3fn(codes)
            scale = self.info(scale_name)
            if '.engram.embed.' in name:
                scales = np.frombuffer(self.rows(scale_name,indices),dtype=np.uint8).reshape(len(indices),scale['shape'][1])
                return fp8_row_groups(codes,scales)
            scales = np.frombuffer(self.raw(scale_name),dtype=np.uint8).reshape(scale['shape'])
            return fp8_block_rows(codes,scales,indices)
        raise ValueError('Unsupported candidate tensor dtype/codec')

    def preload_packed_ram(self,*,budget_bytes,reserve_bytes=80*1024**3,on_progress=None):
        """Verify and own only this bundle's packed files, without expansion."""
        import time
        from .weights import available_memory_bytes
        weights = {name:record for name,record in self.verified.items() if record['kind'] == 'weights'}
        total = sum(record['size'] for record in weights.values())
        if self.ram_shards or type(budget_bytes) is not int or type(reserve_bytes) is not int or reserve_bytes < 1 or total > budget_bytes:
            raise MemoryError('Packed candidate already resident or declared RAM budgets are insufficient')
        if available_memory_bytes() < total+reserve_bytes+min(total,self.read_limit_bytes):
            raise MemoryError('Candidate RAM load would violate the host reserve')
        manifest_path = safe_file(self.source,'model.bundle.json')
        manifest_raw = manifest_path.read_bytes()
        if json.loads(manifest_raw) != self.manifest:
            raise ValueError('Candidate manifest changed before RAM load')
        metadata_sha256 = {}
        for name,record in sorted(self.verified.items()):
            if record['kind'] != 'weights':
                raw = safe_file(self.source,name).read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != record['digest']:
                    raise ValueError('Candidate metadata SHA256 mismatch: '+name)
                metadata_sha256[name] = digest
        buffers,proof,started,copied = {},[],time.perf_counter(),0
        try:
            for name,record in sorted(weights.items()):
                storage = np.empty(record['size'],dtype=np.uint8)
                view,digest = memoryview(storage),hashlib.sha256()
                offset = 0
                try:
                    while offset < record['size']:
                        stop = min(offset+self.read_limit_bytes,record['size'])
                        target = view[offset:stop]
                        received = 0
                        try:
                            while received < target.nbytes:
                                count = os.preadv(self.descriptors[name],[target[received:]],offset+received)
                                if not count:
                                    raise ValueError('Truncated candidate packed RAM load')
                                received += count
                            digest.update(target)
                        finally:
                            target.release()
                        offset = stop
                        copied += received
                        if available_memory_bytes() < reserve_bytes:
                            raise MemoryError('Host reserve exhausted during candidate RAM load')
                        if on_progress:
                            on_progress({'phase':'candidate_packed_ram_copy','bytes_copied':copied,'total_bytes':total})
                finally:
                    view.release()
                if digest.hexdigest() != record['digest']:
                    raise ValueError('Candidate RAM SHA256 differs: '+name)
                storage.flags.writeable = False
                buffers[name] = storage
                proof.append({'name':name,'bytes':record['size'],'sha256':digest.hexdigest(),
                    'buffer_address':int(storage.ctypes.data)})
            if manifest_path.read_bytes() != manifest_raw:
                raise ValueError('Candidate manifest changed during RAM load')
            self.ram_shards = buffers
            self.ram_preload_report = {'phase':'all_candidate_shards_sha256_verified_in_ram','source_revision':REVISION,
                'resident_bytes':total,'budget_bytes':budget_bytes,'reserve_bytes':reserve_bytes,
                'copy_seconds':time.perf_counter()-started,'shards':proof,'dequantized':False,'disk_copy_created':False,
                'model_bundle_sha256':hashlib.sha256(manifest_raw).hexdigest(),
                'checkpoint_kind':'own_bundle','metadata_SHA_verified':True,
                'metadata_sha256':metadata_sha256}
            return self.ram_preload_report
        except BaseException:
            buffers.clear()
            raise

    def close(self):
        for descriptor in self.descriptors.values():
            os.close(descriptor)
        self.descriptors.clear()
        self.ram_shards.clear()
        self.protected_ids.clear()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
