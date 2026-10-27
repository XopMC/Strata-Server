"""One CPU-owned old Output16 bank over NEW authenticated1017 mixed bytes.
No model IO, native DSO or CUDA on import. Root constructs CPU-only then forks
one GPU child; old tasks/graphs retain this owner until joined/drained.
"""
import ctypes as C
import fcntl,hashlib,importlib.util,json,math,mmap,os,re,sys,time,threading
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from types import MappingProxyType
HERE=Path(__file__).resolve().parent
Q=HERE.parents[1];D=Q.parent/'deepseek41'
GIB=1<<30;ALIGN=2<<20;PAGE=4096
EXPERT=re.compile(r'layers\.(?:[0-9]|[1-3][0-9])\.ffn\.experts\.(?:[0-9]|[1-9][0-9]|[12][0-9][0-9]|3[0-7][0-9]|38[0-3])\.w[123]\.weight')
OUTPUT_BYTES=305764761600;RAW_BYTES=129345450600;TOTAL_BYTES=OUTPUT_BYTES+RAW_BYTES
PHASE='new1017_single_old_output16_bank_ready_v1'

def need(ok,text):
 if not ok:raise ValueError(text)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def source(path,name):
 path=Path(path);pins=json.loads((HERE/'DEPENDENCIES.json').read_text());need(digest(path)==pins[str(path)]['sha256'],'source changed: '+str(path))
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def signature(path):
 s=Path(path).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def epoch(pid):
 s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(s[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def available():
 return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
def rounded(n,a=PAGE):return(n+a-1)//a*a

class Projection(C.Structure):
 _fields_=[('data',C.c_void_p),('bytes',C.c_uint64),('row_bytes',C.c_uint64),('rows',C.c_uint32),('columns',C.c_uint32),('owner_generation',C.c_uint64),('struct_size',C.c_uint32),('version',C.c_uint32),('base_e8',C.c_uint8),('global_e8',C.c_uint8),('scale_delta_bits',C.c_uint8),('max_scale_delta',C.c_uint8),('reserved',C.c_uint32),('reserved2',C.c_uint64)]
assert C.sizeof(Projection)==64

class Allocation:
 """Actual2MiB-aligned anonymous allocation; no private malloc/free contract."""
 def __init__(self,size,node=None):
  need(type(size)is int and size>0,'nonempty allocation');self.address=0;self.size=size;self.extent=rounded(size);self.array=None
  self.lib=C.CDLL(None,use_errno=True);self.lib.mmap.argtypes=[C.c_void_p,C.c_size_t,C.c_int,C.c_int,C.c_int,C.c_long];self.lib.mmap.restype=C.c_void_p
  for n in ('munmap','mprotect'):getattr(self.lib,n).argtypes=[C.c_void_p,C.c_size_t]+([C.c_int]if n=='mprotect'else[]);getattr(self.lib,n).restype=C.c_int
  total=self.extent+ALIGN;base=self.lib.mmap(None,total,3,0x22,-1,0)
  if base==C.c_void_p(-1).value:raise OSError(C.get_errno(),'anonymous bank mmap')
  aligned=rounded(int(base),ALIGN);prefix=aligned-int(base);suffix=int(base)+total-aligned-self.extent
  if prefix:need(self.lib.munmap(base,prefix)==0,'unmap alignment prefix')
  if suffix:need(self.lib.munmap(aligned+self.extent,suffix)==0,'unmap alignment suffix')
  self.address=aligned
  self.lib.madvise.argtypes=[C.c_void_p,C.c_size_t,C.c_int];self.lib.madvise.restype=C.c_int
  self.hugepage_hint_result=self.lib.madvise(aligned,self.extent,14) # SAME old best-effort MADV_HUGEPAGE hint
  try:
   if node is not None:
    self.numa=C.CDLL('libnuma.so.1',use_errno=True);self.numa.mbind.argtypes=[C.c_void_p,C.c_ulong,C.c_int,C.c_void_p,C.c_ulong,C.c_uint];self.numa.mbind.restype=C.c_long
    mask=C.c_ulong(1<<node)
    if self.numa.mbind(aligned,self.extent,2,C.byref(mask),8,0):raise OSError(C.get_errno(),'NUMA bind before first fault')
  except BaseException:self.close();raise
 def writable_view(self):return memoryview((C.c_ubyte*self.size).from_address(self.address)).cast('B')
 def readonly(self):
  need(self.lib.mprotect(self.address,self.extent,1)==0,'bank readonly mprotect')
  import numpy as np
  self.array=np.frombuffer((C.c_ubyte*self.size).from_address(self.address),np.uint8);self.array.flags.writeable=False
 def close(self):
  self.array=None
  if self.address:
   address=self.address;self.address=0;need(self.lib.munmap(address,self.extent)==0,'bank munmap')

class OneBankStore:
 read_limit_bytes=64<<20
 def __init__(self,root,manifest_sha256,*,fixture=False):
  need(os.environ.get('DS1017_OLD_COMPLETE_OWNER_V1')=='1','onebank defaultOFF')
  need(sys.dont_write_bytecode,'PYTHONDONTWRITEBYTECODE=1 before process start protects frozen metadata/assets')
  self.root=self.source=Path(root).absolute();self._ready=False;self._closed=False;self._native_leases=0;self._regions=[];self._by_file={};self.cpu_blocked_cache={};self.cpu_blocked_report={};self.ram_preload_report={};self._fds={};self._alloc=[];self._lock=-1;self.fixture=fixture
  self.payload_read_calls=self.payload_bytes_read=self.ram_read_calls=self.ram_bytes_read=0;self.protected_ids={}
  try:
   V=source(Q/'research/compact-mixed-runtime-loader-v6/mapped_loader_v6.py','_old1017_metadata_only_v6');self._V=V
   self.manifest,raw=V.read_json(self.root/'model.bundle.json',128<<20);need(hashlib.sha256(raw).hexdigest()==manifest_sha256,'new bundle manifest SHA differs');self.model_bundle_sha256=manifest_sha256
   self.plans=V.plan_manifest(self.manifest,fixture=fixture);self.config,_=V.read_json(self.root/'runtime-config.json',32<<20)
   if not fixture:V.Bundle._validate_metadata(self)
   self.index={n:dict(r)for n,r in self.manifest['tensors'].items()}
   for n,r in self.index.items():
    enc=r.get('native_encoding',0)
    if enc==15:
     need(r['codec']=='deepseek41-signed-uniform-bf16-group-v1'and r['engram_descriptor']['overlay_rows']==0 and not r.get('overlay_segments'),'only exact physical Uniform4/no overlay Engram bridge')
     r.update(dtype='D41Q',bits=4)
    elif enc==14:need(EXPERT.fullmatch(n),'only canonical routed experts')
   self.verified={n:dict(row,digest=row['sha256'],size=row['bytes'])for n,row in self.manifest['files'].items()}
   if not fixture:
    V.regular(self.root/'.compact-mixed-v5.lock');self._lock=os.open(self.root/'.compact-mixed-v5.lock',os.O_RDONLY|os.O_NOFOLLOW);fcntl.flock(self._lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
   self._manifest_stat=signature(self.root/'model.bundle.json');self._file_stats={}
   for n,r in self.verified.items():
    s=V.regular(self.root/n);need(s.st_size==r['bytes'],'bundle file size differs');self._file_stats[n]=signature(self.root/n)
    if r['kind']=='weights':
     self._fds[n]=os.open(self.root/n,os.O_RDONLY|os.O_NOFOLLOW);v=os.fstat(self._fds[n]);need((v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)==self._file_stats[n],'opened source FD identity differs')
  except BaseException:self.close();raise
 def _stable(self,name=None):
  need(signature(self.root/'model.bundle.json')==self._manifest_stat,'source manifest changed')
  for n in([name]if name else self._file_stats):need(signature(self.root/n)==self._file_stats[n],'source file changed '+n)
 def _source_bytes(self,name):
  r=self.index[name];need(r['bytes']<=self.read_limit_bytes,'bounded source projection only');out=bytearray(r['bytes'])
  for n,a,size in self.plans[name][1]:
   self._stable(n);offset=next(s['tensor_offset']for s in r['segments']if s['file']==n and s['offset']==a);done=0
   while done<size:
    got=os.preadv(self._fds[n],[memoryview(out)[offset+done:offset+size]],a+done);need(got>0,'short source projection');done+=got
   os.posix_fadvise(self._fds[n],a,size,os.POSIX_FADV_DONTNEED);self._stable(n)
  return out
 def allocation_plan(self):
  experts=[];retained={n:[]for n in self._fds}
  for name,(enc,parts)in self.plans.items():
   if enc==14:
    r=self.index[name];n,k=r['shape'];experts.append(dict(name=name,rows=n,columns=k,bits=0,bytes=n//16*(k//32)*288,numa_node=(int(name.split('.')[1])+int(name.split('.')[4]))%2))
   else:
    for f,a,b in parts:retained[f].append((a,a+b))
  spans=[];nodes=[sum(rounded(e['bytes'])for e in experts if e['numa_node']==n)for n in(0,1)]
  for f,items in sorted(retained.items()):
   merged=[]
   for a,b in sorted(items):
    if merged and merged[-1][1]==a:merged[-1]=(merged[-1][0],b)
    else:merged.append((a,b))
   for a,b in merged:
    node=min(range(2),key=lambda n:nodes[n]);size=b-a;nodes[node]+=rounded(size);spans.append(dict(file=f,source_offset=a,bytes=size,numa_node=node))
  return dict(experts=experts,regions=spans,cache_bytes=sum(e['bytes']for e in experts),raw_bytes=sum(e['bytes']for e in spans),allocation_bytes=sum(nodes),node_bytes=nodes)
 def preload_selected(self,packer,packer_sha256,*,reserve_bytes=80*GIB,budget_bytes=432*GIB,on_progress=None,fixture_packer=None,integrity_receipt=None,integrity_sha256=None,workers=8,cancel=None):
  need(not self._ready and not self._alloc,'fresh onebank owner');need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only owner load/fork first')
  need(not any('libcuda.so'in x for x in Path('/proc/self/maps').read_text().splitlines()),'no inherited CUDA context')
  need(workers==8 or(self.fixture and workers==1),'exact bounded8 production pack workers')
  plan=self.allocation_plan();need(self.fixture or(plan['cache_bytes']==OUTPUT_BYTES and plan['raw_bytes']==RAW_BYTES and len(plan['experts'])==46080),'full production inventory')
  need(plan['allocation_bytes']<=budget_bytes and available()>=plan['allocation_bytes']+reserve_bytes+(128<<20),'onebank RAM budget/reserve')
  if not self.fixture:
   sys.path.insert(0,str(D/'tools'));from candidate_numa_balance import node_memory
   for n in(0,1):need(node_memory(n)['available_estimate']>=plan['node_bytes'][n]+32*GIB+(64<<20),'initial per-NUMA32GiB reserve')
  if fixture_packer is None:
   need(digest(packer)==packer_sha256,'explicit CPU packer SHA');lib=C.CDLL(str(packer));self.packer=lib
   lib.ds41_output16_v4_abi_version.restype=C.c_uint32;need(lib.ds41_output16_v4_abi_version()==1,'packer ABI1')
   lib.ds41_output16_v4_required_bytes.argtypes=[C.POINTER(Projection),C.POINTER(C.c_uint64)];lib.ds41_output16_v4_required_bytes.restype=C.c_int
   lib.ds41_output16_v4_pack.argtypes=[C.POINTER(Projection),C.c_uint64,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)];lib.ds41_output16_v4_pack.restype=C.c_int
  else:need(self.fixture,'fake packer cannot admit production');lib=None
  start=time.monotonic();last=[0.];self._owner_epoch=epoch(os.getpid());self.generation=int(self.model_bundle_sha256[:16],16)or 1
  def check_cancel():need(cancel is None or not cancel.is_set(),'cancelled during CPU bank construction')
  def progress(phase,**v):
   check_cancel()
   if on_progress and time.monotonic()-last[0]>=1:on_progress(dict(phase=phase,elapsed_seconds=time.monotonic()-start,**v));last[0]=time.monotonic()
  def allocate(size,node):
   need(available()>=reserve_bytes+rounded(size),'host reserve during load')
   if not self.fixture:need(node_memory(node)['available_estimate']>=rounded(size)+32*GIB,'per-NUMA reserve during load')
   a=Allocation(size,None if self.fixture else node);self._alloc.append(a);return a
  try:
   # Reuse Root's prior FULL SHA pass, bound to exact source stat identities.
   # The receipt's older numerical DSO is integrity provenance only.
   if not self.fixture:
    need(integrity_receipt is not None and integrity_sha256,'explicit prior full-SHA receipt required')
    proof,raw=self._V.read_json(integrity_receipt,4<<20);need(hashlib.sha256(raw).hexdigest()==integrity_sha256,'integrity receipt SHA')
    need(proof.get('phase')=='FULL_PAYLOAD_SHA256_VERIFIED_UNCHANGED_STATS'and Path(proof['bundle_root'])==self.source and proof.get('manifest_sha256')==self.model_bundle_sha256 and proof.get('no_weight_changes')is True,'prior full-SHA source binding')
    result,data=self._V.read_json(proof['native_result'],4<<20);need(hashlib.sha256(data).hexdigest()==proof['native_result_sha256']and result.get('payload_integrity')=='sha256'and result.get('manifest_sha256')==self.model_bundle_sha256 and result.get('status')=='ACTUAL_CHECKPOINT_LOADED'and result.get('owned_mappings_released')is True,'prior actual complete hash registration required')
    stamps=proof.get('payload_stamps',{});need(set(stamps)==set(self._fds),'all source shard stamps required')
    for f,fd in self._fds.items():
     v=os.fstat(fd);actual=(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns);need(tuple(stamps[f])==self._file_stats[f]==actual,'prior SHA file identity changed '+f);self._stable(f)
    self.integrity=dict(mode='prior-full-SHA+exact-dev-inode-size-mtime-ctime',receipt_sha256=integrity_sha256,native_result_sha256=proof['native_result_sha256'],prior_DSO_is_provenance_only=True)
   else:
    for f,r in self.verified.items():need(digest(self.source/f)==r['sha256'],'fixture file SHA')
    self.integrity=dict(mode='synthetic-fixture-full-SHA')
   for s in plan['regions']:
    a=allocate(s['bytes'],s['numa_node']);dst=a.writable_view();done=0
    while done<s['bytes']:
     check_cancel();n=min(32<<20,s['bytes']-done);got=os.preadv(self._fds[s['file']],[dst[done:done+n]],s['source_offset']+done);need(got>0,'short retained interval');done+=got
    del dst;a.readonly();r=dict(s,mapping=a,array=a.array,address=a.address,allocation_bytes=a.extent);self._regions.append(r);self._by_file.setdefault(s['file'],[]).append(r)
    os.posix_fadvise(self._fds[s['file']],s['source_offset'],s['bytes'],os.POSIX_FADV_DONTNEED);progress('new1017_protected_raw_retained',regions=len(self._regions))
   lock=threading.Lock();failed=threading.Event();entries={}
   def one(e):
    check_cancel();need(not failed.is_set(),'cancelled pack queue')
    old_affinity=None
    if not self.fixture:
     from candidate_numa_balance import bind_thread,restore_thread
     old_affinity=bind_thread(e['numa_node'])
    try:
     raw=self._source_bytes(e['name']);r=self.index[e['name']];a=allocate(e['bytes'],e['numa_node']);buf=(C.c_ubyte*len(raw)).from_buffer(raw)
     v=Projection(C.addressof(buf),len(raw),r['row_bytes'],e['rows'],e['columns'],self.generation,64,4,r['base_e8'],r['global_e8'],r['scale_delta_bits'],r['max_scale_delta'],0,0);written=C.c_uint64()
     if lib:
      required=C.c_uint64();need(lib.ds41_output16_v4_required_bytes(C.byref(v),C.byref(required))==0 and required.value==a.size,'packer descriptor/required size')
      need(lib.ds41_output16_v4_pack(C.byref(v),self.generation,a.address,a.size,C.byref(written))==0,'V4→literal old Output16 pack refused')
     else:written.value=fixture_packer(v,a.address,a.size)
     need(written.value==e['bytes'],'exact old Output16 byte count');del buf,raw;a.readonly()
     return dict(e,address=a.address,allocation_bytes=a.extent,allocation=a)
    finally:
     if old_affinity is not None:restore_thread(old_affinity)
   pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='new1017-old-output16')
   pending=set();iterator=iter(plan['experts']);exhausted=False
   try:
    while pending or not exhausted:
     check_cancel()
     while not exhausted and len(pending)<workers*2:
      try:e=next(iterator)
      except StopIteration:exhausted=True;break
      pending.add(pool.submit(one,e))
     if not pending:continue
     done,pending=wait(pending,return_when=FIRST_COMPLETED)
     for task in done:
      entry=task.result();entries[entry['name']]=entry;progress('new1017_old_output16_bank',matrices_complete=len(entries),matrices=len(plan['experts']))
   except BaseException:
    failed.set()
    for task in pending:task.cancel()
    raise
   finally:pool.shutdown(wait=True,cancel_futures=True) # ALL pack/IO owners join before publication/fork/free
   need(len(entries)==len(plan['experts']),'complete CPU output bank required')
   self.cpu_blocked_cache=entries
   self._stable()
   for fd in self._fds.values():os.close(fd)
   self._fds={};self.cpu_blocked_cache=MappingProxyType({n:MappingProxyType(v)for n,v in self.cpu_blocked_cache.items()});self._ready=True
   self.cpu_blocked_report=dict(phase='all_target_experts_CPU_output16_prepared',bytes=plan['cache_bytes'],matrices=len(plan['experts']),CPU_layout_codec='output16-compact-v3',alignment_bytes=ALIGN,model_bundle_sha256=self.model_bundle_sha256,builder_library=str(packer),library_sha256=packer_sha256,raw_expert_duplicate_retained=False,layout_producer='new1017-v4-to-EXACT-old-output16-v1',checkpoint_weights_changed=False)
   self.ram_preload_report=dict(phase=PHASE,owner_pid=os.getpid(),owner_epoch=self._owner_epoch,source_revision=self.manifest['source_revision'],model_bundle_sha256=self.model_bundle_sha256,metadata_SHA_verified=True,resident_bytes=plan['raw_bytes']+plan['cache_bytes'],raw_retained_bytes=plan['raw_bytes'],compact_expert_bytes=plan['cache_bytes'],source_weight_payload_bytes=self.manifest['payload_bytes'],raw_expert_duplicate_retained=False,whole_raw_shards_resident=False,runtime_admissible=not self.fixture,pack_workers=workers,all_pack_workers_joined=True,source_integrity=self.integrity,hugepage_hint_attempted=True,hugepage_hint_nonzero=sum(a.hugepage_hint_result!=0 for a in self._alloc),seconds=time.monotonic()-start)
   if not self.fixture:
    sys.path.insert(0,str(D/'tools'));from candidate_cpu_local import verify_local
    verify_local(self)
   return self.ram_preload_report
  except BaseException:self.close();raise
 def validate_selected(self,*,deep=False,**_):
  need(self._ready and not self._closed and epoch(self._owner_epoch['pid'])==self._owner_epoch,'live ready original owner epoch')
  if deep:
   need(sum(x['bytes']for x in self.cpu_blocked_cache.values())==self.ram_preload_report['compact_expert_bytes'],'bank conservation')
   for a in self._alloc:need(a.address and a.array is not None and not a.array.flags.writeable and a.array.ctypes.data==a.address,'readonly allocation lifetime')
  return self.ram_preload_report
 def validate_selected_owner(self,*,deep=False):return dict(self.validate_selected(deep=deep))
 def info(self,name):
  self.validate_selected();need(name in self.index,'unknown tensor');return self.index[name]
 def _retained_slices(self,name,offset,size):
  self.validate_selected();need(name not in self.cpu_blocked_cache,'raw main expert bytes intentionally omitted');r=self.info(name);need(0<=offset and 0<=size and offset+size<=r['bytes'],'range bounds');parts=[];end=offset+size
  for s in r['segments']:
   lo=max(offset,s['tensor_offset']);hi=min(end,s['tensor_offset']+s['bytes'])
   if lo>=hi:continue
   a=s['offset']+lo-s['tensor_offset'];regions=[x for x in self._by_file[s['file']]if x['source_offset']<=a and a+hi-lo<=x['source_offset']+x['bytes']];need(len(regions)==1,'retained span unavailable');v=regions[0];at=a-v['source_offset'];parts.append(memoryview(v['array'])[at:at+hi-lo].toreadonly())
  need(sum(x.nbytes for x in parts)==size,'retained byte conservation');return parts
 def read_range(self,name,offset,size):
  need(size<=self.read_limit_bytes,'64MiB read bound');parts=self._retained_slices(name,offset,size);self.ram_read_calls+=len(parts);self.ram_bytes_read+=size;return b''.join(parts)
 def raw(self,name):return self.read_range(name,0,self.info(name)['bytes'])
 def raw_view(self,name):
  r=self.info(name);need(r['bytes']<=self.read_limit_bytes,'64MiB view bound');p=self._retained_slices(name,0,r['bytes']);return p[0]if len(p)==1 else memoryview(b''.join(p))
 def retained_tensor_parts(self,name):return self._retained_slices(name,0,self.info(name)['bytes'])
 def row_span_view(self,name,start,end):
  r=self.info(name);need(len(r['shape'])==2 and type(start)is int and type(end)is int and 0<=start<=end<=r['shape'][0],'matrix row span');stride=r['bytes']//r['shape'][0];need((end-start)*stride<=self.read_limit_bytes,'row span64MiB');p=self._retained_slices(name,start*stride,(end-start)*stride);return p[0]if len(p)==1 else memoryview(b''.join(p))
 def rows(self,name,indices):
  import numpy as np
  r=self.info(name);ids=np.asarray(indices);need(ids.ndim==1 and ids.dtype.kind in 'iu' and len(r['shape'])==2 and bool(np.all(ids>=0)&np.all(ids<r['shape'][0])),'row IDs');stride=r['bytes']//r['shape'][0];need(len(ids)*stride<=self.read_limit_bytes,'row read64MiB');return b''.join(self.read_range(name,int(i)*stride,stride)for i in ids)
 def float_rows(self,name,indices):
  # Exact existing numeric row decoder, now on new retained bytes. This method
  # cannot fetch omitted raw main expert bytes, including through fallback.
  from backend.deepseek41.bundle import BundleStore
  return BundleStore.float_rows(self,name,indices)
 def engram_module(self):
  item=next(x for x in self.config['runtime_asset_map']['items']if x['runtime_path']=='inference/engram.py')
  need(digest(self.source/item['output_name'])==item['output_sha256'],'new bundled hash source changed')
  spec=importlib.util.spec_from_file_location('_new1017_old_engine_hash',self.source/item['output_name']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def compact_expert(self,name):self.validate_selected();return self.cpu_blocked_cache[name]
 def retain_native(self):self.validate_selected();self._native_leases+=1
 def release_native(self):need(self._native_leases>0,'balanced native lease');self._native_leases-=1
 def close(self):
  need(getattr(self,'_native_leases',0)==0,'drain engines/registry/graphs before owner close');self._ready=False
  self.cpu_blocked_cache={};self._regions=[];self._by_file={}
  for a in reversed(self._alloc):a.close()
  self._alloc=[]
  for fd in self._fds.values():os.close(fd)
  self._fds={}
  if self._lock>=0:os.close(self._lock);self._lock=-1
  self._closed=True
