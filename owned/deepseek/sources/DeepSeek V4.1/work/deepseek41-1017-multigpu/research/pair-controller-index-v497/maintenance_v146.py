"""Explicit fresh-parent maintenance only. Import performs no IO/native actions."""
from __future__ import annotations
import ctypes as C
import hashlib,json,os,queue,re,threading,sys
from pathlib import Path
from types import MappingProxyType
BYTES=6635520;MATRICES=46080;TOTAL=435110212200;RAW=129345450600;EXPERT=305764761600
COMMAND_KEYS={'action','operation_id','direction','expected_owner_epoch','expected_layout','expected_format_epoch','expected_inventory_sha256','helper_build_receipt_sha256','bounded_native_proof_sha256'}
DIRECTIONS={'old16_to_pair139':(16,139),'pair139_to_old16':(139,16)}
BINDING_KEYS={'schema','layout_tag','format_epoch','owner_generation','owner_epoch','inventory_sha256'}
NAME=re.compile(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight')
class MaintenanceError(RuntimeError):pass
def need(ok,message):
 if not ok:raise MaintenanceError(message)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def epoch():
 pid=os.getpid();v=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(v[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def subset(actual,expected):
 if isinstance(expected,dict):return isinstance(actual,dict)and all(k in actual and subset(actual[k],v)for k,v in expected.items())
 return type(actual)is type(expected)and actual==expected
def pinned(row):
 p=Path(row['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink(),'pin path must be actual canonical regular file')
 need(p.stat().st_size<=64<<20,'maintenance metadata/helper pin exceeds64MiB bound')
 need(digest(p)==row['sha256'],'pinned file changed: '+str(p));return p
def receipt(row):
 p=pinned(row);value=json.loads(p.read_text());need(isinstance(row.get('contract'),dict)and row['contract']and subset(value,row['contract']),'actual proof contract differs: '+str(p));return value
def normalized_binding(value):
 out=dict(value);out['owner_epoch']=dict(out['owner_epoch']);return out
def assert_child_binding(store,expected):
 need(set(expected)==BINDING_KEYS,'child binding schema differs')
 need(getattr(store,'bank_runtime_admissible_v146',False)is True,'child inherited unadmissible maintenance bank')
 actual=normalized_binding(store.bank_format_binding_v146)
 need(actual==expected and os.getpid()!=expected['owner_epoch']['pid'],'child inherited format/epoch differs')
 pid=expected['owner_epoch']['pid'];v=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 need(int(v[19])==expected['owner_epoch']['start_ticks']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==expected['owner_epoch']['boot_id'],'child original parent identity changed')
class Txn(C.Structure):
 _fields_=[('data',C.c_void_p)]+[(k,C.c_uint64)for k in('bytes','parent_pid','source_generation','format_epoch')]+[(k,C.c_uint32)for k in('rows','columns','from_layout','to_layout')]
class Report(C.Structure):
 _fields_=[(k,C.c_uint64)for k in('schema','struct_bytes','parent_pid','source_generation','format_epoch','from_layout','to_layout','bytes','compared_bytes','write_enabled','readonly_restored','source_restored','owner_poisoned')]+[('error',C.c_char*192)]
def maps():
 result=[]
 for line in Path('/proc/self/maps').read_text().splitlines():
  a,perms,offset,device,inode,*rest=line.split();lo,hi=(int(x,16)for x in a.split('-'));result.append((lo,hi,perms,inode))
 return result
def require_ro(entries):
 regions=maps();at=0
 for e in sorted(entries,key=lambda x:x['address']):
  lo=e['address'];end=lo+e['bytes']
  while at<len(regions)and regions[at][1]<=lo:at+=1
  j=at
  while lo<end:
   need(j<len(regions),'matrix VMA missing');a,b,perms,inode=regions[j]
   need(a<=lo<b and perms=='r--p'and inode=='0','matrix not private anonymous actual RO');lo=min(b,end);j+=1

def physical_snapshot(entries):
 """Root-only maintenance check, no payload reads: exact VMA NUMA/THP metadata."""
 require_ro(entries);numa={}
 for line in Path('/proc/self/numa_maps').read_text().splitlines():
  fields=line.split();numa[int(fields[0],16)]=fields[1:]
 huge={};shared={};start=None
 for line in Path('/proc/self/smaps').read_text().splitlines():
  first=line.split(None,1)[0]if line else ''
  if re.fullmatch(r'[0-9a-f]+-[0-9a-f]+',first):start=int(first.split('-')[0],16)
  elif line.startswith('AnonHugePages:')and start is not None:huge[start]=int(line.split()[1])
  elif line.startswith(('Shared_Clean:','Shared_Dirty:'))and start is not None:shared[start]=shared.get(start,0)+int(line.split()[1])
 records=[];thp_histogram={}
 for e in entries:
  a=e['address'];node=e['numa_node'];fields=numa.get(a,[]);counts={int(x.split('=')[0][1:]):int(x.split('=')[1])for x in fields if re.fullmatch(r'N\d+=\d+',x)}
  need('bind:'+str(node)in fields and counts.get(node,0)==BYTES//4096 and sum(counts.values())==BYTES//4096,'expert NUMA/resident metadata differs: '+json.dumps(dict(name=e['name'],address=a,node=node,fields=fields)))
  need(a in huge and 0<=huge[a]<=6144 and huge[a]%2048==0,'invalid expert THP accounting: '+json.dumps(dict(name=e['name'],address=a,huge_kib=huge.get(a))))
  need(shared.get(a,-1)==0,'expert ownership is shared; parent maintenance refused: '+json.dumps(dict(name=e['name'],address=a,shared_kib=shared.get(a),huge_kib=huge.get(a))))
  records.append([a,e['allocation_bytes'],node,counts[node]])
  thp_histogram[str(huge[a])]=thp_histogram.get(str(huge[a]),0)+1
 print(json.dumps(dict(phase='PHYSICAL_PRIVATE_BANK_CHECK',matrices=len(records),THP_kib_histogram=thp_histogram,shared_kib=0)),flush=True)
 return canonical(records)
class Native:
 def __init__(self,cfg,generation):
  need(C.sizeof(Txn)==56 and C.sizeof(Report)==296,'native ABI layout differs')
  self.schema=cfg['abi']['report_schema'];need(type(self.schema)is int and self.schema>0,'explicit actual helper schema required')
  build=receipt(cfg['helper_build']);library=pinned(cfg['library'])
  # Build receipt must independently carry the same actual library identity.
  need((build.get('library')==cfg['library'])or build.get('files',{}).get(str(library))==cfg['library']['sha256'],'helper library absent from actual build receipt')
  receipt(cfg['bounded_native_proof'])
  capabilities={x['capability']for x in cfg['native_gates']}
  need(capabilities=={'actual18_matrix_codec','CPU143_full_pool','GPU141_both_devices','registry145_gpu0'},'all bounded typed-consumer gates required before parent relayout')
  for row in cfg['native_gates']:receipt(row)
  key=cfg['abi']['environment_key'];need(re.fullmatch(r'DS1017_PARENT_RELAYOUT_V\d+',key)and cfg['abi']['environment_value']=='1','helper policy binding differs')
  need(os.environ.get(key)in(None,'1'),'helper policy collision');os.environ[key]='1'
  self.lib=C.CDLL(str(library));symbols=cfg['abi']['symbols'];need(set(symbols)=={'bind','matrix','restore'},'explicit helper symbols required')
  for value in symbols.values():need(re.fullmatch(r'ds\d+_parent_(?:bind|matrix|restore_saved)',value),'helper symbol invalid')
  self.bind=getattr(self.lib,symbols['bind']);self.bind.argtypes=[C.c_uint64,C.c_uint64];self.bind.restype=C.c_int
  for key in('matrix','restore'):
   fn=getattr(self.lib,symbols[key]);fn.argtypes=[C.POINTER(Txn),C.c_void_p,C.c_size_t,C.POINTER(Report)];fn.restype=C.c_int;setattr(self,key,fn)
  need(self.bind(os.getpid(),generation)==0,'fresh parent helper bind refused')
  self.backup_codes=set(cfg['abi']['backup_available_returncodes']);need(self.backup_codes=={0,3,4,5,6},'actual148 backup status contract required;9 always poison')
 def check(self,r,t,restore=False):
  need(r.schema==self.schema and r.struct_bytes==296 and all(getattr(r,k)==getattr(t,k)for k in('parent_pid','source_generation','format_epoch','from_layout','to_layout','bytes')),'native report identity differs')
  need(r.owner_poisoned==0 and r.readonly_restored==1 and r.compared_bytes==BYTES,'native exact+RO report missing')
  need(r.source_restored==int(restore),'native restore/source state differs')
class Manager:
 """Single actual parent owns this object; no live-owner import hooks."""
 def __init__(self,store,owner_module,cfg_path,cfg_sha,state_dir,atomic_json,health,exclusion,emit=print,*,allow_profile_admission_v163=False,profile_validator_v163=None):
  row=dict(path=str(Path(cfg_path).absolute()),sha256=cfg_sha);cfg=json.loads(pinned(row).read_text())
  need(cfg.get('schema')=='deepseek1017-parent-maintenance-config/146'and cfg.get('admitted_for_parent_maintenance')is True,'maintenance default-off; Root actual proof configuration required')
  need(type(allow_profile_admission_v163)is bool,'profile admission opt-in must be bool');self.allow_profile_admission_v163=allow_profile_admission_v163;self.profile_validator_v163=profile_validator_v163;self.extra_profiles_v163=()
  self.store=store;self.owner_module=owner_module;self.cfg=cfg;self.config_pin=row;self.atomic=atomic_json;self.health=health;self.exclusion=exclusion;self.emit=emit
  self.parent=epoch();need(self.parent==store._owner_epoch and self.parent['pid']==os.getpid(),'fresh actual owner PID/boot/start required')
  need(not store.fixture and store._ready and not store._closed and not store._fds,'exact fully loaded original owner required')
  self.generation=store.generation;self.layout=16;self.format_epoch=1;self.phase='READY_OLD16';self.active=None;self.poisoned=False;self.seen=set();self.workspaces=[];self.threads=[];self.workers_ready=False;self.child_drain_proved=True
  self.root=Path(state_dir);need(not self.root.exists(),'fresh maintenance state directory required');self.root.mkdir(mode=0o700)
  self.inventory=self._inventory();self.inventory_sha=canonical(dict(owner_epoch=self.parent,owner_generation=self.generation,model_bundle_sha256=store.model_bundle_sha256,entries=self.inventory))
  self._guard();self.physical=physical_snapshot(self.inventory);self.native=Native(cfg,self.generation)
  self.atomic(self.root/'inventory.json',dict(owner_epoch=self.parent,owner_generation=self.generation,inventory_sha256=self.inventory_sha,entries=self.inventory))
  self.atomic(self.root/'initial.json',self.status());self._publish_field()
 def _inventory(self):
  s=self.store;need(len(s.cpu_blocked_cache)==MATRICES,'exact46080 expert descriptors required')
  out=[];alloc_ids={id(a)for a in s._alloc}
  for name,e in sorted(s.cpu_blocked_cache.items()):
   match=NAME.fullmatch(name);need(match is not None,'noncanonical expert descriptor');l,x,role=match.groups();l=int(l);x=int(x)
   need(0<=l<40 and 0<=x<384 and e['name']==name,'expert role/name mismatch')
   shape=(5120,2304)if role=='w2'else(2304,5120);a=e['allocation']
   need((e['rows'],e['columns'])==shape and s.index[name]['shape']==list(shape),'actual source expert shape differs')
   need(e['bits']==0 and e['bytes']==BYTES and e['address']%2097152==0 and e['allocation_bytes']==BYTES,'expert exact byte/allocation geometry differs')
   need(id(a)in alloc_ids and a.address==e['address']and a.size==BYTES and a.extent==BYTES and a.array is not None and not a.array.flags.writeable and a.array.ctypes.data==a.address,'expert allocation identity/lifetime differs')
   need(e['numa_node']==(l+x)%2,'original expert node differs')
   out.append(dict(name=name,address=e['address'],bytes=BYTES,rows=shape[0],columns=shape[1],bits=0,numa_node=e['numa_node'],allocation_bytes=BYTES))
  need(sum(e['bytes']for e in out)==EXPERT and s.ram_preload_report['resident_bytes']==TOTAL and s.ram_preload_report['raw_retained_bytes']==RAW,'one bank conservation differs')
  spans=[(e['address'],e['address']+BYTES)for e in out]+[(e['address'],e['address']+e['bytes'])for e in s._regions]
  spans.sort();need(all(a[1]<=b[0]for a,b in zip(spans,spans[1:])),'expert/raw ranges overlap')
  return out
 def _guard(self):
  need(epoch()==self.parent and self.store.generation==self.generation and self.store._ready and not self.store._closed,'own parent epoch/source generation/readiness changed')
  need(self.child_drain_proved,'prior child close/drain proof missing')
  need(self._inventory()==self.inventory,'actual immutable matrix inventory/allocation metadata changed')
  need('torch'not in sys.modules,'Torch forbidden in maintenance parent')
  need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and not any('libcuda.so'in x for x in Path('/proc/self/maps').read_text().splitlines()),'parent must remain CUDA empty')
  need(getattr(self.store,'_native_leases',-1)==0,'owner native leases unresolved')
  for p in Path('/proc/self/task').glob('*/children'):need(not p.read_text().strip(),'parent has child/unreaped child')
  self.exclusion();self.health()
  for p in Path('/proc/self/task').glob('*/children'):need(not p.read_text().strip(),'health check left an unreaped child')
  if not self.threads:need(len(list(Path('/proc/self/task').iterdir()))==1,'parent must join every worker before fresh work/fork')
 def binding(self):return dict(schema='deepseek1017-bank-format/146',layout_tag=self.layout,format_epoch=self.format_epoch,owner_generation=self.generation,owner_epoch=dict(self.parent),inventory_sha256=self.inventory_sha)
 def _publish_field(self):
  b=self.binding();b['owner_epoch']=MappingProxyType(b['owner_epoch']);self.store.bank_format_binding_v146=MappingProxyType(b);self.store.bank_runtime_admissible_v146=True
 def status(self):return dict(phase=self.phase,bank_format=self.binding(),one_bank_bytes=TOTAL,raw_retained_bytes=RAW,compact_expert_bytes=EXPERT,runtime_admissible=self.active is None and not self.poisoned and self.child_drain_proved and self.phase in('READY_OLD16','READY_PAIR139'),operation_id=self.active['id']if self.active else None,committed_matrices=sum(x.bit_count()for x in self.active['bits'])if self.active else None,unacknowledged_scratch=len(self._pending()),owner_poisoned=self.poisoned)
 def assert_run(self,req,adapter_root,adapter_sha,libraries):
  self._guard();self._proof_guard();need(self.active is None and not self.poisoned and self.phase in('READY_OLD16','READY_PAIR139'),'partial/poisoned bank cannot run')
  need(req['runtime'].get('bank_format_binding')==self.binding(),'request must bind exact actual layout/epoch/inventory/owner')
  profiles=[x for x in tuple(self.cfg['format_contracts'])+self._extra_profiles_v163()if x['layout_tag']==self.layout and x['adapter_root']==str(adapter_root)and x['adapter_seal_sha256']==adapter_sha]
  need(len(profiles)==1,'adapter not proved for this actual bank format');profile=profiles[0];receipt(profile['proof'])
  for context,(path,sha)in libraries.items():need(profile['libraries'].get(str(context))==dict(path=str(path),sha256=sha),'runtime DSO not proved for this bank format')
  require_ro(self.inventory);return self.binding()
 def _canonical_profile_path_v163(self,value,*,directory=False):
  need(isinstance(value,str),'profile path must be text');p=Path(value);root=Path(__file__).resolve().parents[4]
  need(p.is_absolute()and p.resolve()==p and p.is_relative_to(root)and not p.is_symlink(),'profile path must be canonical and inside canonical DeepSeek section')
  need(p.is_dir()if directory else p.is_file(),'profile path missing');return p
 def _read_profile_v163(self,pin,expected_binding):
  p=self._canonical_profile_path_v163(pin['path']);profile=json.loads(pinned(pin).read_text())
  keys={'schema','profile_id','bank_format_binding','layout_tag','adapter_root','adapter_seal_sha256','pair_build','proof','libraries','quality_admitted','performance_admitted'}
  need(isinstance(profile,dict)and set(profile)==keys and profile['schema']=='deepseek1017-format-profile/163','exact profile163 schema required')
  need(isinstance(profile['profile_id'],str)and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}',profile['profile_id']),'invalid profile ID')
  need(canonical(profile['bank_format_binding'])==canonical(expected_binding),'profile binds another bank owner/layout/epoch/inventory')
  need(type(profile['layout_tag'])is int and profile['layout_tag']in(16,139)and profile['layout_tag']==expected_binding['layout_tag'],'profile layout differs')
  need(profile['quality_admitted']is False and profile['performance_admitted']is False,'profile cannot assert model quality/performance')
  self._canonical_profile_path_v163(profile['adapter_root'],directory=True)
  need(isinstance(profile['adapter_seal_sha256'],str)and re.fullmatch(r'[0-9a-f]{64}',profile['adapter_seal_sha256']),'actual adapter seal SHA required')
  need(isinstance(profile['libraries'],dict)and set(profile['libraries'])=={'262144','524288'},'both contexts required in profile')
  for row in profile['libraries'].values():
   need(isinstance(row,dict)and set(row)=={'path','sha256'}and isinstance(row['sha256'],str)and re.fullmatch(r'[0-9a-f]{64}',row['sha256']),'exact profile DSO path/SHA required');self._canonical_profile_path_v163(row['path'])
  for name in('pair_build','proof'):
   row=profile[name];need(isinstance(row,dict)and set(row)=={'path','sha256','contract'},'actual build/proof pin contract required');self._canonical_profile_path_v163(row['path']);receipt(row)
  need(callable(self.profile_validator_v163),'Root controller profile validator unavailable');self.profile_validator_v163(profile)
  return profile
 def _extra_profiles_v163(self):
  out=[];current=canonical(self.binding())
  for ident,path,sha,binding_json,event_path,event_sha in getattr(self,'extra_profiles_v163',()):
   if canonical(json.loads(binding_json))!=current:continue # Admission never replays into a new epoch.
   pinned(dict(path=event_path,sha256=event_sha));out.append(self._read_profile_v163(dict(path=path,sha256=sha),json.loads(binding_json)))
  return tuple(out)
 def admit_profile_v163(self,command):
  need(getattr(self,'allow_profile_admission_v163',False)is True,'profile admission is disabled; fresh Root startup opt-in required')
  self._guard();self._proof_guard()
  need(self.active is None and not self.poisoned and not self.threads and not self._pending()and self.phase in('READY_OLD16','READY_PAIR139'),'profile admission requires idle exact bank')
  need(set(command)=={'action','profile_path','profile_sha256','expected_bank_format_binding'}and command['action']=='admit_format_profile','exact private profile-admission command required')
  need(canonical(command['expected_bank_format_binding'])==canonical(self.binding()),'admission actual bank binding differs')
  pin=dict(path=command['profile_path'],sha256=command['profile_sha256']);profile=self._read_profile_v163(pin,self.binding());ident=profile['profile_id']
  extras=getattr(self,'extra_profiles_v163',());need(len(extras)<256 and all(x[0]!=ident for x in extras),'duplicate profile ID or256-profile bound')
  identity=lambda x:(x['layout_tag'],x['adapter_root'],x['adapter_seal_sha256'])
  rows=list(self.cfg['format_contracts'])
  for _,path,sha,binding_json,_,_ in extras:rows.append(self._read_profile_v163(dict(path=path,sha256=sha),json.loads(binding_json)))
  need(all(identity(x)!=identity(profile)and x['adapter_root']!=profile['adapter_root']for x in rows),'profile replacement/duplicate adapter is forbidden')
  # Full bank checks stay on operation boundaries; this metadata action writes no bank bytes.
  require_ro(self.inventory);need(physical_snapshot(self.inventory)==self.physical,'profile admission physical bank changed');self._guard();self._proof_guard()
  event_path=self.root/('profile-admission-'+ident+'.json');need(not event_path.exists(),'profile admission journal already exists')
  event=dict(schema='deepseek1017-profile-admission/163',phase='ROOT_EXPLICIT_PROFILE_ADMITTED_NOT_MODEL_VALIDATED',profile_pin=pin,profile=profile,bank_format_binding=self.binding(),quality_admitted=False,performance_admitted=False)
  self.atomic(event_path,event) # file+directory fsync ACK precedes any in-memory admission.
  event_sha=digest(event_path);self.extra_profiles_v163=extras+((ident,pin['path'],pin['sha256'],json.dumps(self.binding(),sort_keys=True,separators=(',',':')),str(event_path),event_sha),)
  return dict(phase=event['phase'],profile_id=ident,profile_sha256=pin['sha256'],bank_format_binding=self.binding(),extra_profiles=len(self.extra_profiles_v163),quality_admitted=False,performance_admitted=False)
 def _proof_guard(self):
  pinned(self.config_pin);pinned(self.cfg['library']);receipt(self.cfg['helper_build']);receipt(self.cfg['bounded_native_proof'])
  for row in self.cfg['native_gates']:receipt(row)
 def _pending(self):return [w for w in self.workspaces if w.get('pending')is not None]
 def _checkpoint(self,bits,phase):
  a=self.active;payload=dict(schema='deepseek1017-parent-relayout-checkpoint/146',operation_id=a['id'],owner_epoch=self.parent,source_generation=self.generation,inventory_sha256=self.inventory_sha,from_layout=a['from'],to_layout=a['to'],source_format_epoch=a['epoch'],next_format_epoch=a['epoch']+1,sequence=a['sequence']+1,phase=phase,bitmap_hex=bytes(bits).hex(),committed_matrices=sum(x.bit_count()for x in bits),helper_build_receipt_sha256=self.cfg['helper_build']['sha256'],bounded_native_proof_sha256=self.cfg['bounded_native_proof']['sha256'])
  self.atomic(a['path']/'checkpoint.json',payload);a['sequence']+=1;a['bits']=bytearray(bits)
 def _start(self):
  need(not self.threads,'maintenance workers already running');cpus=self.cfg['worker_cpus'];need(len(cpus)==8 and len(set(cpus))==8 and all(type(c)is int and 0<=c<256 for c in cpus),'eight owned physical worker CPUs required')
  need([c//128 for c in cpus]==[0]*4+[1]*4 and set(cpus)<=os.sched_getaffinity(0),'four allowed worker CPUs per node required')
  cores=[]
  for cpu in cpus:
   root=Path('/sys/devices/system/cpu')/('cpu'+str(cpu));need((root/('node'+str(cpu//128))).exists(),'actual worker NUMA node differs')
   cores.append(((root/'topology/physical_package_id').read_text().strip(),(root/'topology/core_id').read_text().strip()))
  need(len(set(cores))==8,'maintenance requires eight distinct physical cores')
  if not self.workspaces:self.workspaces=[dict(cpu=c,node=c//128,scratch=None,pending=None,q=queue.Queue())for c in cpus]
  ready=queue.Queue()
  def worker(w):
   try:
    os.sched_setaffinity(0,{w['cpu']});need(os.sched_getaffinity(0)=={w['cpu']},'maintenance own-thread affinity differs')
    if w['scratch']is None:
     w['scratch']=self.owner_module.Allocation(BYTES,w['node']);C.memset(w['scratch'].address,0,BYTES)
    ready.put(None)
   except BaseException as exc:ready.put(exc);return
   while True:
    task=w['q'].get()
    if task is None:return
    try:
     fn=self.native.restore if task['restore']else self.native.matrix
     task['rc']=fn(C.byref(task['txn']),w['scratch'].address,BYTES,C.byref(task['report']))
    except BaseException as exc:task['error']=str(exc)
    finally:task['done'].set()
  for w in self.workspaces:
   t=threading.Thread(target=worker,args=(w,),name='relayout146-node'+str(w['node']));t.start();self.threads.append(t)
  errors=[ready.get()for _ in self.threads];need(not any(x is not None for x in errors),'scratch worker initialization failed: '+str(errors));self.workers_ready=True
 def _stop(self):
  for w,t in zip(self.workspaces,self.threads):
   if t.is_alive():w['q'].put(None)
  for t in self.threads:t.join()
  self.threads=[];self.workers_ready=False
 def _free(self):
  need(not self.threads and not self._pending(),'cannot free live/unACKed scratch')
  for w in self.workspaces:
   if w['scratch']is not None:w['scratch'].close()
  self.workspaces=[]
 def _task(self,w,index,from_layout,to_layout,restore=False,txn=None):
  need(self.workers_ready,'all maintenance workers must initialize before native calls');e=self.inventory[index]
  if txn is None:txn=Txn(e['address'],BYTES,self.parent['pid'],self.generation,self.active['epoch'],e['rows'],e['columns'],from_layout,to_layout)
  task=dict(index=index,txn=txn,report=Report(),done=threading.Event(),rc=None,error=None,restore=restore)
  if not restore:w['pending']=task
  try:w['q'].put(task)
  except BaseException as exc:task['error']=str(exc);task['done'].set();raise
  return task
 def _batch(self,indices,from_layout,to_layout,rollback):
  picked=[]
  for node in(0,1):
   local=[i for i in indices if self.inventory[i]['numa_node']==node][:4]
   for w,i in zip(self.workspaces[node*4:node*4+4],local):
    need(w['pending']is None,'scratch reused before durable ACK');task=self._task(w,i,from_layout,to_layout);w['pending']=task;picked.append(w)
  need(picked,'empty transaction batch')
  for w in picked:w['pending']['done'].wait()
  for w in picked:
   t=w['pending'];r=t['report']
   if t['error']or t['rc']not in self.native.backup_codes:self.poisoned=True;raise MaintenanceError('native backup state unknown; owner poisoned')
   if r.owner_poisoned:self.poisoned=True;raise MaintenanceError('native reports owner poisoned')
   need(t['rc']==0,'native transaction refused: '+str(t['rc'])+' '+bytes(r.error).decode(errors='replace'));
   try:self.native.check(r,t['txn'])
   except BaseException:self.poisoned=True;raise
  # V163: every native report above proves full-span RO restoration. Global
  # maps checks remain at initial/final/admission/run and unACKed recovery.
  bits=bytearray(self.active['bits'])
  for w in picked:
   i=w['pending']['index'];mask=1<<(i%8)
   if rollback:bits[i//8]&=~mask
   else:bits[i//8]|=mask
  self._checkpoint(bits,'BITMAP_BATCH_DURABLE')
  for w in picked:w['pending']=None
 def _restore_unacked(self):
  pending=self._pending()
  for w in pending:w['pending']['done'].wait()
  tasks=[]
  for w in pending:
   old=w['pending'];need(old['rc']in self.native.backup_codes and old['error']is None,'no known original backup for recovery')
   tasks.append((w,self._task(w,old['index'],0,0,restore=True,txn=old['txn'])))
  for w,t in tasks:t['done'].wait()
  for w,t in tasks:
   if t['report'].owner_poisoned:self.poisoned=True
   need(t['error']is None and t['rc']==0,'restore_saved failed; keep original scratch')
   try:self.native.check(t['report'],t['txn'],restore=True)
   except BaseException:self.poisoned=True;raise
  require_ro([self.inventory[w['pending']['index']]for w in pending]);self._checkpoint(self.active['bits'],'ROLLBACK_REQUIRED_UNADMISSIBLE')
  for w in pending:w['pending']=None
 def _batches(self,indices):
  local=[[i for i in indices if self.inventory[i]['numa_node']==node]for node in(0,1)]
  for start in range(0,max(map(len,local),default=0),4):yield local[0][start:start+4]+local[1][start:start+4]
 def _rollback(self):
  need(not self.poisoned,'poisoned owner cannot be reused');self.phase='ROLLBACK_REQUIRED_UNADMISSIBLE';self._restore_unacked()
  ids=[i for i in range(MATRICES)if self.active['bits'][i//8]&(1<<(i%8))]
  for batch in self._batches(ids):self._batch(batch,self.active['to'],self.active['from'],True)
  self._finish(self.active['from'],'ROLLED_BACK')
 def _finish(self,layout,phase):
  need(not self._pending(),'unACKed scratch at final publication');self._guard();self._proof_guard();need(physical_snapshot(self.inventory)==self.physical,'NUMA/THP/address conservation changed')
  a=self.active;target=self.format_epoch+1;need(target<1<<64,'format epoch overflow')
  future=self.binding();future.update(layout_tag=layout,format_epoch=target);future['owner_epoch']=MappingProxyType(future['owner_epoch']);future_field=MappingProxyType(future)
  self._checkpoint(a['bits'],'TRANSFORM_COMPLETE_VERIFIED')
  self.atomic(a['path']/'publication.json',dict(phase=phase,bitmap_hex=bytes(a['bits']).hex(),sequence=a['sequence'],from_layout=a['from'],to_layout=a['to'],layout_tag=layout,format_epoch=target,inventory_sha256=self.inventory_sha,owner_epoch=self.parent,owner_generation=self.generation))
  self.store.bank_format_binding_v146=future_field;self.store.bank_runtime_admissible_v146=True;self.layout=layout;self.format_epoch=target;self.phase='READY_OLD16'if layout==16 else'READY_PAIR139';self.active=None
 def maintain(self,command,cancel):
  self._guard();self._proof_guard();need(set(command)==COMMAND_KEYS and command['action']=='maintenance_relayout','exact private9-key command required')
  need(not self.poisoned,'owner poisoned; status/close only')
  need(command['expected_owner_epoch']==self.parent and type(command['expected_layout'])is int and command['expected_layout']==self.layout and type(command['expected_format_epoch'])is int and command['expected_format_epoch']==self.format_epoch and command['expected_inventory_sha256']==self.inventory_sha,'actual owner/layout/epoch/inventory differs')
  need(command['helper_build_receipt_sha256']==self.cfg['helper_build']['sha256']and command['bounded_native_proof_sha256']==self.cfg['bounded_native_proof']['sha256'],'actual helper/proof pins differ')
  op=command['operation_id'];need(isinstance(op,str)and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}',op),'invalid operation ID');direction=DIRECTIONS.get(command['direction']);need(direction is not None,'direction refused')
  continuation=self.active is not None
  if continuation:need(op==self.active['id']and direction==(self.active['to'],self.active['from']),'only exact same-operation reverse rollback continuation allowed')
  else:
   need(op not in self.seen and direction[0]==self.layout,'duplicate operation or wrong source format');self.seen.add(op);self.phase='MAINTENANCE_INTENT_DURABLE';self.store.bank_runtime_admissible_v146=False
   path=self.root/op;path.mkdir(mode=0o700);self.active=dict(id=op,path=path,from_=direction[0],to=direction[1],epoch=self.format_epoch,bits=bytearray(MATRICES//8),sequence=0);self.active['from']=self.active.pop('from_')
   self._checkpoint(self.active['bits'],'MAINTENANCE_INTENT_DURABLE')
  failure=None
  try:
   self._start()
   if continuation:self._rollback()
   else:
    ids=[i for i in range(MATRICES)if not(self.active['bits'][i//8]&(1<<(i%8)))]
    for batch in self._batches(ids):
     if cancel.is_set():raise MaintenanceError('cooperative cancel between native batches')
     self.phase='MATRIX_BATCH_RW_INFLIGHT';self._batch(batch,*direction,False)
    self._finish(direction[1],'COMMITTED')
  except BaseException as exc:
   failure=str(exc);self.store.bank_runtime_admissible_v146=False;self.phase='OWNER_POISONED_NO_REUSE'if self.poisoned else'ROLLBACK_REQUIRED_UNADMISSIBLE'
   if not continuation and not self.poisoned and self.workers_ready:
    try:self._rollback()
    except BaseException as rollback_error:failure+='; rollback unresolved: '+str(rollback_error)
  finally:
   self._stop()
   if self.active is None:self._free()
  result=self.status();result['operation_error']=failure;self.emit(json.dumps(result,sort_keys=True));return result
 def block_after_child_failure(self):
  self.child_drain_proved=False;self.store.bank_runtime_admissible_v146=False
  if self.active is None:self.phase='CHILD_DRAIN_UNPROVEN_NO_REUSE'
 def prepare_close(self):
  need(epoch()==self.parent,'only actual parent may join/free maintenance workers');self._stop()
 def close_after_bank(self):
  """Only after controller store.close has unmapped bank and all workers joined."""
  self._stop();need(self.store._closed,'cannot release unresolved scratch before bank unmap')
  for w in self.workspaces:
   if w['scratch']is not None:w['scratch'].close()
  self.workspaces=[]
