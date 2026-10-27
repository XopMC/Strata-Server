#!/usr/bin/env python3
"""Private pair foreground N1 supervisor. Metadata by default; unready profiles refuse execution."""
import argparse,contextlib,fcntl,hashlib,importlib,json,os,selectors,signal,socket,stat,struct,subprocess,sys,time,traceback
from pathlib import Path
from lifecycle import acquire_leases,validate_close_proof,write_exclusive
from maintenance_v146 import Manager,normalized_binding
from index_devices_v536 import resolve_current_pair
from admission_v536 import Registry,validate_manager_profile,revalidate_plan,runtime_plan,validate_descriptor,control_admission_pin
from bindings import HERE,M,Q,D,ROOT,MODEL,BANK_BYTES,OUT,FACTORY_ROOT,PRIOR_OWNER_STATE,need,read,sha,run_name,plan,profile_path,lease_paths,import_owner

# Publication admission only; original numerical/lifetime body follows unchanged.
PUBLICATION_FACTORY_ENABLED = False

def write(path,value):
 path=Path(path);temp=path.with_suffix('.partial')
 with temp.open('w')as f:json.dump(value,f,ensure_ascii=False,default=str,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def epoch(pid):
 raw=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(raw[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def sanitize():
 for k in list(os.environ):
  if k.startswith(('DS41_','DS1017_','DEEPSEEK41_','STRATA_','CUDA_')):del os.environ[k]
 os.environ.update(CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',DEEPSEEK41_NUMA_POLICY='interleave-both',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
def parent_cpu_guard():
 # This guard also runs DURING Manager146's eight owned maintenance workers.
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU parent CUDA mask')
 need('torch'not in sys.modules and not any(x in Path('/proc/self/maps').read_text()for x in('libcuda.so','libcudart.so','libtorch_cuda.so')),'CPU parent must stay framework/CUDA-free')
 need(os.sched_getaffinity(0)==set(range(256)),'outer physical mask0..255')
def parent_guard():
 # Keep actual startup/fork sites strict; no worker may survive into os.fork.
 parent_cpu_guard()
 need({int(p.name)for p in Path('/proc/self/task').iterdir()}=={os.getpid()},'fork requires exactly one parent thread')
def maintenance_exclusion(child):
 need(child is None,'owned child not reaped')
 parent_cpu_guard()
 # Manager146 performs the same child check and its own thread-aware single-
 # thread guard; the callback adds no blanket exception for any live child.
 for task in Path('/proc/self/task').glob('*/children'):
  need(not task.read_text().strip(),'maintenance parent has a child/unreaped child')
def numa_guard():
 r=subprocess.run(['/usr/bin/numactl','--show'],text=True,capture_output=True,check=True)
 v={s.split(':',1)[0].strip():s.split(':',1)[1].strip()for s in r.stdout.splitlines()if':'in s}
 need(v.get('policy')=='interleave'and set(v.get('interleavemask','').split())=={'0','1'},'launch under numactl --interleave=0,1');return v
def assert_pair_current(pair):
 need(tuple(resolve_current_pair())==tuple(pair),'current NVIDIA indices0/1 changed; close without selecting replacements')
def pair_idle(pair):
 assert_pair_current(pair)
 for row in pair:
  g=row['uuid']
  r=subprocess.run(['nvidia-smi','-i',g,'--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True,timeout=15)
  need(not r.stdout.strip(),'current index0/1 GPU already owned: '+g)
def pair_health(pair):
 pair_idle(pair)
 for row in pair:
  g=row['uuid']
  r=subprocess.run(['nvidia-smi','-i',g,'--query-gpu=uuid,gpu_recovery_action','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True,timeout=15)
  rows=[line.strip()for line in r.stdout.splitlines()if line.strip()]
  need(len(rows)==1,'one exact health row per ephemeral UUID');values=[x.strip()for x in rows[0].split(',')]
  need(len(values)==2 and values[0]==g and values[1]in('None','0'),'current pair recovery action must be0')
def require_prior_owner_closed():
 # Read-only refusal. Never attach, signal or close the benchmark/controller owner.
 if not PRIOR_OWNER_STATE.exists():return
 v=read(PRIOR_OWNER_STATE);old=v.get('maintenance',{}).get('bank_format',{}).get('owner_epoch')
 if old is None:old=v.get('parent')
 pid=v.get('owner_pid')if isinstance(v.get('owner_pid'),int)else old.get('pid')if isinstance(old,dict)else None
 if pid is None:return
 try:actual=epoch(pid)
 except FileNotFoundError:return
 need(isinstance(old,dict)and actual!=old,'existing CPU bank owner is still live; Root must close it explicitly before this cold launcher')

def signal_child(pid,pidfd,sig):
 if pid is None:return
 try:
  if pidfd is None:os.kill(pid,sig) # exact own still-unreaped fork only
  else:signal.pidfd_send_signal(pidfd,sig)
 except ProcessLookupError:pass

def serve_child(store,v,pair,generation,ready_fd,shutdown_seconds,idle_seconds,close_path):
 if not PUBLICATION_FACTORY_ENABLED:
  raise RuntimeError("DeepSeek publication factory is disabled: clean-build binding/reseal and cold lifecycle qualification are pending")
 # API/FIFO threads exist only in this fresh CUDA child, never CPU parent.
 import threading
 stop=threading.Event();drain=threading.Event()
 for sid in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sid,lambda *_:stop.set())
 signal.signal(signal.SIGUSR1,lambda *_:drain.set())
 assert_pair_current(pair)
 os.environ.update(CUDA_VISIBLE_DEVICES=','.join(row['uuid']for row in pair),CUDA_DEVICE_ORDER='PCI_BUS_ID',DS1017_OLD_COMPLETE_OWNER_V1='1',DS1017_OLD_COMPLETE_RUNTIME_V1='1',DS1017_CPU_GROUP_SUM_V2='0')
 v=revalidate_plan(v);factory_root=Path(v['factory_root'])
 sys.path[:0]=[str(factory_root),str(D/'Strata'),str(D/'tools'),str(HERE)]
 from serving_transport import make_owner_class,LoggedRuntime,load_transport
 from backend.deepseek41.fifo import InferenceOwner
 def log(phase,**fields):print(json.dumps(dict(phase=phase,time=time.time(),**fields),default=str),flush=True)
 def factory():
  selected=importlib.import_module('serving_runtime_v529');runtime=None
  need(Path(selected.__file__).resolve()==factory_root/'serving_runtime_v529.py','unique529 factory import')
  try:
   runtime=selected.create(store,context=v['context'])
   cfg=runtime.configuration();p=cfg['pair_runtime'];cpu=p['CPU_projection']
   need((cfg['context_capacity'],cfg['mtp'],cfg['kv_cache_type'])==(v['context'],'native4','bf16'),'selected runtime context/MTP/KV')
   need(p['cuda_visible_devices']==','.join(row['uuid']for row in pair)and p['CPU_group_sum']=='0','fresh current-index pair/A8 policy')
   need((cpu['path'],cpu['sha256'],cpu['mode'])==(v['CPU_projection']['path'],v['CPU_projection']['sha256'],'0'),'explicit CPU variant binding')
   from one_bank_store import source
   codec=source(Q/'research/compact-runtime-feature-closure-v7/features_v7.py','_pair_serving_bundle_codec').api_codec(store)
   need(codec.MODEL_ID==MODEL,'canonical bundle codec')
   wrapped=LoggedRuntime(runtime,stop,codec,log,sequence_base=generation<<32);wrapped.config.update(serving_pair_UUIDs=[row['uuid']for row in pair],serving_generation=generation,serving_variant=v['variant'],pair_API_proof_pending=True)
   return wrapped
  except BaseException:
   if runtime is not None:runtime.close()
   raise
 owner=make_owner_class(InferenceOwner)(factory,queue_limit=8);http=thread=None
 try:
  started=time.monotonic()
  while not owner.ready.wait(.25):
   if stop.is_set():return
   need(time.monotonic()-started<1800,'finite pair child startup')
  if stop.is_set():return
  if owner.startup_error:raise owner.startup_error
  owner.engine.activity=owner.touch_activity
  transport=load_transport(D,owner.engine.codec);http=transport.ApiServer(('0.0.0.0',49287),owner,context_capacity=v['context'],request_timeout_seconds=v['request_timeout_seconds'])
  thread=threading.Thread(target=http.serve_forever,name='pair-http',daemon=True);thread.start();owner.touch_activity()
  log('PAIR_API_LISTENING',address='http://0.0.0.0:49287',model=MODEL,context=v['context'],N=1,api_key_required=False,configuration=owner.engine.configuration())
  os.write(ready_fd,b'R');os.close(ready_fd);ready_fd=None
  while not stop.wait(.25):
   if drain.is_set():owner.begin_draining();drain.clear();log('API_ADMISSION_DRAINING')
   if owner.expire_if_idle(idle_seconds):log('API_INACTIVITY_EXPIRED',idle_seconds=idle_seconds);break
   need(owner.thread.is_alive()and not owner.closed.is_set(),'FIFO runtime stopped '+owner.phase)
 finally:
  owner.begin_draining(cancel_active=True)
  if thread is not None:http.shutdown();thread.join(5);need(not thread.is_alive(),'HTTP accept thread did not stop')
  if http is not None:http.server_close()
  owner.close(timeout=shutdown_seconds)
  if ready_fd is not None:os.close(ready_fd)
  need(owner.closed.is_set() and not owner.thread.is_alive(),'FIFO issuer not closed')
  need(type(store._native_leases)is int and store._native_leases==0,'child native leases not zero')
  write_exclusive(close_path,dict(schema=252,phase='PAIR_CHILD_CLOSED_ZERO_LEASES',parent=normalized_binding(store.bank_format_binding_v146)['owner_epoch'],child=epoch(os.getpid()),context=v['context'],runtime_library=v['library'],runtime_sha256=v['library_sha256'],bank_format_binding=normalized_binding(store.bank_format_binding_v146),native_leases=store._native_leases,FIFO_closed=True,quality_admitted=False))
  log('PAIR_CUDA_CHILD_CLOSED',close_proof=str(close_path))

def execute(args,v):
 if not PUBLICATION_FACTORY_ENABLED:
  raise RuntimeError("DeepSeek publication factory is disabled: clean-build binding/reseal and cold lifecycle qualification are pending")
 need(sys.dont_write_bytecode and Path(sys.prefix)==D/'.venv-reference','use existing reference Python -B')
 sanitize();parent_guard();numa=numa_guard();require_prior_owner_closed();pair=resolve_current_pair();health=lambda:pair_health(pair);name=run_name(args.run_id)
 OUT.mkdir(parents=True,exist_ok=True);folder=OUT/name;folder.mkdir(mode=0o700)
 state=dict(schema='deepseek1017-pair-N1-server-owner/536',actual_source_revision=565,initial_factory_source_revision=563,initial_native_source_revision=558,phase='PREFLIGHT',parent=epoch(os.getpid()),child=None,child_reaped=True,bank_closed=False,CPU_bank_bytes=0,context=v['context'],variant=v['variant'],profile=v['profile'],generation=0,selected_system_indices=[0,1],ephemeral_pair=list(pair),persistent_UUID_allowlist=False,NUMA=numa,API_actual_proof=False)
 stop=False;drain=False;next_plan=None;unload=False;child=pidfd=store=None;maintenance=None;registry=None;selected_id='initial563';old={};control=None
 runtime_dir=Path('/run/user')/str(os.getuid())
 st=runtime_dir.lstat();need(stat.S_ISDIR(st.st_mode)and st.st_uid==os.getuid()and not(st.st_mode&0o077),'private owned XDG runtime directory')
 control_dir=runtime_dir/'ds1017-pair-v565';control_dir.mkdir(mode=0o700,exist_ok=True)
 cs=control_dir.lstat();need(stat.S_ISDIR(cs.st_mode)and cs.st_uid==os.getuid()and not(cs.st_mode&0o077),'private control directory')
 control_path=control_dir/(str(os.getpid())+'-'+hashlib.sha256(name.encode()).hexdigest()[:16]+'.sock')
 need(len(os.fsencode(control_path))<108,'bounded AF_UNIX pathname')
 with (folder/'console.log').open('xb',buffering=0)as output:
  def relay(data):output.write(data);sys.stdout.write(data.decode('utf8','replace'));sys.stdout.flush()
  def log(phase,**fields):relay((json.dumps(dict(phase=phase,time=time.time(),**fields),default=str)+'\n').encode())
  def save():write(folder/'state.json',state)
  def handle_signal(sig,_):
   nonlocal stop,drain
   if sig==signal.SIGUSR1:drain=True
   else:stop=True
  for sid in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGUSR1):old[sid]=signal.signal(sid,handle_signal)
  try:
   with contextlib.ExitStack()as held:
    held.enter_context(acquire_leases(lease_paths(pair)))
    require_prior_owner_closed();health();parent_guard();save()
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as probe:
     probe.bind((v['host'],v['port']))
    # This bounded availability check is not a reservation; child bind still fails closed.
    proof=read(v['packer_proof']);need(sha(v['packer_proof'])==v['packer_proof_sha256']and proof['status']=='ACTUAL_OLD_OUTPUT16_V4_PACKER_BYTE_PARITY_PASS'and proof['all_four_shapes_exact']and proof['endpoint_cases']==2,'qualified packer proof')
    need(sha(v['packer'])==v['packer_sha256']==proof['library_sha256'],'packer identity')
    need(sha(v['payload_integrity_receipt'])==v['payload_integrity_receipt_sha256'],'prior full payload integrity receipt')
    owner=import_owner(v);os.environ['DS1017_OLD_COMPLETE_OWNER_V1']='1'
    class Cancel:
     def is_set(self):return stop
    last=0.
    def progress(row):
     nonlocal last
     if time.monotonic()-last>10:last=time.monotonic();state.update(phase='CPU_BANK_LOADING',progress=row);save();log('CPU_BANK_PROGRESS',progress=row)
    try:
     log('CPU_BANK_LOAD_BEGIN',bytes=BANK_BYTES)
     store=owner.OneBankStore(v['model_root'],v['model_manifest_sha256'])
     report=store.preload_selected(v['packer'],v['packer_sha256'],on_progress=progress,integrity_receipt=v['payload_integrity_receipt'],integrity_sha256=v['payload_integrity_receipt_sha256'],reserve_bytes=80*(1<<30),budget_bytes=432*(1<<30),workers=8,cancel=Cancel())
     need(report['resident_bytes']==BANK_BYTES and report.get('all_pack_workers_joined')is True,'one exact CPU bank with joined packers');store.validate_selected_owner(deep=True);parent_guard()
     def exclusion():
      maintenance_exclusion(child)
     maintenance=Manager(store,owner,v['maintenance_config'],v['maintenance_config_sha256'],folder/'maintenance',write,health,exclusion,emit=lambda line:log('BANK_MAINTENANCE',event=json.loads(line)),allow_profile_admission_v163=True,profile_validator_v163=validate_manager_profile)
     binding=maintenance.binding()
     command=dict(action='maintenance_relayout',operation_id='server-'+__import__('uuid').uuid4().hex,direction='old16_to_pair139',expected_owner_epoch=binding['owner_epoch'],expected_layout=binding['layout_tag'],expected_format_epoch=binding['format_epoch'],expected_inventory_sha256=binding['inventory_sha256'],helper_build_receipt_sha256=maintenance.cfg['helper_build']['sha256'],bounded_native_proof_sha256=maintenance.cfg['bounded_native_proof']['sha256'])
     log('BANK_139_RELAYOUT_BEGIN',owner=binding['owner_epoch'])
     relayout=maintenance.maintain(command,Cancel())
     need(relayout['phase']=='READY_PAIR139' and relayout['runtime_admissible'] is True and relayout['bank_format']['layout_tag']==139,'actual139 relayout did not commit')
     need(not stop,'stopped after cold preparation')
     state['bank_format_binding']=maintenance.binding()
     state.update(CPU_bank_bytes=BANK_BYTES,bank_identity=dict(parent=state['parent'],model_manifest_sha256=v['model_manifest_sha256']),progress=None);write(folder/'CPU-owner.json',report);save()
     control=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);control.bind(str(control_path));os.chmod(control_path,0o600);control.listen(4);control.setblocking(False)
     registry=Registry(maintenance,v,write,folder)
     def receive(conn,handler):
      with conn:
       conn.settimeout(1)
       try:
        _,uid,_=struct.unpack('3i',conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12));need(uid==os.getuid(),'control UID mismatch')
        raw=b''
        while not raw.endswith(b'\n')and len(raw)<=4096:
         part=conn.recv(4097-len(raw))
         if not part:break
         raw+=part
        need(len(raw)<=4096 and raw.endswith(b'\n')and raw.count(b'\n')==1,'bounded one-line control request')
        reply=handler(raw.decode('utf8').strip())
       except Exception as e:reply={'error':str(e)}
       try:conn.sendall((json.dumps(reply,default=str)+'\n').encode())
       except OSError:pass
     def bank_idle():
      nonlocal stop,selected_id
      parent_guard();health();maintenance._guard()
      state.update(phase='BANK_IDLE',admitted_profile_ids=sorted(registry.entries),control_socket=str(control_path));save();log('BANK_IDLE_RETAINED',CPU_bank_bytes=BANK_BYTES,bank_format_binding=maintenance.binding())
      chosen=None;idle_started=time.monotonic()
      def idle_command(text):
       nonlocal stop,chosen,selected_id
       if text=='status':return state
       if text in('stop','quit','exit'):stop=True;return {'accepted':'stop'}
       if text=='unload':return {'status':'ALREADY_UNLOADED','same_CPU_bank':True}
       need(not stop and chosen is None,'stop/select already pending')
       parts=text.split()
       if text.startswith('admit '):
        pin=control_admission_pin(text)
        result=registry.admit(pin);state['admitted_profile_ids']=sorted(registry.entries);save();log('RUNTIME_PROFILE_ADMISSION',**result);return result
       need(len(parts)==3 and parts[0]=='select'and parts[2]in('256','512'),'idle commands: status|stop|unload|admit JSON(path,sha256)|select ID 256/512')
       chosen=registry.select(parts[1],int(parts[2])*1024);selected_id=parts[1]
       return {'accepted':'select','profile_id':selected_id,'context':chosen['context'],'same_CPU_bank':True,'fresh_CUDA_child_required':True}
      idle_selector=selectors.DefaultSelector();idle_selector.register(control,selectors.EVENT_READ,'control')
      if sys.stdin.isatty():idle_selector.register(sys.stdin,selectors.EVENT_READ,'console')
      try:
       while not stop and chosen is None:
        if time.monotonic()-idle_started>=args.ttl_seconds:stop=True;log('BANK_IDLE_EXPIRED',idle_seconds=args.ttl_seconds);break
        for key,_ in idle_selector.select(.25):
         if key.data=='control':conn,_=control.accept();receive(conn,idle_command)
         else:
          line=sys.stdin.readline()
          if not line:idle_selector.unregister(sys.stdin);continue
          try:log('CONSOLE_CONTROL',response=idle_command(line.strip()))
          except Exception as e:log('CONTROL_REFUSED',error=str(e))
      finally:idle_selector.close()
      return chosen
     while not stop:
      if v is None:
       v=bank_idle()
       if stop:break
      parent_guard();health();v=revalidate_plan(v);store.validate_selected_owner(deep=True)
      if 'profile_descriptor'in v:
       descriptor=validate_descriptor(v['profile_descriptor'],maintenance.binding(),registry.base);libraries={int(c):(Path(r['library']['path']),r['library']['sha256'])for c,r in descriptor['contexts'].items()}
      else:libraries={n:(Path(plan(profile_path(528,n),execution=True)['library']),plan(profile_path(528,n),execution=True)['library_sha256'])for n in(262144,524288)}
      expected_bank=maintenance.assert_run(dict(runtime={'bank_format_binding':maintenance.binding()}),Path(v['adapter_root']),v['adapter_seal_sha256'],libraries)
      close_path=folder/('child-close-'+str(state['generation']+1)+'.json')
      log_read,log_write=os.pipe();ready_read,ready_write=os.pipe();sys.stdout.flush();sys.stderr.flush()
      child=os.fork()
      if child==0:
       control.close();os.close(log_read);os.close(ready_read);os.dup2(log_write,1);os.dup2(log_write,2);os.close(log_write)
       if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(line_buffering=True,write_through=True)
       try:serve_child(store,v,pair,state['generation']+1,ready_write,args.shutdown_seconds,args.ttl_seconds,close_path);code=0
       except BaseException:traceback.print_exc();code=1
       sys.stdout.flush();sys.stderr.flush();os._exit(code)
      maintenance.child_drain_proved=False
      os.close(log_write);os.close(ready_write);pidfd=os.pidfd_open(child)
      state.update(phase='CUDA_CHILD_STARTING',child=epoch(child),child_reaped=False,context=v['context'],variant=v['variant'],profile=v['profile'],selected_profile_id=selected_id,factory_root=v['factory_root'],native_source_evidence=v['native_source_evidence'],generation=state['generation']+1,control_socket=str(control_path));save()
      sel=selectors.DefaultSelector();sel.register(log_read,selectors.EVENT_READ,'log');sel.register(ready_read,selectors.EVENT_READ,'ready');sel.register(control,selectors.EVENT_READ,'control')
      if sys.stdin.isatty():sel.register(sys.stdin,selectors.EVENT_READ,'console')
      started=time.monotonic();stop_at=None;ready=False;next_plan=None;unload=False;exit_code=None
      def command(text):
       nonlocal stop,drain,next_plan,unload
       if text in('stop','quit','exit'):stop=True;return {'accepted':'stop'}
       if text=='drain':drain=True;return {'accepted':'drain'}
       if text=='status':return state
       if text=='unload':
        need(not stop and stop_at is None and next_plan is None,'shutdown/transition already active');unload=True;return {'accepted':'unload','same_CPU_bank':True}
       parts=text.split()
       need(len(parts)==2 and parts[0]=='switch'and parts[1]in('256','512'),'command: status|drain|stop|switch256/512 (with space)')
       need(not stop and stop_at is None and not unload,'shutdown/transition already active')
       context=int(parts[1])*1024
       if 'profile_descriptor'in v:
        descriptor=validate_descriptor(v['profile_descriptor'],maintenance.binding(),registry.base);new=runtime_plan(plan(profile_path(528,context),execution=True),descriptor,v['profile_descriptor'],context)
       else:new=plan(profile_path(v['variant'],context),execution=True)
       need(all(new[k]==v[k]for k in('model_root','model_manifest_sha256','owner_source','owner_sha256','packer','packer_sha256','payload_integrity_receipt','payload_integrity_receipt_sha256')),'context switch must retain same bank cohort')
       if new['context']==v['context']:return {'status':'ALREADY_CURRENT','context':v['context']}
       next_plan=new;return {'accepted':'context_switch','context':new['context'],'same_CPU_bank':True,'fresh_CUDA_child_required':True}
      try:
       while child is not None:
        now=time.monotonic()
        if drain:signal_child(child,pidfd,signal.SIGUSR1);drain=False
        if not ready and now-started>=1800:stop=True
        if (stop or next_plan is not None or unload)and stop_at is None:
         stop_at=now;state['phase']='UNLOAD_DRAINING'if unload else'CONTEXT_SWITCH_DRAINING'if next_plan else'CHILD_STOP_REQUESTED';save();signal_child(child,pidfd,signal.SIGTERM)
        if stop_at is not None and now-stop_at>args.shutdown_seconds+10:signal_child(child,pidfd,signal.SIGKILL)
        for key,_ in sel.select(.25):
         if key.data=='log':
          chunk=os.read(log_read,65536)
          if chunk:relay(chunk)
          else:sel.unregister(log_read)
         elif key.data=='ready':
          token=os.read(ready_read,1);sel.unregister(ready_read)
          if token==b'R':ready=True;state.update(phase='API_READY',ready_at=time.time(),inactivity_seconds=args.ttl_seconds);save();log('SERVER_READY',context=v['context'],variant=v['variant'],URL='http://0.0.0.0:49287')
         elif key.data=='console':
          line=sys.stdin.readline()
          if not line:sel.unregister(sys.stdin);continue
          try:log('CONSOLE_CONTROL',response=command(line.strip()))
          except Exception as e:log('CONTROL_REFUSED',error=str(e))
         else:conn,_=control.accept();receive(conn,command)
        pid,status=os.waitpid(child,os.WNOHANG)
        if pid:
         exit_code=os.waitstatus_to_exitcode(status);state.update(child=None,last_child=state['child'],child_reaped=True,child_exit_code=exit_code);child=None;save()
      finally:
       sel.close()
       # Only this still-unreaped fork can be signalled. Never unmap its bank.
       if child is not None:
        signal_child(child,pidfd,signal.SIGTERM);deadline=time.monotonic()+args.shutdown_seconds
        while child is not None:
         pid,status=os.waitpid(child,os.WNOHANG)
         if pid:state.update(child=None,child_reaped=True,child_exit_code=os.waitstatus_to_exitcode(status));child=None;save();break
         if time.monotonic()>=deadline:signal_child(child,pidfd,signal.SIGKILL)
         time.sleep(.1)
       if pidfd is not None:os.close(pidfd);pidfd=None
       os.set_blocking(log_read,False)
       while True:
        try:chunk=os.read(log_read,65536)
        except BlockingIOError:break
        if not chunk:break
        relay(chunk)
       os.close(log_read);os.close(ready_read)
      need(exit_code==0,'child failed/abnormal; no replacement or automatic replay')
      close=read(close_path);validate_close_proof(close,parent=state['parent'],child=state['last_child'],bank=expected_bank,context=v['context'],library=v['library'],library_sha=v['library_sha256'],exit_code=exit_code)
      maintenance.child_drain_proved=True
      health();maintenance._guard()
      state['child_close_proof']=dict(path=str(close_path),sha256=sha(close_path));save()
      if unload and not stop:
       v=None;unload=False;continue
      if next_plan is not None and not stop:
       need(exit_code==0,'context switch child did not close cleanly; replacement refused')
       log('CONTEXT_CHILD_REAPED_SAME_BANK',previous=v['context'],next=next_plan['context'],CPU_bank_bytes=BANK_BYTES)
       v=next_plan;next_plan=None;continue
      need(exit_code in(None,0),'owned CUDA child failed');break
    finally:
     # Also covers failure immediately after fork (e.g. pidfd/selector setup).
     if child is not None:
      signal_child(child,pidfd,signal.SIGTERM);deadline=time.monotonic()+args.shutdown_seconds
      while child is not None:
       pid,status=os.waitpid(child,os.WNOHANG)
       if pid:state.update(child=None,child_reaped=True,child_exit_code=os.waitstatus_to_exitcode(status));child=None;save();break
       if time.monotonic()>=deadline:signal_child(child,pidfd,signal.SIGKILL)
       time.sleep(.1)
     if pidfd is not None:os.close(pidfd);pidfd=None
     need(child is None,'bank must remain pinned until owned child reaped')
     if control is not None:control.close();control=None;control_path.unlink(missing_ok=True)
     if maintenance is not None:maintenance.prepare_close()
     if store is not None:
      store.close()
      if maintenance is not None:maintenance.close_after_bank()
      store=None;state.update(bank_closed=True,CPU_bank_bytes=0);save()
   state.update(phase='STOPPED',leases_released=True,finished=time.time());save();log('SERVER_STOPPED',bank_closed=True);return 0
  except BaseException as e:
   state.update(phase='FAILED',error=repr(e),finished=time.time());save();log('SERVER_FAILED',error=repr(e));raise
  finally:
   for sid,fn in old.items():signal.signal(sid,fn)

def control(run,text):
 f=OUT/run_name(run);v=read(f/'state.json');need(v['schema']=='deepseek1017-pair-N1-server-owner/536','owner state schema')
 try:need(epoch(v['parent']['pid'])==v['parent'],'stale parent epoch')
 except FileNotFoundError:
  print(json.dumps({'status':'ALREADY_EXITED','state':v}));return 0
 cmd=(Path('/proc')/str(v['parent']['pid'])/'cmdline').read_bytes().split(b'\0');need(str(HERE/'serve.py').encode()in cmd and b'--execute'in cmd and run.encode()in cmd,'not this supervisor')
 if not v.get('control_socket'):
  if text=='status':print(json.dumps(v));return 0
  need(text=='stop','startup allows only stop/status before control readiness')
  fd=os.pidfd_open(v['parent']['pid'])
  try:
   need(epoch(v['parent']['pid'])==v['parent'],'parent epoch changed');signal.pidfd_send_signal(fd,signal.SIGTERM)
  finally:os.close(fd)
  print(json.dumps({'accepted':'startup_stop'}));return 0
 with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)as s:s.settimeout(60);s.connect(v['control_socket']);s.sendall((text+'\n').encode());print(s.recv(65536).decode())
 return 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',type=Path);p.add_argument('--run-id');p.add_argument('--execute',action='store_true');p.add_argument('--ttl-seconds',type=int,default=7200);p.add_argument('--shutdown-seconds',type=int,default=600);p.add_argument('--control-run');p.add_argument('--command',default='status');p.add_argument('--no-hold-console',action='store_true');args=p.parse_args()
 if args.control_run:need(not args.execute,'control only');return control(args.control_run,args.command)
 need(args.profile is not None,'explicit profile required');need(1<=args.ttl_seconds<=7200 and 30<=args.shutdown_seconds<=600,'bounded TTL/shutdown')
 v=plan(args.profile,execution=args.execute)
 if not args.execute:print(json.dumps(v,ensure_ascii=False,indent=2));return 0
 need(args.run_id,'fresh ASCII run ID required')
 try:return execute(args,v)
 finally:
  if not args.no_hold_console and sys.stdin.isatty():
   try:input('Owned pair server finished. Press Enter to close console. ')
   except(EOFError,KeyboardInterrupt):pass
if __name__=='__main__':raise SystemExit(main())
