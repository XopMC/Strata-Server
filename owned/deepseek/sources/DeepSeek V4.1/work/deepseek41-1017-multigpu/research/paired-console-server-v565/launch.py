#!/usr/bin/python3
"""Private console launcher; no installation or start without explicit Root-admitted profile."""
import argparse,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from bindings import HERE,M,D,OUT,need,plan,profile_path

# Publication admission only; original numerical/lifetime body follows unchanged.
PUBLICATION_FACTORY_ENABLED = False

def environment():
 names=('HOME','USER','LOGNAME','PATH','LANG','LC_ALL','XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS')
 env={k:os.environ[k]for k in names if k in os.environ}
 env.update(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DEEPSEEK41_NUMA_POLICY='interleave-both')
 return env

def launch_plan(variant,context,*,execute=False):
 if not PUBLICATION_FACTORY_ENABLED:
  raise RuntimeError("DeepSeek publication factory is disabled: clean-build binding/reseal and cold lifecycle qualification are pending")
 profile=profile_path(variant,context*1024);v=plan(profile,execution=execute)
 rid=f'pair-{context}-v563-parent565-'+time.strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:8]
 unit='ds1017-pair-'+uuid.uuid4().hex[:12]
 command=['/usr/bin/systemd-run','--user','--scope','--collect','--unit='+unit,'--property=MemoryMax=512G','--property=MemorySwapMax=0','--property=TasksMax=1024','--property=AllowedCPUs=0-255','/usr/bin/numactl','--interleave=0,1','/usr/bin/taskset','-c','0-255',str(D/'.venv-reference/bin/python'),'-B',str(HERE/'serve.py'),'--profile',str(profile),'--run-id',rid,'--execute','--no-hold-console']
 return dict(schema='deepseek1017-pair-console-launch/536',actual_source_revision=565,initial_factory_source_revision=563,initial_native_source_revision=558,profile=v,run_id=rid,unit=unit,command=command,environment=environment(),cwd=str(M),log=str(OUT/rid/'console.log'),state=str(OUT/rid/'state.json'),status='SOURCE_PLAN_NOT_STARTED')

def main():
 a=argparse.ArgumentParser();a.add_argument('--variant',type=int,choices=(528,),default=528);a.add_argument('--context',type=int,choices=(256,512),required=True);a.add_argument('--start',action='store_true');args=a.parse_args()
 try:
  v=launch_plan(args.variant,args.context,execute=args.start)
  if not args.start:print(json.dumps(v,ensure_ascii=False,indent=2));return 0
  print(f"DeepSeek1017 | CPU + GPU0 + GPU1 | {args.context}k | parent565 / adapter563 / native558 (ABI{args.variant}) | N1 FIFO",flush=True)
  print('One435GB RAM bank. Direct LAN API0.0.0.0:49287 without key; 2h inactivity. Current indices0/1 resolved and leased afresh.',flush=True)
  print('Commands: status / drain / stop / switch 256 / switch 512 / unload; while idle: admit JSON, select ID 256|512. Same bank, fresh child.',flush=True)
  print('Logs: '+v['log'],flush=True)
  # Foreground scope inherits this terminal; native stdout/stderr always stay visible.
  stopping=False;requested=False;old={}
  def halt(*_):
   nonlocal stopping
   stopping=True
  for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):old[sig]=signal.signal(sig,halt)
  proc=None
  try:
   proc=subprocess.Popen(v['command'],cwd=v['cwd'],env=v['environment'])
   while proc.poll()is None:
    if stopping and not requested and Path(v['state']).is_file():
     try:
      import serve
      serve.control(v['run_id'],'stop');requested=True
     except(FileNotFoundError,KeyError,ValueError,OSError):pass
    time.sleep(.25)
   return proc.returncode
  finally:
   for sig,fn in old.items():signal.signal(sig,fn)
 except Exception as e:print('Pair launch refused/failed: '+str(e),file=sys.stderr,flush=True);return 1
 finally:
  if args.start and sys.stdin.isatty():
   try:input('Pair console finished. Press Enter to close. ')
   except(EOFError,KeyboardInterrupt):pass
if __name__=='__main__':raise SystemExit(main())
