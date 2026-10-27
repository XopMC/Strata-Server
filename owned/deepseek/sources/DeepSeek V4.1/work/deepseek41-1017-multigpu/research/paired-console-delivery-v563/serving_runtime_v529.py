"""Static source/native pin validation; create is fresh CUDA-child only.
No cold loader, bank allocation, owner attachment or model import on module import.
"""
from pathlib import Path
import hashlib,importlib,json,os,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CONTEXTS=(262144,524288)

# Publication admission only; original numerical/lifetime body follows unchanged.
PUBLICATION_FACTORY_ENABLED = False

def need(v,m):
 if not v:raise RuntimeError('delivery529: '+m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink(),'canonical regular pin '+str(p));need(sha(p)==r['sha256']and('bytes'not in r or p.stat().st_size==r['bytes']),'changed pin '+str(p));return p

def describe(*,execution=False):
 d=json.loads((HERE/'DELIVERY_BINDINGS.json').read_text());need(d['schema']==529 and set(d['contexts'])==set(map(str,CONTEXTS)),'explicit dual-context static delivery')
 need(d['source_revision']>=529 and d['native_revision']==528 and d['factory_module']=='serving_runtime_v529','explicit factory/native versions')
 need(d['native_contract']==dict(revision=528,prepare_symbol='ds528_prepare_request_v1',getter_symbol='ds528_prefill_plan_info',getter_words=16,geometry='generic-prefill-immutable-v528'),'exact typed native528 contract')
 if execution:need(d['ready']is True,'Root source/native528 bind/seal pending; API not admitted')
 pin(d['factory_source']);pin(d['native528_build']);pin(d['idle528']);pin(d['native528_source_pins']);pin(d['direct531_component']);pin(d['cohort539_component']);pin(d['cohort539_run']);pin(d['cohort539_source'])
 for context,row in d['contexts'].items():
  need(row['module_version']==(512 if context=='262144' else 506),'faithful numerical parent version')
  root=Path(row['adapter_root']);need(root==HERE/'source/adapters'/context and root.is_dir()and not root.is_symlink(),'static explicit derivative root')
  if execution:
   need(sha(root/'SEALED.json')==row['adapter_seal_sha256'],'derivative seal')
   for f in json.loads((root/'SEALED.json').read_text())['files']:pin(dict(f,path=str(root/f['path'])))
  for key in('library','CPU_projection','Task_library'):pin(row[key])
 return d

def create(store,context):
 if not PUBLICATION_FACTORY_ENABLED:
  raise RuntimeError("DeepSeek publication factory is disabled: clean-build binding/reseal and cold lifecycle qualification are pending")
 need(type(context)is int and context in CONTEXTS,'API256/512 only')
 bank=dict(getattr(store,'bank_format_binding_v146',{}));need(bank.get('layout_tag')==139 and bank.get('format_epoch',0)>=2 and getattr(store,'bank_runtime_admissible_v146',False)is True,'Manager actual inherited139 admissibility required')
 need(bank.get('owner_generation')==store.generation and bank.get('owner_epoch')==store._owner_epoch and bank['owner_epoch']['pid']!=os.getpid(),'fresh inherited child; no PID attach/reload/current-owner reinterpretation')
 need('torch'not in sys.modules,'fresh API child must select current indices before framework import')
 d=describe(execution=True);entry=d['contexts'][str(context)];source=Path(entry['adapter_root'])/'source';version=entry['module_version']
 sys.path.insert(0,str(source))
 from cohort_binding_v539 import WORKERS, ROLE_CPUS, RESERVED
 allowed=os.sched_getaffinity(0);inference_cpus=set(WORKERS)|set(ROLE_CPUS)
 need(inference_cpus<=allowed and not(inference_cpus&set(RESERVED)),'all inference CPUs admitted, upload cores excluded')
 os.sched_setaffinity(0,inference_cpus)
 need(os.sched_getaffinity(0)==inference_cpus,'fresh child must inherit only252 non-upload physical cores')
 # Same qualified adapter index resolver: fresh NVML indices0/1, then exact CUDA
 # ordinal/native501 check. No fixed UUID table or PCI identity in this delivery.
 stage=importlib.import_module(f'stage_devices_pair_v{version}');stage.activate_pair()
 binding=importlib.import_module('serving_binding_v529');need(Path(binding.__file__).resolve()==source/'serving_binding_v529.py','dedicated fresh child per context; no module collision')
 binding.configure(store,version)
 factory=importlib.import_module(f'pair_factory_v{version}');need(Path(factory.__file__).resolve()==source/f'pair_factory_v{version}.py','explicit derivative factory source')
 runtime=factory.create(store,context=context,library_path=entry['library']['path'],library_sha256=entry['library']['sha256'],mtp='native4',pair_cache_mode='off',expert_cache_gib=0,packed_ram_gib=448,pair_prefill='0',gu_hidden='0',CPU_group_sum='0')
 configuration=runtime.configuration();need(configuration['bank_format_binding_v146']==bank and configuration['context_capacity']==context,'actual runtime must report exact fresh owner capability and context')
 cpu=configuration['pair_runtime']['CPU_projection'];need((cpu['path'],cpu['sha256'],cpu['mode'])==(entry['CPU_projection']['path'],entry['CPU_projection']['sha256'],'0'),'actual selected CPU418')
 return runtime
