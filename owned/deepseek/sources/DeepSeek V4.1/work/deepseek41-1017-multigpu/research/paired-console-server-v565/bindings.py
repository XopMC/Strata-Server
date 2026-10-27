"""Actual565 / family536 stdlib metadata only. No model, factory, DSO or device import/query."""
import hashlib,json,os,re,sys
from content_cache_v565 import CONTENT_HASH_CACHE
from delivery_contract_v536 import header as check_delivery_header,receipts as check_native_receipts,provenance_revision
from pathlib import Path
HERE=Path(__file__).resolve().parent;M=HERE.parents[1];Q=M.parent/'deepseek41-1017-nvfp4';D=M.parent/'deepseek41';ROOT=M.parents[2]
MODEL='deepseek-v4.1-flash-1017-gentle-heretic-mixed';BANK_BYTES=435110212200
ACTUAL_SOURCE_REVISION=565
OUT=ROOT/'DeepSeek V4.1/outputs/deepseek41-1017-multigpu/pair-serving-v565'
FACTORY_ROOT=M/'research/paired-console-delivery-v563';FACTORY_MANIFEST=FACTORY_ROOT/'DELIVERY_BINDINGS.json'
PRIOR_OWNER_STATE=ROOT/'DeepSeek V4.1/outputs/deepseek41-1017-multigpu/pair-serving-v536/pair-544-recovery-20261026/state.json'
def need(v,m):
 if not v:raise ValueError(m)
def read(p,limit=8<<20):
 p=Path(p);need(p.is_file()and not p.is_symlink()and p.stat().st_size<=limit,'bounded regular metadata '+str(p));return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(r,*,content=True):
 p=Path(r['path']);need(p.is_absolute()and p.resolve()==p and p.is_file()and not p.is_symlink(),'canonical regular pin '+str(p))
 need('bytes'not in r or p.stat().st_size==r['bytes'],'pinned size changed '+str(p))
 if content:
  # Only a supplied expected digest is memoized. sha(path) stays uncached.
  CONTENT_HASH_CACHE.verify_project(dict(path=str(p),sha256=r['sha256'],bytes=r.get('bytes',p.stat().st_size)))
 return p
def run_name(v):
 need(isinstance(v,str)and v.isascii()and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',v),'ASCII run ID');return v
def profile_path(variant,context):
 need(type(variant)is int and variant==528,'explicit serving528 only');need(type(context)is int and context in(262144,524288),'pair256/512 only')
 return HERE/'profiles'/f'v{variant}-{context}.json'
def source_guard():
 frozen=read(HERE/'SOURCE_READY.json')
 need(frozen['phase']=='SOURCE536_COLD_PARENT_FROZEN_NOT_STARTED','536 source freeze')
 need(frozen.get('actual_source_revision')==ACTUAL_SOURCE_REVISION,'actual565 source provenance required')
 for row in frozen['files']:pin(row)
 from check_source_v565 import check
 need(check()['phase']=='SOURCE536_LITERAL_LIFECYCLE_AND_CURRENT_INDEX_AST_PASS','536 source/AST guard')
 return frozen

def native_provenance(v,root,build,idle):
 source_pin=v['native528_source_pins'];need(set(source_pin)=={'path','bytes','sha256'},'explicit frozen native source pin')
 source_path=pin(source_pin);need(source_path.name=='SOURCE_READY.json'and source_path.is_relative_to(M/'research'),'native source freeze inside canonical research')
 source=read(source_path);members=source.get('files');need(isinstance(members,list)and 1<=len(members)<=512,'bounded actual native source closure')
 for member in members:pin(member)
 version=provenance_revision(v,build,source,initial=root==FACTORY_ROOT)
 need(build.get('source_ready_sha256',build.get('source_freeze_sha256'))==source_pin['sha256'],'Root build belongs to another native source freeze')
 recipe=Path(source['root_recipe']);need(recipe.is_absolute()and recipe.is_relative_to(M/'research'),'explicit canonical native Root recipe')
 recipe_pin=dict(path=str(recipe),bytes=recipe.stat().st_size,sha256=build['source_recipe_sha256']);pin(recipe_pin)
 need(recipe_pin in members,'Root recipe not in frozen native source closure')
 commands=read(recipe)['commands'];need(isinstance(commands,list)and len(commands)==len(build['steps'])and all(step.get('argv')==command.get('argv')and isinstance(command.get('argv'),list)and command['argv']and all(isinstance(arg,str)for arg in command['argv'])for step,command in zip(build['steps'],commands)),'actual Root build argv differs from frozen recipe')
 need(idle.get('build')==v['native528_build'],'idle receipt belongs to another actual build')
 tool=idle['tool'];need(set(tool)=={'path','bytes','sha256'}and Path(tool['path']).is_relative_to(M),'explicit canonical Root idle-probe provenance');pin(tool)
 return dict(native_source_revision=version,source_freeze=source_pin,root_recipe=recipe_pin,build=v['native528_build'],idle=v['idle528'],idle_probe_tool=tool)

def delivery(*,execution=False,factory_root=None):
 """Read only529's explicit JSON handoff; NEVER import its factory in parent."""
 root=FACTORY_ROOT if factory_root is None else Path(factory_root)
 need(root.is_absolute()and root.resolve()==root and root.parent==M/'research'and re.fullmatch(r'paired-console-delivery-v[0-9]+',root.name)and not root.is_symlink(),'canonical frozen serving delivery root only')
 manifest=root/'DELIVERY_BINDINGS.json'
 v=read(manifest);check_delivery_header(v)
 if root==FACTORY_ROOT:need(v.get('source_revision')==563 and v.get('native_source_revision')==558,'565 initial runtime must be actual563/native558')
 need(set(v['contexts'])=={'262144','524288'},'two explicit serving contexts')
 need(Path(v['factory_source']['path'])==root/'serving_runtime_v529.py','unique529 factory source')
 pin(v['factory_source'])
 need(Path(v['source_manifest']['path'])==root/'SOURCE_MANIFEST.json','explicit delivery source freeze')
 frozen=read(pin(v['source_manifest']));need(isinstance(frozen.get('files'),list)and 1<=len(frozen['files'])<=512,'bounded factory source freeze')
 for member in frozen['files']:pin(member)
 for key in('native528_build','idle528'):pin(v[key])
 build=read(v['native528_build']['path']);idle=read(v['idle528']['path'])
 check_native_receipts(build,idle,v['contexts'])
 native_evidence=native_provenance(v,root,build,idle)
 checked_dependencies={};system_links={}
 for c,row in v['contexts'].items():
  packet=Path(row['adapter_root']);need(packet.is_absolute()and packet.resolve()==packet and packet==root/'source/adapters'/c and packet.is_dir()and not packet.is_symlink(),'unique private529 derivative')
  need(row['library']==build['artifacts'][row['library']['path']],'actual528 library row')
  need(sha(packet/'SEALED.json')==row['adapter_seal_sha256'],'529 derivative seal changed')
  for member in read(packet/'SEALED.json')['files']:
   rel=Path(member['path']);need(not rel.is_absolute()and '..'not in rel.parts,'safe derivative seal member');pin(dict(member,path=str(packet/rel)))
  dependencies=read(packet/'DEPENDENCY_PINS.json').get('files');need(isinstance(dependencies,list)and 1<=len(dependencies)<=4096,'bounded full asset/source/native closure')
  if execution:
   for dependency in dependencies:
    previous=checked_dependencies.get(dependency['path']);need(previous is None or previous==dependency,'conflicting closure pin')
    if previous is None:
     dependency_path=Path(dependency['path'])
     if dependency_path.is_symlink()or dependency_path.resolve()!=dependency_path:
      from dependency_pins_v536 import verify_installed_dependency
      system_links[dependency['path']]=verify_installed_dependency(dependency)
     else:pin(dependency)
     checked_dependencies[dependency['path']]=dependency
   pin(row['library'])
 need(len({x['adapter_root']for x in v['contexts'].values()})==2,'distinct percontext derivatives')
 for name in('CPU_projection','Task_library'):
  values=[r[name]for r in v['contexts'].values()];need(values[0]==values[1],'same '+name+' across context derivatives');pin(values[0],content=execution)
 v['_system_dependency_links']=system_links
 v['_native_source_evidence']=native_evidence
 return v

def plan(path,*,execution=False):
 path=Path(path).absolute();v=read(path)
 need(v['schema']=='deepseek1017-pair-N1-serving-profile/536'and path==profile_path(v['variant'],v['context']),'canonical536 profile')
 need(v.get('actual_source_revision')==ACTUAL_SOURCE_REVISION and v.get('initial_factory_source_revision')==563 and v.get('initial_native_source_revision')==558,'actual565 initial profile provenance')
 need(v['model_id']==MODEL and v['selected_system_indices']==[0,1]and v['persistent_UUID_allowlist']is False,'current-index model/pair contract')
 need('gpu_uuids'not in v and 'GPU_PCIs'not in v,'no persistent hardware identities')
 need((v['host'],v['port'],v['api_key_required'],v['inference_concurrency'],v['queue_limit'],v['TTL_seconds'])==('0.0.0.0',49287,False,1,8,7200),'literal N1 LAN/TTL contract')
 need((v['CPU_bank_bytes'],v['packed_ram_gib'],v['pack_workers'],v['NUMA_policy'],v['mtp'],v['expert_cache_gib'])==(BANK_BYTES,432,8,'interleave-both','native4',0),'one-bank native4 contract')
 need(v.get('bank_layout')==139 and v['CPU_group_sum']=='0'and v['request_timeout_seconds']==1800,'typed139/originalA8/bounded request')
 expected=Q/'research/old-complete-runtime-1017-adapter-v1/one_bank_store.py';need(Path(v['owner_source'])==expected and sha(expected)==v['owner_sha256'],'canonical cold bank owner')
 need(Path(v['factory_manifest'])==FACTORY_MANIFEST,'explicit529 handoff only')
 source_guard();d=delivery(execution=execution);selected=d['contexts'][str(v['context'])]
 for key in('adapter_root','adapter_seal_sha256','library'):need(v[key]==selected[key],'selected529/528 profile '+key)
 need(v['factory_manifest_sha256']==sha(FACTORY_MANIFEST),'529 manifest changed')
 need(v['CPU_projection']==selected['CPU_projection']and v['Task_library']==selected['Task_library'],'CPU418/Task489 binding')
 if execution:
  need(v['ready']is True and isinstance(v['serving_admission'],dict),'Root536 source/API admission pending')
  a=read(pin(v['serving_admission']));need(a['phase']in('ROOT_SOURCE_ADMITTED_FOR_PAIR536_API_PROOF','ACTUAL_PAIR536_API_LIFECYCLE_PASS'),'Root536 admission phase')
  need(a.get('actual_source_revision')==ACTUAL_SOURCE_REVISION,'Root admission must qualify actual565 source')
  need(a['factory_manifest_sha256']==v['factory_manifest_sha256']and a['source_ready_sha256']==sha(HERE/'SOURCE_READY.json'),'actual source/factory admission')
  need(a['contexts'][str(v['context'])]==selected,'actual context admission')
  need(a['native_source_evidence']==d['_native_source_evidence'],'qualified native source/build/idle provenance changed')
  need(a['system_dependency_links']==d['_system_dependency_links'],'installed dependency link/target/content/stat changed since Root source admission')
  need(sha(v['maintenance_config'])==v['maintenance_config_sha256'],'536 maintenance config changed')
 return {**v,'library':v['library']['path'],'library_sha256':v['library']['sha256'],'profile':str(path),'factory_root':str(FACTORY_ROOT),'system_dependency_links':d['_system_dependency_links'],'native_source_evidence':d['_native_source_evidence'],'status':'EXPLICIT_PAIR536_PLAN_NOT_API_PROOF','one_CPU_preload':True,'fresh_CUDA_child_per_context':True}

def lease_paths(pair):
 need(len(pair)==2 and [r['index']for r in pair]==[0,1],'fresh selected pair required for leases')
 return [Q/'state/owner.lock',M/'state/coordinator.lock',D/'state/selected-speed-owner-v289.lock',D/'state/selected-server-v270.lock',ROOT/'Swift Flash Next MultiGPU/work/swift-flash-next-multigpu/state/coordinator.lock',ROOT/'Qwen3 Coder Next/work/qwen3-coder-next-speed/state/GPU0-probe.lock',ROOT/'RVN Qwen3.8 Flash Next/outputs/qwenflash-next-speed/launchers-speed-v10/speed-v10-console.lock',*[ROOT/'Shared/work/strata-server/state/devices'/(g['uuid']+'.lock')for g in pair],M/'state/pair-controller-v1.lock']
def import_owner(v):
 need('torch'not in sys.modules,'cold CPU parent must import owner before framework')
 p=Path(v['owner_source']);need(sha(p)==v['owner_sha256'],'canonical owner changed')
 sys.path.insert(0,str(p.parent));import one_bank_store
 need(Path(one_bank_store.__file__).resolve()==p,'wrong inherited owner module');return one_bank_store
