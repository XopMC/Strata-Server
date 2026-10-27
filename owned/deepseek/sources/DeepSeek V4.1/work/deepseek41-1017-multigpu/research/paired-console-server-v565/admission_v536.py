"""Finite stdlib serving profiles. Never imports candidate source or any DSO."""
import copy,json,re,time
from pathlib import Path
import bindings as b
from cpu_expected_preflight_v512_r2 import check_expected

CONTEXTS=('262144','524288')
COHORT=('model_id','model_root','model_manifest_sha256','owner_source','owner_sha256','packer','packer_sha256','packer_proof','packer_proof_sha256','payload_integrity_receipt','payload_integrity_receipt_sha256','CPU_bank_bytes','NUMA_policy','bank_layout','CPU_group_sum','mtp','selected_system_indices')
DESCRIPTOR_KEYS={'schema','profile_id','parent_source_ready','bank_format_binding','bank_cohort','factory_root','factory_manifest','factory_source_manifest','contexts','cpu_expected','format_profiles','source_admission_proof','native_source_evidence','quality_admitted','performance_admitted'}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def row(p):
 p=Path(p);return dict(path=str(p),sha256=b.sha(p),bytes=p.stat().st_size)
def path(value,*,directory=False):
 p=Path(value);b.need(isinstance(value,str)and p.is_absolute()and p.resolve()==p and p.is_relative_to(b.M/'research')and not p.is_symlink(),'canonical profile path inside M/research required')
 b.need(p.is_dir()if directory else p.is_file(),'profile path missing');return p
def pinned(pin):
 b.need(isinstance(pin,dict)and set(pin)=={'path','sha256','bytes'},'exact path/SHA/size pin')
 p=path(pin['path']);b.pin(pin);return p
def control_admission_pin(text):
 b.need(text.startswith('admit '),'explicit admit JSON required');v=json.loads(text[6:])
 b.need(isinstance(v,dict)and set(v)=={'path','sha256'}and isinstance(v['sha256'],str)and re.fullmatch(r'[0-9a-f]{64}',v['sha256']),'admit requires exact path/SHA JSON')
 p=path(v['path']);b.need(p.stat().st_size<=8<<20,'bounded descriptor metadata');r=row(p);b.need(r['sha256']==v['sha256'],'explicit descriptor SHA differs');return r
def cohort(v):return {k:v[k]for k in COHORT}
def cpu_record(d):
 r=d['contexts']['524288']['CPU_projection']
 return dict(path=r['path'],sha256=r['sha256'],mode='0',proof_scope='synthetic-correctness-only',model_pending=True,quality_admitted=False,model_speed_admitted=False,performance_admitted=False)
def check_cpu(expected,d):
 b.need(expected==cpu_record(d),'explicit418 CPU expected record changed');check_expected(expected,dict(CPU_group_sum='0'))

def validate_manager_profile(profile):
 """Manager163 callback: explicit serving/native528 schema, not old497 whole checker."""
 packet=path(profile['adapter_root'],directory=True)
 b.need(packet.name in CONTEXTS and packet.parent.name=='adapters'and packet.parent.parent.name=='source','two typed serving derivative roots only')
 root=packet.parents[2];d=b.delivery(execution=True,factory_root=root);selected=d['contexts'][packet.name]
 b.need(str(packet)==selected['adapter_root']and profile['adapter_seal_sha256']==selected['adapter_seal_sha256'],'exact derivative seal')
 libraries={c:{k:d['contexts'][c]['library'][k]for k in('path','sha256')}for c in CONTEXTS}
 b.need(profile['libraries']==libraries and profile['layout_tag']==139,'actual528 both-context libraries and139 required')
 build=d['native528_build'];b.need(profile['pair_build']==dict(path=build['path'],sha256=build['sha256'],contract={'phase':'ROOT528_NATIVE_BUILD_PASS_API_PENDING'}),'actual528 build receipt contract')
 proof=profile['proof'];b.need(set(proof)=={'path','sha256','contract'},'explicit source admission proof pin')
 p=path(proof['path']);b.need(b.sha(p)==proof['sha256'],'source admission proof changed');v=b.read(p)
 expected=dict(phase='ROOT536_RUNTIME_PROFILE_SOURCE_ADMITTED_API_PENDING',quality_admitted=False,performance_admitted=False)
 b.need(proof['contract']==expected and all(v.get(k)==x for k,x in expected.items()),'Root536 source admission proof scope')
 b.need(v['bank_format_binding']==profile['bank_format_binding']and v['factory_manifest']==row(root/'DELIVERY_BINDINGS.json')and v['contexts']==d['contexts'],'proof current bank/factory/context closure')
 b.need(v['parent_source_ready']==row(b.HERE/'SOURCE_READY.json'),'proof same frozen536 parent')
 check_cpu(v['cpu_expected'],d)
 b.need(v['native_source_evidence']==d['_native_source_evidence'],'profile native source/build/idle proof changed')
 b.need(v['system_dependency_links']==d['_system_dependency_links'],'source proof installed dependency evidence changed')
 b.need(profile['quality_admitted']is False and profile['performance_admitted']is False,'no invented quality/performance admission')

def validate_descriptor(pin,binding,base):
 p=pinned(pin);v=b.read(p);b.need(set(v)==DESCRIPTOR_KEYS and v['schema']=='deepseek1017-serving-admission/536','exact serving536 descriptor')
 ident=v['profile_id'];b.need(isinstance(ident,str)and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}',ident),'bounded ASCII profile ID')
 b.need(canonical(v['bank_format_binding'])==canonical(binding),'descriptor binds another live bank epoch/inventory')
 b.need(binding.get('layout_tag')==139 and v['bank_cohort']==cohort(base),'same model/manifest/owner/bank/NUMA/pair cohort only')
 b.need(v['parent_source_ready']==row(b.HERE/'SOURCE_READY.json'),'descriptor different frozen536 parent')
 root=path(v['factory_root'],directory=True);d=b.delivery(execution=True,factory_root=root)
 b.need(v['factory_manifest']==row(root/'DELIVERY_BINDINGS.json')and v['factory_source_manifest']==row(root/'SOURCE_MANIFEST.json')and v['contexts']==d['contexts'],'immutable whole delivery closure')
 b.need(v['quality_admitted']is False and v['performance_admitted']is False,'admission cannot assert model quality/performance')
 b.need(v['native_source_evidence']==d['_native_source_evidence'],'descriptor native source/build/idle closure changed')
 for selected in d['contexts'].values():
  b.need(selected['CPU_projection']==base['CPU_projection']and selected['Task_library']==base['Task_library'],'runtime guard upgrades preserve accepted CPU418/Task489 math')
 b.need(d['_system_dependency_links']==base['system_dependency_links'],'candidate installed dependency evidence differs from current qualified parent')
 check_cpu(v['cpu_expected'],d);pinned(v['source_admission_proof'])
 b.need(set(v['format_profiles'])==set(CONTEXTS),'both163 profiles required before selection')
 for c in CONTEXTS:
  fp=b.read(pinned(v['format_profiles'][c]));validate_manager_profile(fp)
  b.need(fp['bank_format_binding']==binding and fp['profile_id']==ident+'-'+c and fp['adapter_root']==d['contexts'][c]['adapter_root'],'profile163 identity/binding/context')
  b.need(fp['proof']['path']==v['source_admission_proof']['path']and fp['proof']['sha256']==v['source_admission_proof']['sha256'],'same explicit Root source proof')
 return v

def runtime_plan(base,descriptor,pin,context):
 b.need(str(context)in CONTEXTS,'256/512 only');r=copy.deepcopy(base);selected=descriptor['contexts'][str(context)];r.update(**selected)
 r.update(context=context,factory_root=descriptor['factory_root'],factory_manifest=descriptor['factory_manifest']['path'],factory_manifest_sha256=descriptor['factory_manifest']['sha256'],library=selected['library']['path'],library_sha256=selected['library']['sha256'],admitted_profile_id=descriptor['profile_id'],profile_descriptor=pin,admission_bank_binding=descriptor['bank_format_binding'],native_source_evidence=descriptor['native_source_evidence'],status='EXPLICIT_ROOT536_ADMITTED_PROFILE_API_PENDING')
 return r

def revalidate_plan(v):
 if 'profile_descriptor'not in v:return b.plan(v['profile'],execution=True)
 base=b.plan(b.profile_path(528,v['context']),execution=True)
 d=validate_descriptor(v['profile_descriptor'],v['admission_bank_binding'],base)
 checked=runtime_plan(base,d,v['profile_descriptor'],v['context']);b.need(checked==v,'selected runtime plan changed');return checked

class Registry:
 """Current process only. No journal replay into another parent/bank epoch."""
 def __init__(self,manager,base,atomic,state_dir):
  self.manager=manager;self.base=copy.deepcopy(base);self.atomic=atomic;self.root=Path(state_dir);self.entries={};self.initial='initial563'
  self.entries[self.initial]=dict(initial=True,plans={c:b.plan(b.profile_path(528,int(c)),execution=True)for c in CONTEXTS})
 def _idle(self):
  b.need(self.manager.child_drain_proved,'child close/drain not proved');self.manager._guard()
  b.need(not self.manager.threads and self.manager.active is None and not self.manager.poisoned and not self.manager._pending(),'admission/select requires fully idle bank')
 def admit(self,pin):
  self._idle();d=validate_descriptor(pin,self.manager.binding(),self.base);ident=d['profile_id'];prior=self.entries.get(ident)
  if prior is not None:
   b.need(not prior.get('initial')and prior['pin']==pin,'profile ID already bound to different immutable descriptor')
   return dict(status='ALREADY_ADMITTED',profile_id=ident)
  b.need(len(self.entries)<33,'at most32 additional dual-context revisions')
  for c in CONTEXTS:
   fp=d['format_profiles'][c]
   self.manager._read_profile_v163(dict(path=fp['path'],sha256=fp['sha256']),self.manager.binding())
  # Both closures validate first. Partial Manager ACKs are inert, and an exact
  # retry can complete them; no bank rollback/poison or selectable half-profile.
  existing={x['profile_id']:x for x in self.manager._extra_profiles_v163()}
  for c in CONTEXTS:
   fp=d['format_profiles'][c];profile=b.read(fp['path']);old=existing.get(profile['profile_id'])
   if old is not None:
    b.need(old==profile,'partial admission profile ID differs')
    exact=next(x for x in self.manager.extra_profiles_v163 if x[0]==profile['profile_id'])
    b.need(exact[1:3]==(fp['path'],fp['sha256']),'partial admission pin differs')
   else:self.manager.admit_profile_v163(dict(action='admit_format_profile',profile_path=fp['path'],profile_sha256=fp['sha256'],expected_bank_format_binding=self.manager.binding()))
  self.atomic(self.root/('runtime-profile-'+ident+'.json'),dict(schema=536,phase='BOTH_CONTEXTS_SOURCE_ADMITTED_NOT_API_PROVED',descriptor_pin=pin,bank_format_binding=self.manager.binding(),time=time.time(),quality_admitted=False,performance_admitted=False))
  self.entries[ident]=dict(pin=copy.deepcopy(pin));return dict(status='ADMITTED_BOTH_CONTEXTS',profile_id=ident)
 def select(self,ident,context):
  self._idle();b.need(ident in self.entries,'profile not completely admitted');entry=self.entries[ident]
  if entry.get('initial'):return b.plan(entry['plans'][str(context)]['profile'],execution=True)
  d=validate_descriptor(entry['pin'],self.manager.binding(),self.base)
  return runtime_plan(b.plan(b.profile_path(528,context),execution=True),d,entry['pin'],context)

def prepare(factory_root,owner_state,ident,output):
 """Root metadata preparation only, from actual live parent status JSON."""
 output=Path(output);b.need(output.is_absolute()and output.parent.resolve()==output.parent and output.is_relative_to(b.M/'research')and not output.exists(),'fresh canonical research descriptor directory')
 b.need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}',ident)is not None,'profile ID')
 state=b.read(owner_state);b.need(state['schema']=='deepseek1017-pair-N1-server-owner/536'and state['phase']=='BANK_IDLE'and state['child']is None and state['child_reaped']is True and not state['bank_closed'],'current unloaded536 owner required')
 binding=state['bank_format_binding'];base=b.plan(b.profile_path(528,262144),execution=True);d=b.delivery(execution=True,factory_root=factory_root);root=Path(factory_root)
 b.need(state['bank_identity']['model_manifest_sha256']==base['model_manifest_sha256'],'same model manifest')
 expected=cpu_record(d);check_cpu(expected,d);output.mkdir(mode=0o700)
 def put(p,v):
  with p.open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
 proof=dict(phase='ROOT536_RUNTIME_PROFILE_SOURCE_ADMITTED_API_PENDING',parent_source_ready=row(b.HERE/'SOURCE_READY.json'),bank_format_binding=binding,factory_manifest=row(root/'DELIVERY_BINDINGS.json'),contexts=d['contexts'],system_dependency_links=d['_system_dependency_links'],native_source_evidence=d['_native_source_evidence'],cpu_expected=expected,quality_admitted=False,performance_admitted=False)
 proof_path=output/'source-proof.json';put(proof_path,proof);profiles={};libraries={c:{k:d['contexts'][c]['library'][k]for k in('path','sha256')}for c in CONTEXTS}
 for c in CONTEXTS:
  selected=d['contexts'][c];build=d['native528_build'];fp=dict(schema='deepseek1017-format-profile/163',profile_id=ident+'-'+c,bank_format_binding=binding,layout_tag=139,adapter_root=selected['adapter_root'],adapter_seal_sha256=selected['adapter_seal_sha256'],pair_build=dict(path=build['path'],sha256=build['sha256'],contract={'phase':'ROOT528_NATIVE_BUILD_PASS_API_PENDING'}),proof=dict(path=str(proof_path),sha256=b.sha(proof_path),contract={k:proof[k]for k in('phase','quality_admitted','performance_admitted')}),libraries=libraries,quality_admitted=False,performance_admitted=False)
  dest=output/('format-'+c+'.json');put(dest,fp);profiles[c]=row(dest)
 descriptor=dict(schema='deepseek1017-serving-admission/536',profile_id=ident,parent_source_ready=row(b.HERE/'SOURCE_READY.json'),bank_format_binding=binding,bank_cohort=cohort(base),factory_root=str(root),factory_manifest=row(root/'DELIVERY_BINDINGS.json'),factory_source_manifest=row(root/'SOURCE_MANIFEST.json'),contexts=d['contexts'],native_source_evidence=d['_native_source_evidence'],cpu_expected=expected,format_profiles=profiles,source_admission_proof=row(proof_path),quality_admitted=False,performance_admitted=False)
 dest=output/'PROFILE.json';put(dest,descriptor);pin=row(dest);validate_descriptor(pin,binding,base);return pin
