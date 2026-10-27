"""Typed501 admission +503 K5 leaf/504 relink proof. No native calls/counters."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import hashlib,json,os
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
_CACHE=None
def need(v,m):
 if not v:raise RuntimeError('K5 composite506: '+m)
def identity(p):
 p=Path(p);s=p.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink(),'canonical direct pin '+str(p));need(hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']and('bytes'not in r or p.stat().st_size==r['bytes']),'changed pin '+str(p));return p
def ar_objects(data):
 """GNU/SystemV archive parser; skips only symbol/name tables, returns raw objects."""
 need(data.startswith(b'!<arch>\n')and len(data)<=16<<20,'bounded ordinary ar archive')
 off=8;names=b'';objects=[]
 while off<len(data):
  h=data[off:off+60];need(len(h)==60 and h[58:60]==b'`\n','ar member header')
  try:size=int(h[48:58]);name=h[:16].decode('ascii').strip()
  except(ValueError,UnicodeDecodeError):raise RuntimeError('K5 composite506: invalid ar fields')
  off+=60;body=data[off:off+size];need(size>=0 and len(body)==size,'ar member extent');off+=size+(size&1)
  if name=='//':names=body;continue
  if name in('/','/SYM64/'):continue
  if name.startswith('/')and name[1:].isdigit():
   start=int(name[1:]);need(start<len(names),'ar long-name offset');name=names[start:].split(b'/\n',1)[0].decode('ascii')
  else:name=name.rstrip('/')
  need(name and '/'not in name,'flat ar object name');objects.append((name,body))
 need(off==len(data),'ar trailing bytes');return objects

def verify_link_argv(recipe,link,old_archive,new_archive,base_contexts,contexts):
 need(len(link['steps'])==2 and set(x['context']for x in link['steps'])=={'262144','524288'},'both faithful link contexts')
 for step in link['steps']:
  c=step['context'];need(step['exit_code']==0,'actual504 link success');want=next((x['argv']for x in recipe['commands']if '-shared'in x['argv']and base_contexts[c]['path']in x['argv']),None)
  need(want is not None and want.count(old_archive)==2,'original177 archive occurs twice in original501 link')
  expected=[contexts[c]['path']if x==base_contexts[c]['path']else new_archive if x==old_archive else x for x in want]
  need(step['argv']==expected,'504 link must differ only in output and two K5 archive arguments')

def component_check(v):
 need(v['phase']=='K5_503_RAW_TOP512_CHANGED_GRAPH_EXACT_COMPONENT_PASS'and v['no_model_TPS_claim']is True,'actual503 component scope')
 need(len(v['cases'])==6 and {x['case']for x in v['cases']}=={'small','tail','holes','masked','filled256','filled512'},'six selected503 cases')
 need(all(x['raw_score_bits_equal']is True and x['top512_IDs_equal']is True and x['first_difference']is None for x in v['cases']),'actual raw F32 score/top512 exactness')
 need(next(x for x in v['cases']if x['case']=='holes')['changed_graph_replays']==2,'actual changed graph replays')
 need(v['resources']==dict(registers=42,static_shared_bytes=24704,local_bytes=0,active_CTA_per_SM=3,binary_version=89),'actual503 component resource receipt')


def check(admission501):
 global _CACHE
 data=(P/'K5_COMPOSITE_BINDING.json').read_bytes();fingerprint=hashlib.sha256(data).hexdigest();v=json.loads(data)
 need(v.get('ready')is True and v['schema']=='deepseek1017-index501-k5-composite/506'and v['phase']=='ROOT506_INDEX501_PLUS_K5_503_COMPOSITE_BOUND_MODEL_PENDING','Root composite binding pending')
 need(v['admission_foundation501']['build']==admission501['build']and v['admission_foundation501']['contexts']==admission501['contexts'],'separate exact501 admission foundation')
 need(v['bank_format_binding']==admission501['bank_format_binding']and v['runtime_context']==524288,'same actual139 owner and strict filled512 factory')
 if _CACHE is not None and _CACHE[0]==os.getpid():
  need(_CACHE[1]==fingerprint,'immutable composite changed')
  for p,i in _CACHE[2]:need(identity(p)==i,'immutable composite input identity changed '+str(p))
  return v
 rows=[v[k]for k in('parent505_seal','link504','leaf_build503','leaf_component503','leaf_component_producer503','leaf_library503','leaf_source503','leaf_source_plan503','leaf_object503','leaf_archive504','original177_archive')]+list(v['contexts'].values())
 for r in rows:pin(r)
 b=json.loads(Path(v['link504']['path']).read_text());compile503=json.loads(Path(v['leaf_build503']['path']).read_text());component=json.loads(Path(v['leaf_component503']['path']).read_text())
 need(b['phase']=='ROOT504_INDEX501_PLUS_K5_503_LINK_PASS_MODEL_PENDING'and b['GPU_or_model_run']is False and b['other501_host_and_numerical_objects_reused']is True and b['weights_unchanged']is True,'actual504 only K5 relink scope')
 need(b['base_admission_build']=='state/index-native-admission-root-build-v501.json'and b['leaf_archive']=={k:v['leaf_archive504'][k]for k in('path','sha256')},'actual504 leaf+foundation identity')
 for c,row in v['contexts'].items():need(b['libraries'][c]==row,'actual504 library bytes/hash')
 need(compile503['library']['sha256']==v['leaf_library503']['sha256'],'actual503 component library')
 need(compile503['phase']=='K5_503_COMPILED_ONLY_NATIVE_PENDING'and compile503['exit_code']==compile503['link']['exit_code']==0 and compile503['source_sha256']==v['leaf_source503']['sha256'],'actual selected503 compilation')
 need('-DDS1017_K5_SWIZZLED_REGISTER_V503=1'in compile503['argv']and v['leaf_source503']['path']in compile503['argv']and compile503['argv'][compile503['argv'].index(v['leaf_source503']['path'])-1]=='-c'and compile503['argv'][compile503['argv'].index('-o')+1]==v['leaf_object503']['path'],'actual503 source+enabled leaf object')
 objects=ar_objects(Path(v['leaf_archive504']['path']).read_bytes());need(len(objects)==1 and objects[0][0]=='candidate.o'and objects[0][1]==Path(v['leaf_object503']['path']).read_bytes(),'one-member504 archive is exact selected503 compiled object')
 component_check(component)
 old_objects=ar_objects(Path(v['original177_archive']['path']).read_bytes());need(len(old_objects)==1 and old_objects[0][0]=='k5_indexer.cu.o'and hashlib.sha256(old_objects[0][1]).hexdigest()==component['actual_oracle_object'],'actual503 oracle is sole original177 object')
 recipe=json.loads(Path(admission501['recipe']['path']).read_text());verify_link_argv(recipe,b,v['original177_archive_path'],v['leaf_archive504']['path'],admission501['contexts'],v['contexts'])
 _CACHE=(os.getpid(),fingerprint,[(r['path'],identity(r['path']))for r in rows])
 return v

def record(admission501):
 v=check(admission501)
 return dict(schema=v['schema'],active_contexts=v['contexts'],leaf_mode='503shared_rowXOR8_B_reuse',source=v['leaf_source503'],object=v['leaf_object503'],archive=v['leaf_archive504'],actual504=v['link504'],actual_component=v['leaf_component503'],component_resources=v['component_resources'],component_latency=v['component_latency'],component_timing_is_not_model_TPS=True,resource_getter_called=False,resource_getters_do_not_prove_execution=True,hot_path_counters_added=False,model_parity_pending=True,quality_admitted=False,performance_admitted=False)
