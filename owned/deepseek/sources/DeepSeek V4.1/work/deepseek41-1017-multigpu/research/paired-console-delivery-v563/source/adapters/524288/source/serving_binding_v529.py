"""Explicit immutable528 library plus fresh owner capability; no native imports."""
from pathlib import Path
import hashlib,json,os
P=Path(__file__).resolve().parents[1]
BASE_FLAGS={'DS1017_PAIR_SERVING_V516':'1','DS1017_UPLOAD_RESERVED_COHORT_V539':'1','DS1017_PASS_UPLOAD_LEASE_V554':'1','DS1017_PASS_UPLOAD_ASYNC_V558':'1'}
_CAPABILITY=None
_STORE=None
def need(v,m):
 if not v:raise RuntimeError('serving529 binding: '+m)
def pin(r):
 p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'changed canonical pin '+str(p));return p
def check():
 v=json.loads((P/'SERVING_BINDING_v529.json').read_text());need(v.get('schema')==529 and v.get('ready')is True,'Root528 build/fresh-bank source binding pending; no API execution')
 need(type(v['features']['direct_fp8'])is bool and type(v.get('request_prepare_flags'))is int and v['request_prepare_flags']in(0,4),'typed independent feature flags')
 need(v['context']in(262144,524288)and v['flags']==dict(BASE_FLAGS,DS1017_PREFILL_DIRECT_FP8_V531=str(int(v['features']['direct_fp8']))),'explicit context/native serving flag')
 need(v['prepare_symbol']=='ds528_prepare_request_v1'and v['getter_symbol']=='ds528_prefill_plan_info','actual528 C ABI')
 for key in('native_library','native528_build','idle528','native528_source_pins','direct531_component','cohort539_component','cohort539_run','cohort539_source','numerical_parent_seal','source_manifest529'):pin(v[key])
 need(_CAPABILITY is not None and _STORE is not None,'fresh inherited owner capability must be configured by shipping factory')
 need(_CAPABILITY==dict(_STORE.bank_format_binding_v146),'live inherited capability changed')
 v=dict(v,bank_format_binding=dict(_CAPABILITY),historical_epochs_are_not_current_capabilities=True)
 need(v['bank_format_binding']['layout_tag']==139 and v['bank_format_binding']['format_epoch']>=2,'current paired139 capability')
 need(v['quality_admitted']is False and v['performance_admitted']is False,'shipping source is not model quality/speed proof')
 return v
def configure(store,version):
 global _CAPABILITY,_STORE
 from importlib import import_module
 bank=import_module(f'pair_format_v{version}').binding(store)
 need(getattr(store,'bank_runtime_admissible_v146',False)is True,'Manager must explicitly publish actual139 runtime admissibility')
 need(_CAPABILITY is None or(_CAPABILITY==bank and _STORE is store),'no source/owner switch inside a CUDA child')
 _CAPABILITY=dict(bank);_STORE=store
 return check()
def assert_store(store):
 need(store is _STORE and dict(store.bank_format_binding_v146)==_CAPABILITY,'same actual inherited store capability')
 return dict(_CAPABILITY)

def latch():
 v=check()
 for k,w in v['flags'].items():need(os.environ.get(k)in(None,w),'conflicting '+k);os.environ[k]=w

def overlay(value):
 v=check();need(set(value['contexts'])=={str(v['context'])},'faithful selected context scope')
 return dict(value,historical_numerical_contexts=value['contexts'],contexts={str(v['context']):v['native_library']},serving528_binding=v)
