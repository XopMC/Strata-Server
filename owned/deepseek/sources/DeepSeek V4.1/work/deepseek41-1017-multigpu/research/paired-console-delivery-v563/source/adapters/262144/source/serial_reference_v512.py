"""Deny-until-bound actual352serial reference; pure pinned JSON, no native import."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
def need(v,m):
 if not v:raise RuntimeError('353 serial reference: '+m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pinned(row):
 need(isinstance(row,dict)and set(row)>={'path','sha256'},'actualRoot reference pin missing')
 p=Path(row['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and digest(p)==row['sha256'],'changed actualRoot reference');return p
def get_reference():
 v=json.loads((P/'SERIAL_REFERENCE_BINDING.json').read_text());need(v.get('schema')==442 and v.get('ready')is True,'NO ACTUAL352SERIAL REFERENCE: binding/seal/publish/runtime prohibited')
 rawpath=pinned(v['raw']);summarypath=pinned(v['Root_summary'])
 need(summarypath.is_relative_to(M/'state')and summarypath.suffix=='.json','Root actual completed352serial state summary only')
 expected=M.parents[1]/'outputs/deepseek41-1017-multigpu/pair-controller-v214/root-serial352-new256-two-pass'
 need(rawpath.is_relative_to(expected)and rawpath.name=='raw-result.json','canonical actual352serial raw only')
 summary=json.loads(summarypath.read_text());need(summary.get('phase')=='ACTUAL352_NEW256_TWO_PASS_SERIAL_REFERENCE_COMPLETE','actualRoot352completed phase');need(summary.get('raw')==str(rawpath)and summary.get('raw_sha256')==v['raw']['sha256'],'Root completed serial result must bind actual raw SHA')
 raw=json.loads(rawpath.read_text())
 seam=json.loads((P/'SEAM_BINDING.json').read_text());dso=seam['contexts']['262144'];need(raw.get('runtime_configuration',{}).get('implementation_sha256',{}).get(dso['path'])==dso['sha256'],'actual352 serial must use SAME351 DSO')
 copy=raw.get('serial_stage_copy_v351',{});c=copy.get('after_prefill',{});a=c.get('serial351',[]);need(len(a)==12 and a[:3]==[351,1,1]and a[3]>0 and 0<a[4]<=a[3]and a[5]>0 and a[6:10]==[1,1,4,15]and a[10:]==[0,0]and c.get('legacy3')==a[3:6],'actual352 validated351 copycensus')
 need(c==copy.get('after_generation')==copy.get('after_close'),'actual352 copies immutable throughclose')
 need(raw.get('actual_prefill_shape')==dict(passes=2,chunk_tokens=131072,attention_batch=2048),'actual serial samegeometry')
 need(raw.get('usage',{}).get('prompt_tokens')==261568 and raw['usage'].get('completion_tokens')==512,'actual full serial prompt/output count')
 ids=raw.get('generated_token_ids');need(isinstance(ids,list)and len(ids)==512 and all(type(x)is int and x>=0 for x in ids),'actual512 serial IDs')
 need(len(set(ids))>=16,'actual varied serial output')
 geometry=raw.get('prefill_geometry_v335',{}).get('after_prefill',{});g=geometry.get('geometry335');w=geometry.get('wavefront325')
 need(isinstance(g,list)and len(g)==32 and g[:20]==[335,1,1,262144,261568,131072,131072,2048,4096,1,1,0,0,2,80,2,0,0,0,0],'actual335 SERIAL census not a requested shape')
 need(w==[325]+[0]*47,'serial reference used no wavefront')
 need(geometry==raw['prefill_geometry_v335'].get('after_generation')==raw['prefill_geometry_v335'].get('after_close'),'serial geometry/state census immutable throughclose')
 need(raw.get('mtp',{}).get('mode')=='native4'and raw.get('cancelled')is False,'actual completed native4 serial')
 for k in('prefill_logits_sha256','encoded_prompt_sha256','encoded_prompt_token_ids_sha256'):
  x=raw.get(k);need(isinstance(x,str)and len(x)==64 and all(c in'0123456789abcdef'for c in x),'actual serial '+k)
 return v,raw
def validate_prefill_reference(actual):
 v,ref=get_reference()
 for k in('actual_prefill_shape','prefill_logits_sha256','encoded_prompt_sha256','encoded_prompt_token_ids_sha256','checkpoint'):
  need(actual.get(k)==ref.get(k),'wavefront prefill differs from completed actualserial '+k)
 need(actual.get('prompt_tokens')==261568,'wavefront prompt count')
 # Historical352 is a numerical reference only; current live139 bank is authenticated separately.
 need(actual.get('bank_format_binding_v146',{}).get('layout_tag')==139,'current bank must remain paired139')
 return dict(actual352_raw=v['raw'],Root_summary=v['Root_summary'],samegeometry_prefill_logits_exact=True,historical_bank_reference=True,speed_comparison_paired=False,output512_parity_pending=True)
def validate_result_reference(actual):
 v,ref=get_reference()
 need(actual.get('generated_token_ids')==ref['generated_token_ids'],'all512 wavefront output IDs must equal actual352serial')
 need(actual.get('checkpoint')==ref['checkpoint']and actual.get('actual_prefill_shape')==ref['actual_prefill_shape'],'actual checkpoint/geometry match')
 need(actual.get('prefill_logits_sha256')==ref['prefill_logits_sha256'],'actual logits exact')
 return dict(actual352_raw=v['raw'],Root_summary=v['Root_summary'],all512_IDs_exact=True,prefill_logits_exact=True,quality_admitted=False)
