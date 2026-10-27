"""Small owner/transport adapter; no model/CUDA import on module import."""
import hashlib,importlib.util,json,math,sys,threading,time
from pathlib import Path
MODEL_ID='deepseek-v4.1-flash-1017-gentle-heretic-mixed'

def make_owner_class(Base):
 from strict_owner import make_owner_class as strict
 return strict(Base)

def compact_result_proof(result):
 """Small result-only parity ledger; never prompt/messages/full output/token IDs."""
 ids=result.get('generated_token_ids');valid=isinstance(ids,list)and all(type(x)is int for x in ids)
 encoded=json.dumps(ids,separators=(',',':')).encode()if valid else None
 mtp=result.get('mtp',{});stats=mtp.get('statistics',{})if isinstance(mtp,dict)else{}
 keys=('max_draft_tokens','confidence_min','confidence_calibrated','proposals','proposed_tokens','draft_tokens','accepted_draft_tokens','target_forward_calls','target_input_tokens','committed_inputs','rejected_tentative_inputs','declined_proposals','draft_seconds','verify_seconds','main_commit_seconds')
 summary={k:stats[k]for k in keys if type(stats.get(k))in(int,float,bool)}
 if isinstance(stats.get('native_verifier_ms'),list)and len(stats['native_verifier_ms'])==4 and all(type(x)in(int,float)and math.isfinite(x)for x in stats['native_verifier_ms']):summary['native_verifier_ms']=list(stats['native_verifier_ms'])
 # Windows/target_predictions/verified_drafts can contain full token IDs: omitted.
 native=result.get('serving528',{});native_keys=('schema','prepare_epoch','armed','mode','context','N','position','split','cap','B','E','first_rows','second_rows','carry_rows','live_Impls','active_prefill')
 serving={phase:{k:row[k]for k in native_keys if type(row.get(k))is int}for phase in('prepared','prefill','after_engine_close')if isinstance(native,dict)and isinstance((row:=native.get(phase)),dict)}
 modes={k:mtp[k]for k in('mode','requested_mode')if isinstance(mtp,dict)and mtp.get(k)in('off','native1','native2','native3','native4','native5','native6')}
 return dict(output_token_ids_sha256=hashlib.sha256(encoded).hexdigest()if encoded is not None else None,
  prefill_logits_sha256=result.get('prefill_logits_sha256'),encoded_prompt_sha256=result.get('encoded_prompt_sha256'),encoded_prompt_token_ids_sha256=result.get('encoded_prompt_token_ids_sha256'),checkpoint=result.get('checkpoint'),actual_prefill_shape=result.get('actual_prefill_shape'),output_unique_ids=len(set(ids))if valid else None,usage=result.get('usage'),timings=result.get('timings'),mtp_summary=dict(**modes,statistics=summary),serving528=serving,cancelled=result.get('cancelled',False))

class LoggedRuntime:
 """All run/close calls execute on the original FIFO CUDA-issuing thread."""
 def __init__(self,runtime,stop,codec,log,*,sequence_base=0):
  self.request_sequence=sequence_base;self.request_sequence_limit=sequence_base+(1<<32)-1
  self.runtime=runtime;self.stop=stop;self.codec=codec;self.log=log;self.activity=lambda:None
  self.config=runtime.configuration();self.config={**self.config,'serving_model_id':MODEL_ID,
   'serving_policy':'N1 FIFO; no N2-4 parallel claim','API_tools_vision_wrapper_verified':False}
 def configuration(self):return self.config
 def run(self,body,cancel,emit):
  # Covers the dequeue/active_job publication shutdown gap of the legacy FIFO.
  # A running native stage is never interrupted by an asynchronous exception.
  if self.stop.is_set():cancel.set();return
  if self.request_sequence>=self.request_sequence_limit:raise RuntimeError('Per-child request sequence exhausted')
  self.request_sequence+=1;sequence=self.request_sequence
  self.activity();started=time.monotonic();self.log('REQUEST_START',request_sequence_id=sequence,stream=body.get('stream',False),max_tokens=body.get('max_tokens'))
  def observed(event):
   if event.get('event')=='result':
    v=event['result'];self.log('REQUEST_RESULT',request_sequence_id=sequence,**compact_result_proof(v))
   return emit(event)
  try:return self.runtime.run(body,cancel,observed)
  finally:
   self.activity();self.log('REQUEST_END',request_sequence_id=sequence,elapsed_seconds=time.monotonic()-started,cancelled=cancel.is_set())
 def close(self):
  self.log('RUNTIME_CLOSE_BEGIN')
  try:result=self.runtime.close()
  except BaseException as error:
   self.log('RUNTIME_CLOSE_FAILED',error=repr(error));raise
  self.log('RUNTIME_CLOSE_END',success=True);return result

def load_transport(D,codec):
 # Only the authenticated BUNDLE's DSML/API codec is installed into this fresh
 # child. This does not reuse old checkpoint/admission/model identities.
 sys.modules['backend.deepseek41.api_codec']=codec
 name='backend.deepseek41._private1017_HTTP_transport_v252'
 path=Path(__file__).resolve().parent/'api_transport_v252.py'
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module)
 if module.MODEL_ID!=MODEL_ID:raise ValueError('Bundled API codec model identity differs')
 return module
