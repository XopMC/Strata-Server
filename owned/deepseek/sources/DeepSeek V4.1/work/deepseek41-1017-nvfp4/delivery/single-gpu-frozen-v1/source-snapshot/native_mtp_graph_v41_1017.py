"""Capture the GPU DSpark proposal with dynamic true RoPE/window positions."""
import math,time,types,os
import numpy as np
import torch
from native_mtp_capture_stream_v92 import capture_stream
from native_cpp_speculative_recent import ReferenceDSpark as EagerDSpark
from backend.deepseek41.sm89_attention import attention_kernel

class ReferenceDSpark(EagerDSpark):
 def __init__(self,reference):
  if os.environ.get('DS41_ATTENTION_WOA_1017_SOURCE_PORT')!='1':raise ValueError('Private1017 boundary defaultOFF; Root operator proof required')
  if getattr(reference.graph,'_woa_1017_reference_boundary',False) is not True:raise ValueError('Private1017 eager/reference graph binding required before MTP capture')
  if reference.graph.fp8_block_size!=32 or reference.graph.scale_fmt!='ue8m0' or reference.graph.scale_dtype!=torch.float8_e8m0fnu:raise ValueError('Exact1017 group32/E8M0 graph required')
  super().__init__(reference);self.saved=[];self.cuda_graph=None
  self.position_tensor=torch.ones(1,dtype=torch.long,device=reference.device)
  self.query_offsets=torch.arange(5,device=reference.device);self.key_offsets=torch.arange(133,device=reference.device)
  self.pending_tensor=torch.zeros(1,dtype=torch.long,device=reference.device)
  self.main_tensor=torch.empty((1,1,15360),dtype=torch.bfloat16,device=reference.device)
  self.capture_stream=capture_stream(reference)
  self.metadata=dict(mode='CUDA graph DSpark; dynamic position and KV',captures=0,replays=0,setup_seconds=0.,draft_parity_checks=[],capture_stream_policy='reference-single-stream-v92',capture_stream=int(self.capture_stream.cuda_stream))
  for block in reference.model.mtp:
   attn=block.attn
   if attn.window_size!=128 or attn.compress_ratio!=0 or attn.head_dim!=512 or attn.n_local_heads!=64 or attn.n_local_groups!=8 or attn.o_lora_rank!=1024:raise ValueError('Expected bounded DSpark attention geometry')
   self.saved.append((attn,attn.forward));attn.forward=types.MethodType(self._forward_attention(),attn)

 def _forward_attention(self):
  graph=self.reference.graph;position=self.position_tensor;queries=self.query_offsets;keys=self.key_offsets
  def forward(attn,x,start_pos,main_x):
   # The caller's Python start_pos is irrelevant to a replay. Both RoPE and
   # ring writes use the mutable GPU position, including crossing128 slots.
   main_freqs=attn.freqs_cis.index_select(0,position)
   main_kv=attn.kv_norm(attn.wkv(main_x));graph.apply_rotary_emb(main_kv[...,-attn.rope_head_dim:],main_freqs);graph.act_quant(main_kv,graph.fp8_block_size,graph.scale_fmt,graph.scale_dtype,True)
   freqs=attn.freqs_cis.index_select(0,position+1+queries)
   qr=attn.q_norm(attn.wq_a(x));q=attn.wq_b(qr).unflatten(-1,(64,512));graph.apply_rotary_emb(q[...,-attn.rope_head_dim:],freqs)
   kv=attn.kv_norm(attn.wkv(x));graph.apply_rotary_emb(kv[...,-attn.rope_head_dim:],freqs);graph.act_quant(kv,graph.fp8_block_size,graph.scale_fmt,graph.scale_dtype,True)
   attn.window_kv_cache[:1].index_copy_(1,position%128,main_kv)
   kv=torch.cat([attn.window_kv_cache[:1],kv],dim=1)
   valid=(position+1).clamp(max=128)
   indices=torch.where(keys<valid,keys,torch.where(keys<valid+5,128+keys-valid,-1)).int().view(1,1,133).expand(1,5,133).contiguous()
   # Shape/range is guaranteed by construction. Avoid eager GPU->CPU range
   # assertions during capture; execute the same64-key TileLang arithmetic.
   output=torch.empty_like(q)
   for first in range(0,64,16):
    query=torch.empty((1,5,16,512),device=q.device,dtype=q.dtype);query.copy_(q[:,:,first:first+16]);result=torch.empty_like(query)
    attention_kernel(16,512,float(attn.softmax_scale))(query,kv,result,attn.attn_sink[first:first+16].contiguous(),indices)
    output[:,:,first:first+16]=result
   graph.apply_rotary_emb(output[...,-attn.rope_head_dim:],freqs,True)
   output=output.view(1,5,attn.n_local_groups,-1)
   graph.act_quant(output,graph.fp8_block_size,graph.scale_fmt,graph.scale_dtype,True)
   weight=attn.wo_a.weight.view(attn.n_local_groups,attn.o_lora_rank,-1)
   return attn.wo_b(torch.einsum('bsgd,grd->bsgr',output,weight).flatten(2))
  return forward

 def _evaluate(self):
  return self.reference.model.forward_spec(self.pending_tensor,self.main_tensor,1)

 def _capture(self,pending,position):
  started=time.perf_counter();dynamic=[(attn,attn.forward)for attn,_ in self.saved]
  try:
   for attn,forward in self.saved:attn.forward=forward
   with torch.device(self.reference.device):expected=self.reference.model.forward_spec(self.pending_tensor,self.main_tensor,position)
   expected_ids=expected[0][0].tolist()
  finally:
   for attn,forward in dynamic:attn.forward=forward
  stream=self.capture_stream;stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream),torch.device(self.reference.device):
   for _ in range(3):self._evaluate()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  self.cuda_graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.cuda_graph,stream=stream),torch.device(self.reference.device):
   self.output=self._evaluate()
   self.valid_output=torch.isfinite(self.output[1]).all()&torch.isfinite(self.output[2]).all()
  self.cuda_graph.replay();torch.cuda.synchronize()
  actual_ids=self.output[0][0].tolist()
  self.metadata['draft_parity_checks'].append(dict(position=position,all_six_ids_exact=actual_ids==expected_ids))
  if actual_ids!=expected_ids:raise ValueError('Initial graphed draft differs from eager own-quant draft')
  self.metadata['captures']+=1;self.metadata['setup_seconds']+=time.perf_counter()-started

 def propose(self,pending,main,position):
  if type(position)is not int or not 0<position<self.reference.model.max_seq_len-5:raise ValueError('Actual bounded DSpark position required')
  self.metadata.setdefault('first_position',position);self.metadata['last_position']=position;self.metadata['crossed_window_128']=position//128!=self.metadata['first_position']//128
  self.pending_tensor.fill_(pending);self.position_tensor.fill_(position);self.main_tensor.copy_(self.tensor(main))
  if self.cuda_graph is None:self._capture(pending,position)
  else:self.cuda_graph.replay()
  self.metadata['replays']+=1
  if not bool(self.valid_output):raise ValueError('Nonfinite graphed DSpark output')
  ids=self.output[0][0].tolist()
  if ids[0]!=pending or len(ids)!=6 or any(not 0<=v<129280 for v in ids):raise ValueError('Graphed DSpark token geometry changed')
  self.raw_confidence=[float(v)for v in self.output[2][0].tolist()]
  self.confidence=[1/(1+math.exp(-max(-80.,min(80.,v))))for v in self.raw_confidence]
  return [int(v)for v in ids[1:]]

 def close(self):
  torch.cuda.synchronize()
  if self.cuda_graph is not None:self.cuda_graph.reset()
  self.cuda_graph=None
  self.output=None;self.valid_output=None
  for attn,forward in self.saved:attn.forward=forward
  self.saved.clear()
