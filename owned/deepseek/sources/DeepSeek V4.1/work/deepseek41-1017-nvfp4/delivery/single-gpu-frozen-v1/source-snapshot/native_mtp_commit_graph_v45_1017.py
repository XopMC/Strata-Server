"""Capture updates of DSpark windows for1..6 actually accepted target rows."""
import numpy as np
import torch
from native_mtp_graph_v41_1017 import ReferenceDSpark as ProposalGraph

class ReferenceDSpark(ProposalGraph):
 def __init__(self,reference):
  super().__init__(reference);self.commit_graphs={};self.metadata.update(commit_graph_captures=0,commit_graph_replays=0,commit_graph_bit_parity=[])

 def _evaluate_commit(self,entry):
  model=self.reference.model;graph=self.reference.graph
  main_x=model.mtp[0].main_norm(model.mtp[0].main_proj(entry['input']))
  positions=entry['first']+entry['offsets'];valid=torch.isfinite(entry['input']).all()
  for layer in model.mtp:
   attn=layer.attn;values=attn.kv_norm(attn.wkv(main_x))
   graph.apply_rotary_emb(values[...,-attn.rope_head_dim:],attn.freqs_cis.index_select(0,positions))
   graph.act_quant(values,graph.fp8_block_size,graph.scale_fmt,graph.scale_dtype,True)
   valid=valid&torch.isfinite(values).all();attn.window_kv_cache[:1].index_copy_(1,positions%128,values)
  return valid

 def _capture_commit(self,main,first):
  rows=len(main);entry=dict(input=self.tensor(main),first=torch.tensor([first],dtype=torch.long,device=self.reference.device),offsets=torch.arange(rows,device=self.reference.device))
  stream=self.capture_stream;stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream),torch.device(self.reference.device):
   for _ in range(2):self._evaluate_commit(entry)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();entry['graph']=torch.cuda.CUDAGraph()
  with torch.cuda.graph(entry['graph'],stream=stream),torch.device(self.reference.device):entry['valid']=self._evaluate_commit(entry)
  self.commit_graphs[rows]=entry;self.metadata['commit_graph_captures']+=1

 def prepare_commits(self,main,last_position):
  for rows in range(1,7):
   example=np.repeat(main[-1:],rows,axis=0);first=last_position-rows+1
   self.commit(example,first)
   expected=[layer.attn.window_kv_cache.clone()for layer in self.reference.model.mtp]
   super().commit(example,first)
   exact=all(torch.equal(a,layer.attn.window_kv_cache)for a,layer in zip(expected,self.reference.model.mtp))
   self.metadata['commit_graph_bit_parity'].append(dict(rows=rows,first_position=first,all_window_bits_exact=exact))
   if not exact:raise ValueError('Graphed DSpark window update differs from eager')

 def commit(self,main,first,*,clear=False):
  if clear or len(main)>6:return super().commit(main,first,clear=clear)
  if not 1<=len(main)<=6 or type(first)is not int or first<0 or first+len(main)>self.reference.model.max_seq_len:raise ValueError('Accepted bounded target rows required')
  rows=len(main)
  if rows not in self.commit_graphs:self._capture_commit(main,first)
  entry=self.commit_graphs[rows];entry['input'].copy_(self.tensor(main));entry['first'].fill_(first);entry['graph'].replay();self.metadata['commit_graph_replays']+=1
  if not bool(entry['valid']):raise ValueError('Nonfinite graphed committed DSpark window')

 def close(self):
  torch.cuda.synchronize()
  for entry in self.commit_graphs.values():entry['graph'].reset()
  self.commit_graphs.clear();super().close()
