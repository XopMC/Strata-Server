"""Resident DSpark dense/embedding path. Only proposals change; target unchanged."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import ctypes as C,types,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
ROOT=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41')) # fixed producer root; this private companion lives elsewhere
@contextmanager
def resident_draft_dense(reference,engine):
 from native_cuda_loader_pair_v506 import load_isolated
 from backend.deepseek41.context_capacity import cache_budget
 ops=reference.ops;store=reference.store;old_linear=ops.linear;old_rows=ops.rows;cache=ops.weight_cache;old_budget=cache.budget_bytes
 if ops.device!=torch.device('cuda:0')or ops.observer is not None:raise ValueError('GPU0 draft-only path')
 lib=load_isolated(ROOT/'build/native-mtp-dense-stream-v41.so');fn=lib.ds41_mtp_dense_fp8;fn.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_void_p]*3;fn.restype=C.c_int
 saved=[];packed={};dense={};stats=dict(bytes=0,fp8_calls=0,dense_calls=0,embedding_calls=0,target_modified=False);scratch=torch.empty(8*32768,dtype=torch.float32,device=ops.device)
 def raw(name,dtype):
  info=store.info(name);shape=info['shape']
  if info['bytes']%shape[0]:raise ValueError('Dense tensor row byte geometry differs')
  row_bytes=info['bytes']//shape[0];chunk=max(1,(32*1024**2)//row_bytes)
  # Allocate the final GPU payload once. torch.cat would temporarily own both
  # all uploaded chunks and a second full payload at the long-context limit.
  payload=torch.empty(info['bytes'],dtype=torch.uint8,device=ops.device)
  for first in range(0,shape[0],chunk):
   last=min(first+chunk,shape[0]);part=np.frombuffer(store.rows(name,np.arange(first,last)),np.uint8).copy()
   if part.size!=(last-first)*row_bytes:raise ValueError('Dense upload row payload differs')
   payload[first*row_bytes:last*row_bytes].copy_(torch.from_numpy(part))
  result=payload.view(dtype).reshape(shape);stats['bytes']+=result.numel()*result.element_size();stats['bounded_GPU_upload']=True;return result
 cache_budget(cache,0);cache_budget(cache,128*1024**2);start=time.perf_counter()
 try:
  selected=[name for name in store.manifest['tensors']if name.startswith('mtp.')and '.experts.'not in name and name.endswith('.weight')and len(store.info(name)['shape'])==2]
  for name in selected:
   info=store.info(name)
   if info['dtype']=='F8_E4M3':
    if '.wo_a.'in name:dense[name]=old_rows(name,np.arange(info['shape'][0]),torch.bfloat16)
    else:packed[name]=(raw(name,torch.uint8),raw(name[:-6]+'scale',torch.uint8))
   elif info['dtype']=='BF16':
    matrix=raw(name,torch.bfloat16);dense[name]=matrix.float()if (('head.'in name and '.embed.'not in name)or '.ffn.gate.'in name)else matrix
  shared=engine.lib.ds41_native_engine_shared_dense;shared.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p];shared.restype=C.c_int
  borrowed=[]
  class SharedDense:
   def __init__(self,kind):
    self.owner=engine;address=C.c_size_t();rows=C.c_size_t();cols=C.c_size_t()
    if shared(engine.handle,kind,C.byref(address),C.byref(rows),C.byref(cols))or(rows.value,cols.value)!=(129280,5120):raise ValueError('Invalid borrowed native embedding/head')
    self.__cuda_array_interface__={'shape':(rows.value,cols.value),'typestr':'<u2','data':(address.value,False),'version':3}
  for kind,name in enumerate(('embed.weight','head.weight')):
   owner=SharedDense(kind);borrowed.append(owner);dense[name]=torch.as_tensor(owner,device=ops.device).view(torch.bfloat16)
  stats['borrowed_target_bytes']=sum(dense[n].numel()*2 for n in('embed.weight','head.weight'))
  # Full resident rows remove GPU->CPU token-ID round trips in both embeddings.
  for name,module in (('embed.weight',reference.model.embed),('mtp.2.markov_head.embed.weight',reference.model.mtp[2].markov_head.embed)):
   saved.append((module,module.forward))
   def embed(this,ids,name=name):stats['embedding_calls']+=1;return torch.nn.functional.embedding(ids,dense[name])
   module.forward=types.MethodType(embed,module)
  def rows(name,indices,dtype=torch.bfloat16):
   if name in dense and '.wo_a.'in name:
    indices=np.asarray(indices).reshape(-1)
    if np.array_equal(indices,np.arange(len(dense[name]))):return dense[name].to(dtype)
   return old_rows(name,indices,dtype)
  def linear(name,x,weight_dtype,bias_name=None):
   m=x.numel()//x.shape[-1]
   if name in packed and 1<=m<=8 and bias_name is None:
    w,s=packed[name];flat=x.reshape(m,-1).contiguous();out=torch.empty((m,w.shape[0]),dtype=torch.bfloat16,device=ops.device)
    if flat.dtype!=torch.bfloat16:raise ValueError('Draft FP8 expects BF16')
    if fn(flat.data_ptr(),w.data_ptr(),s.data_ptr(),m,w.shape[0],flat.shape[1],out.data_ptr(),scratch.data_ptr(),torch.cuda.current_stream().cuda_stream):raise RuntimeError('Draft dense GPU kernel rejected input')
    stats['fp8_calls']+=1;return out.reshape(*x.shape[:-1],w.shape[0])
   if name=='head.weight':
    # Input and source weights are BF16; preserve FP32 logits from Tensor Cores.
    flat=x.reshape(m,-1).to(torch.bfloat16);stats['dense_calls']+=1
    return torch.mm(flat,dense[name].T,out_dtype=torch.float32).reshape(*x.shape[:-1],dense[name].shape[0])
   if name in dense and '.wo_a.'not in name:
    weight=dense[name];activation=x.to(weight.dtype);bias=ops.small(bias_name,weight.dtype)if bias_name else None
    stats['dense_calls']+=1;return torch.nn.functional.linear(activation,weight,bias).to(x.dtype)
   return old_linear(name,x,weight_dtype,bias_name)
  ops.rows=rows;ops.linear=linear;torch.cuda.synchronize();stats['load_seconds']=time.perf_counter()-start;stats['resident_tensor_bytes']=sum(v.numel()*v.element_size()for v in dense.values())+sum(v.numel()*v.element_size()for pair in packed.values()for v in pair)
  yield stats
 finally:
  ops.linear=old_linear;ops.rows=old_rows
  for module,forward in saved:module.forward=forward
  torch.cuda.synchronize();packed.clear();dense.clear();cache_budget(cache,old_budget)
