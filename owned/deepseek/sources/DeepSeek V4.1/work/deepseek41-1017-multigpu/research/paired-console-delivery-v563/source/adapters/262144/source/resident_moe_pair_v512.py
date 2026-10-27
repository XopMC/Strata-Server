"""Resident original DSpark experts with GPU-only routing dispatch; target unchanged."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import ctypes as C,time,types
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
ROOT=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41')) # fixed producer root; this private companion lives elsewhere
class ResidentDraftMoe:
 def __init__(self,store,device='cuda:0',budget=8*1024**3,on_progress=None):
  from native_cuda_loader_pair_v512 import load_isolated
  self.store=store;self.device=torch.device(device);self.tensors=[];self.tables=[];self.scratch=None;self.stats=dict(calls=0,loaded_bytes=0,load_seconds=0,target_experts_modified=False,draft_topk=3,draft_stages=3,experts_per_stage=128)
  self.lib=load_isolated(ROOT/'build/native-mtp-resident-moe-v2.so');p=C.c_void_p;i=C.c_int;s=C.c_size_t
  self.lib.ds41_mtp_moe_bytes.argtypes=[i];self.lib.ds41_mtp_moe_bytes.restype=s;self.lib.ds41_mtp_moe_run.argtypes=[p,p,p,p,i,p,p,s];self.lib.ds41_mtp_moe_run.restype=i
  names=[];total=0
  for stage in range(3):
   layer=[]
   for expert in range(128):
    for proj in('w1','w3','w2'):
     prefix=f'mtp.{stage}.ffn.experts.{expert}.{proj}';rows,cols=(5120,2304)if proj=='w2'else(2304,5120)
     wi=store.info(prefix+'.weight');si=store.info(prefix+'.scale')
     if wi['dtype']!='I8'or wi['shape']!=[rows,cols//2]or si['dtype']!='F8_E8M0'or si['shape']!=[rows,cols//32]:raise ValueError('Expected original pinned DSpark MXFP4/E8M0 weights')
     for name in(prefix+'.weight',prefix+'.scale'):layer.append(name);total+=store.info(name)['bytes']
   names.append(layer)
  if total>budget or torch.cuda.mem_get_info()[0]<total+768*1024**2:raise MemoryError('Full resident DSpark experts do not fit remaining GPU0 VRAM')
  start=time.perf_counter()
  for stage,layer in enumerate(names):
   if on_progress:on_progress(dict(phase='draft_experts_loading',stage=stage,loaded_bytes=self.stats['loaded_bytes'],total_bytes=total))
   pointers=[]
   for name in layer:
    raw=np.frombuffer(store.raw(name),np.uint8).copy();tensor=torch.from_numpy(raw).to(self.device);self.tensors.append(tensor);pointers.append(tensor.data_ptr());self.stats['loaded_bytes']+=tensor.numel()
    if on_progress and len(pointers)%96==0:on_progress(dict(phase='draft_experts_loading',stage=stage,loaded_bytes=self.stats['loaded_bytes'],total_bytes=total))
   self.tables.append(torch.tensor(pointers,dtype=torch.int64,device=self.device))
  self.scratch=torch.empty(self.lib.ds41_mtp_moe_bytes(5),dtype=torch.uint8,device=self.device);torch.cuda.synchronize();self.stats['load_seconds']=time.perf_counter()-start
 def run(self,stage,x,ids,weights):
  if not torch.is_inference_mode_enabled()or stage not in(0,1,2)or x.dtype!=torch.bfloat16 or x.device!=self.device or x.ndim!=2 or x.shape[1]!=5120 or not 1<=len(x)<=5 or ids.shape!=(len(x),3)or weights.shape!=ids.shape or ids.dtype!=torch.int64:raise ValueError('Bounded GPU0 DSpark geometry required')
  if torch.cuda.current_stream().cuda_stream!=torch.cuda.default_stream().cuda_stream:raise ValueError('Resident DSpark uses the default CUDA stream')
  x=x.contiguous();ids=ids.contiguous();weights=weights.float().contiguous();out=torch.empty_like(x,dtype=torch.float32)
  rc=self.lib.ds41_mtp_moe_run(x.data_ptr(),ids.data_ptr(),weights.data_ptr(),self.tables[stage].data_ptr(),len(x),out.data_ptr(),self.scratch.data_ptr(),self.scratch.numel())
  if rc:raise RuntimeError('Resident DSpark CUDA launch failed: '+str(rc))
  self.stats['calls']+=1;return out
 def close(self):
  torch.cuda.synchronize(self.device);self.tables.clear();self.tensors.clear();self.scratch=None
@contextmanager
def resident_draft_moe(reference):
 if len(reference.model.mtp)!=3 or any(b.ffn.n_routed_experts!=128 or b.ffn.n_activated_experts!=3 for b in reference.model.mtp):raise ValueError('Expected three128-expert/top3 DSpark stages')
 cache=ResidentDraftMoe(reference.store,reference.device);saved=[]
 try:
  for stage,block in enumerate(reference.model.mtp):
   module=block.ffn;saved.append((module,module.forward))
   def forward(this,x,image_mask=None,stage=stage):
    if image_mask is not None:raise ValueError('DSpark draft stages are text-only')
    shape=x.shape;flat=x.reshape(-1,5120);weights,ids=this.gate(flat);routed=cache.run(stage,flat,ids,weights);return(routed+this.shared_experts(flat)).to(x.dtype).reshape(shape)
   module.forward=types.MethodType(forward,module)
  yield cache.stats
 finally:
  for module,forward in saved:module.forward=forward
  cache.close()
