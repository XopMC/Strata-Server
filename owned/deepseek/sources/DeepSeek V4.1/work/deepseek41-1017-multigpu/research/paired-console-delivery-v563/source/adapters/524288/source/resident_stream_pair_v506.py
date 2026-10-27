"""GPU0 DSpark own2/own3 experts; target model remains unchanged."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import ctypes as C
import hashlib,json,mmap,time,types
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from resident_moe_pair_v506 import ResidentDraftMoe,resident_draft_moe as original_moe
ROOT=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41')) # fixed producer root; this private companion lives elsewhere

class QuantDraftMoe(ResidentDraftMoe):
 def __init__(self,store,bits,device='cuda:0',budget=8*1024**3):
  from native_cuda_loader_pair_v506 import load_isolated
  if bits not in(2,3):raise ValueError('Explicit two/three-bit DSpark format required')
  bundled=Path(store.source)/'drafts'
  directory=bundled if (bundled/f'own{bits}.manifest.json').is_file() else ROOT/'temporary-weights/mtp-v39'
  manifest_path=directory/f'own{bits}.manifest.json';manifest=json.loads(manifest_path.read_text())
  payload_path=directory/manifest['file']
  if manifest_path.is_symlink() or not payload_path.resolve().is_relative_to(directory.resolve()):raise ValueError('MTP payload must stay inside its declared head directory')
  if manifest['file']!=f'own{bits}.bin'or manifest['format']!='deepseek41-signed-uniform-bf16-group-v1'or manifest['bits']!=bits or manifest['group_size']!=32 or manifest['model_bundle_sha256']!=store.ram_preload_report['model_bundle_sha256']or manifest['source_revision']!=store.manifest['source_revision']:raise ValueError('MTP quant identity differs from our checkpoint')
  with open(payload_path,'rb')as file:
   if payload_path.stat().st_size!=manifest['payload_bytes']or hashlib.file_digest(file,'sha256').hexdigest()!=manifest['sha256']:raise ValueError('MTP quant payload is incomplete')
  self.store=store;self.device=torch.device(device);self.tensors=[];self.tables=[];self.scratch=None;self.bits=bits
  self.stats=dict(calls=0,loaded_bytes=0,load_seconds=0,target_experts_modified=False,draft_topk=3,draft_stages=3,experts_per_stage=128,weight_bits=bits,scale_dtype='BF16',group_size=32,quant_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),quant_payload_sha256=manifest['sha256'],quant_payload_path=str(payload_path.resolve()),standalone_head=directory==bundled)
  self.lib=load_isolated(ROOT/'build/native-mtp-resident-moe-stream-v41.so');p=C.c_void_p;i=C.c_int;s=C.c_size_t
  self.lib.ds41_mtp_moe_bytes.argtypes=[i];self.lib.ds41_mtp_moe_bytes.restype=s
  self.lib.ds41_mtp_moe_run.argtypes=[p,p,p,p,i,i,p,p,s,p];self.lib.ds41_mtp_moe_run.restype=i
  total=manifest['payload_bytes']
  if total>budget or torch.cuda.mem_get_info()[0]<total+768*1024**2:raise MemoryError('Quantized DSpark does not fit GPU0 reserve')
  start=time.perf_counter();expected=set()
  with open(payload_path,'rb')as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)as payload:
   for stage in range(3):
    pointers=[]
    for expert in range(128):
     for proj in('w1','w3','w2'):
      prefix=f'mtp.{stage}.ffn.experts.{expert}.{proj}';expected.add(prefix);record=manifest['tensors'][prefix];rows,cols=(5120,2304)if proj=='w2'else(2304,5120)
      if record['shape']!=[rows,cols]or record['weight_bytes']!=rows*cols*bits//8 or record['scale_bytes']!=rows*cols//32*2:raise ValueError('MTP quant geometry mismatch')
      for field in('weight','scale'):
       offset,count=record[field+'_offset'],record[field+'_bytes']
       if type(offset)is not int or offset<0 or offset+count>total:raise ValueError('MTP quant payload bounds changed')
       raw=np.frombuffer(payload,np.uint8,count=count,offset=offset).copy();tensor=torch.from_numpy(raw).to(self.device);self.tensors.append(tensor);pointers.append(tensor.data_ptr());self.stats['loaded_bytes']+=tensor.numel()
    self.tables.append(torch.tensor(pointers,dtype=torch.int64,device=self.device))
  if expected!=set(manifest['tensors'])or self.stats['loaded_bytes']!=total:raise ValueError('MTP quant inventory mismatch')
  self.scratch=torch.empty(self.lib.ds41_mtp_moe_bytes(5),dtype=torch.uint8,device=self.device);torch.cuda.synchronize();self.stats['load_seconds']=time.perf_counter()-start
 def run(self,stage,x,ids,weights):
  if not torch.is_inference_mode_enabled()or stage not in(0,1,2)or x.dtype!=torch.bfloat16 or x.device!=self.device or x.ndim!=2 or x.shape[1]!=5120 or not 1<=len(x)<=5 or ids.shape!=(len(x),3)or weights.shape!=ids.shape or ids.dtype!=torch.int64:raise ValueError('Bounded GPU0 DSpark geometry required')
  x=x.contiguous();ids=ids.contiguous();weights=weights.float().contiguous();out=torch.empty_like(x,dtype=torch.float32)
  rc=self.lib.ds41_mtp_moe_run(x.data_ptr(),ids.data_ptr(),weights.data_ptr(),self.tables[stage].data_ptr(),len(x),self.bits,out.data_ptr(),self.scratch.data_ptr(),self.scratch.numel(),torch.cuda.current_stream().cuda_stream)
  if rc:raise RuntimeError('Quantized DSpark CUDA launch failed: '+str(rc))
  self.stats['calls']+=1;return out

@contextmanager
def resident_draft_moe(reference,bits=0):
 if bits==0:
  with original_moe(reference)as stats:yield stats
  return
 if len(reference.model.mtp)!=3 or any(b.ffn.n_routed_experts!=128 or b.ffn.n_activated_experts!=3 for b in reference.model.mtp):raise ValueError('Expected own DSpark geometry')
 cache=QuantDraftMoe(reference.store,bits,reference.device);saved=[]
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
