"""Stream-aware original MXFP4 or our2/3-bit MTP weights, only on GPU0."""
import ctypes as C
import types
from contextlib import contextmanager
import torch
from resident_moe_pair_v512 import ResidentDraftMoe
from resident_stream_pair_v512 import QuantDraftMoe,ROOT

class SourceDraftMoe(ResidentDraftMoe):
 def __init__(self,store,device='cuda:0'):
  super().__init__(store,device)
  from native_cuda_loader_pair_v512 import load_isolated
  self.lib=load_isolated(ROOT/'build/native-mtp-resident-moe-stream-v41.so');self.bits=0
  p=C.c_void_p;i=C.c_int;s=C.c_size_t
  self.lib.ds41_mtp_moe_run.argtypes=[p,p,p,p,i,i,p,p,s,p];self.lib.ds41_mtp_moe_run.restype=i
  self.stats.update(weight_bits=4,scale_dtype='F8_E8M0',group_size=32,draft_weights='protected source MXFP4 in existing SHA-verified candidate RAM')
 run=QuantDraftMoe.run

@contextmanager
def resident_draft_moe(reference,bits=0):
 if len(reference.model.mtp)!=3 or any(b.ffn.n_routed_experts!=128 or b.ffn.n_activated_experts!=3 for b in reference.model.mtp):raise ValueError('Expected own DSpark geometry')
 cache=SourceDraftMoe(reference.store,reference.device)if bits==0 else QuantDraftMoe(reference.store,bits,reference.device);saved=[]
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
