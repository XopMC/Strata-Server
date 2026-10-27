from index_selection_v512 import uuids as _index_uuids
"""Old selected reference binder, NEW1017 source graph/retained bytes only.
The target is the original C++ Engine; target Python forward is prohibited.
No Torch/CUDA/model IO on module import.
"""
import ast,importlib,importlib.util,inspect,json,os,sys,types
from pathlib import Path
from private_source_loader_v512 import private_source
from one_bank_store import HERE,Q,D,need,source
from stage_devices_pair_v512 import validate_visible_pair

class NativeReference:
 def __init__(self,store,*,context=4096,packed_budget=2<<30):
  self.model=self.ops=None
  try:self._initialize(store,context=context,packed_budget=packed_budget)
  except BaseException:self.close();raise
 def _initialize(self,store,*,context=4096,packed_budget=2<<30):
  need(store.validate_selected_owner()['runtime_admissible']is True,'production new onebank owner required')
  self.store=store;self._owns_store=False;self.model=None;self.ops=None;self.chunked=None;self.context=context
  import torch
  need(torch.__version__=='2.10.0+cu128'and torch.version.cuda=='12.8','existing DS runtime required')
  need(not torch.backends.cuda.matmul.allow_tf32,'same original noTF32 arithmetic')
  self.torch=torch;self.device=torch.device('cuda:0');self.runtime_config_path=store.source/'runtime-config.json'
  # Reuse the literal selected binder/packed operators and their complete old
  # numerical closure; load ONLY official model source from this new bundle.
  old=source(D/'Strata/backend/deepseek41/streamed_reference_selected_v250.py','backend.deepseek41._old1017_binder')
  from backend.deepseek41.packed_cuda import PackedCudaOperations
  from backend.deepseek41.streamed_reference import NativeOperations
  from backend.deepseek41.residency import PackedTileCache
  from backend.deepseek41 import sm89_quant,sm89_attention
  from backend.deepseek41 import portable_kernels
  aux=private_source(Path(__file__).resolve().parent/'auxiliary_index_v512.py','_index_bundle_graph_loader_v512')
  self.graph=aux.load_bundle_graph(store,portable_kernels)
  self.graph._woa_1017_reference_boundary=True # source is authenticated NEW official model, not patched old forward
  owner=self
  def admit():
   store.validate_selected_owner();need(validate_visible_pair()[0]==_index_uuids()[0],'same physical GPU0 target under exact pair mask')
  def rebind_admission(module,name,next_anchor):
   body=inspect.getsource(getattr(module,name));start=body.index('    if not _ADMITTED:');end=body.index(next_anchor,start)
   body=body[:start]+'    _new_owner_admit()\n'+body[end:];scope=dict(module.__dict__);scope['_new_owner_admit']=admit;exec(compile(body,str(module.__file__),'exec'),scope);return scope[name]
  quantize=rebind_admission(sm89_quant,'quantize_activation','    shape,width =')
  attention=rebind_admission(sm89_attention,'sparse_attn','    batch,queries,heads,dim =')
  asset=next(a for a in store.config['runtime_asset_map']['items']if a['runtime_path']=='inference/kernel.py')
  from one_bank_store import digest
  need(digest(store.source/asset['output_name'])==asset['output_sha256']=='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455','same literal original operator source')
  spec=importlib.util.spec_from_file_location('_new1017_old_operator_kernel',store.source/asset['output_name']);kernels=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernels)
  class Ops(PackedCudaOperations):
   def __init__(self):
    NativeOperations.__init__(self,store,owner.device,4096,None);self.row_decode_mode='cuda';self.activation_quantizer='sm89-fragment';self.weight_cache=PackedTileCache(packed_budget,self.device);self.kernels=kernels;self.packed_linear_calls=0
   def quantize_activation(self,x):return quantize(x)
  self.ops=Ops();self.graph.sparse_attn=attention
  config={k:v for k,v in store.config.items()if k not in('native_geometry_contract','runtime_asset_map','compact_mixed_source_provenance')};config.update(max_batch_size=1,max_seq_len=context,temperature=0)
  self.args=args=self.graph.ModelArgs(**config)
  from transformers import AutoTokenizer
  self.tokenizer=AutoTokenizer.from_pretrained(store.source,local_files_only=True,trust_remote_code=False)
  previous=torch.get_default_dtype()
  try:
   torch.set_default_dtype(torch.bfloat16)
   with torch.device('meta'):self.model=self.graph.Transformer(args,self.tokenizer)
  finally:torch.set_default_dtype(previous)
  # Same old meta-backed binding/allocation schedule. Main expert matrices are
  # never requested; only their unused module parameters remain on meta.
  old.NativeReference._bind(self,args)
  self.runtime_corrections=dict(index_cache_owner_publication_layers='official_1017_forward',woa_boundary='official1017_model_source',target_python_forward=False,linear_mode='old-selected-native-cuda',activation_quantizer='sm89-fragment')
  self.model.eval()
 def forward(self,*_,**__):raise RuntimeError('ordinary target inference is the selected complete C++ Engine only')
 def close(self):
  if self.model is not None:
   self.torch.cuda.synchronize(self.device);self.model=None
  if self.ops is not None:self.ops.weight_cache.clear();self.ops=None
