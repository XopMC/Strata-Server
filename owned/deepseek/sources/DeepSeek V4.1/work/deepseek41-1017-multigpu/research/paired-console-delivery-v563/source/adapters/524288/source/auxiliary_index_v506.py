"""Private auxiliary-only official1017 DSpark/vision over a registered V6 bundle.
No IO/Torch/DSO/CUDA on import. Root execution only, fresh process, same issuer.
No target Transformer, no old source/checkpoint/RAM admission impersonation.
"""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from contextlib import contextmanager
from pathlib import Path
import ast,hashlib,importlib,importlib.util,json,math,os,sys,types
HERE=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-nvfp4/research/compact-runtime-auxiliary-v8'));Q=HERE.parents[1]
PREPARED=Q.parent/'deepseek41-1017-heretic/research/native-fast-capture-candidate-v1/prepared-fast'
from index_selection_v506 import uuids as _index_uuids
GPU0=_index_uuids()[0]
MODES=('source-f32-row128','selected-bf16-tc-f32')
def need(ok,reason):
    if not ok:raise ValueError(reason)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pins():return json.loads((HERE/'SOURCE_PINS.json').read_text())
def source(path):
    p=Path(path);need(digest(p)==pins()[str(p)]['sha256'],'pinned helper changed: '+str(p));return p.read_text()
def features_module():
    path=HERE.parent/'compact-runtime-feature-closure-v7';sys.path.insert(0,str(path))if str(path)not in sys.path else None
    source(path/'features_v7.py');return importlib.import_module('features_v7')
def budget_admission(free,packed,dense):
    need(type(packed)is int and (256<<20)<=packed<=(8<<30),'packed cache256MiB..8GiB')
    need(type(dense)is int and dense in (0,256<<20),'protected dense cache0/256MiB only')
    need(free>=packed+dense+(2<<30),'cache/dense must leave2GiB for state/temporaries')
    return {'packed_budget_bytes':packed,'protected_dense_budget_bytes':dense,'remaining_reserved_bytes':2<<30}
def build_auxiliary(graph,torch,args,*,mtp,vision):
    """Only selected auxiliary modules. This function never calls Transformer()."""
    graph.world_size=1;graph.rank=0;graph.default_dtype=torch.float8_e4m3fn if args.dtype=='fp8'else torch.bfloat16
    class Auxiliary(torch.nn.Module):
        def __init__(self):
            super().__init__();self.max_seq_len=args.max_seq_len;self.hc_mult=args.hc_mult;self.target_layer_ids=args.dspark_target_layer_ids
            self.embed=graph.ParallelEmbedding(args.vocab_size,args.dim)
            self.head=graph.ParallelHead(args.vocab_size,args.dim,args.norm_eps,args.hc_eps)
            self.mtp=torch.nn.ModuleList([graph.DSparkBlock(args.n_layers+i,args)for i in range(args.n_mtp_layers)]if mtp else [])
            for layer in self.mtp:layer.embed=self.embed;layer.head=self.head
            self.vision=None
            if vision:
                self.vision=graph.ViT(args);self.aligner=graph.Aligner(args)
                self.image_start=torch.nn.Parameter(torch.empty(args.dim));self.image_end=torch.nn.Parameter(torch.empty(args.dim));self.image_newline=torch.nn.Parameter(torch.empty(args.dim))
        def forward(self,*_,**__):raise RuntimeError('Auxiliary has NO target forward')
    model=Auxiliary()
    model.forward_spec=types.MethodType(graph.Transformer.forward_spec,model)
    model.encode_image=types.MethodType(graph.Transformer.encode_image,model)
    need(not hasattr(model,'layers') and len(model.mtp)==(3 if mtp else 0),'no40-layer target allocation')
    return model

def load_bundle_graph(bundle,portable):
    rows={a['runtime_path']:a for a in bundle.config['runtime_asset_map']['items']};names=('engram','vision','image_processor','model')
    saved={n:sys.modules.get(n)for n in ('kernel',*names)}
    try:
        sys.modules['kernel']=portable
        for n in names:
            row=rows['inference/'+n+'.py'];path=bundle.root/row['output_name'];need(digest(path)==row['output_sha256'],'bundle official source changed')
            spec=importlib.util.spec_from_file_location(n,path);module=importlib.util.module_from_spec(spec);sys.modules[n]=module;spec.loader.exec_module(module)
        return sys.modules['model']
    finally:
        for n,v in saved.items():
            if v is None:sys.modules.pop(n,None)
            else:sys.modules[n]=v

def bind_auxiliary(reference,args,streamed):
    # Literal prepared _bind with ONLY irrelevant whole-target hash-state rebuild
    # removed: this auxiliary has neither target layers nor an Engram module.
    tree=ast.parse(source(PREPARED/'runtime/deepseek41_1017/streamed_reference.py'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativeReference')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_bind')
    body=ast.get_source_segment(source(PREPARED/'runtime/deepseek41_1017/streamed_reference.py'),method)
    start=body.index('        # Rebuild the small hashing state');end=body.index('        graph.precompute_freqs_cis.cache_clear()',start)
    body=body[:start]+body[end:]
    import numpy as np,torch
    scope={'np':np,'torch':torch,'types':types};exec(compile(body,str(PREPARED/'runtime/deepseek41_1017/streamed_reference.py'),'exec'),scope)
    scope['_bind'](reference,args)

class AuxiliaryReference:
    """Actual eager DSpark/ViT constructor. No CUDA graph compatibility claim.
    Source K32 packed math unchanged; head arithmetic mode explicit.
    Close before target/Bundle, after all auxiliary consumers finish.
    """
    def __init__(self,bundle,target,*,context,enable_mtp=True,enable_vision=False,packed_budget=2<<30,dense_budget=256<<20,head_mode='source-f32-row128'):
        need(os.environ.get('DS1017_COMPACT_AUXILIARY_V8')=='1','auxiliary default OFF')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(_index_uuids()) and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID','physical GPU0 only')
        need(not bundle.closed and bundle.checkpoint and target.base.checkpoint is bundle.checkpoint,'same current mapped checkpoint/target required')
        need(context in (131072,262144,524288) and type(enable_mtp)is bool and type(enable_vision)is bool and (enable_mtp or enable_vision),'bounded actual auxiliary geometry')
        need(head_mode in MODES,'explicit protected-head arithmetic mode')
        if head_mode!='source-f32-row128':need(os.environ.get('DS1017_AUX_SELECTED_HEAD_V8')=='1' and dense_budget==256<<20,'selected draft-head arithmetic requires separate opt-in and protected256MiB cache')
        self.bundle=bundle;self.target=target;self.context=context;self.head_mode=head_mode;self._closed=False;self._failed=False;self._leases=[];self.model=self.ops=None;self.confidence=[];self.raw_confidence=[]
        self.store=features_module().ProtectedStore(bundle)
        try:
            # The prepared closure is SOURCE only. Never call its old-source constructors/admission.
            closure=PREPARED/'state/fast-source-closure.json';need(digest(closure)=='f5ba5096381b5cc7f995874ab1a1ee17c8652fb6d95b5c34b01a9214b688cf60','prepared source closure changed')
            runtime=str(PREPARED/'runtime');sys.path.insert(0,runtime)if runtime not in sys.path else None
            admission=importlib.import_module('deepseek41_1017.fast_admission');admission.assert_fast_closure(PREPARED,digest(closure))
            import torch,numpy as np
            need(torch.__version__=='2.10.0+cu128' and torch.version.cuda=='12.8','existing admitted DS Torch2.10cu128 runtime required')
            need(not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision()=='highest','protected F32 arithmetic requires original highest/noTF32 mode')
            self.torch=torch;self.device=torch.device('cuda:0')
            free,_=torch.cuda.mem_get_info(self.device);self.budget=budget_admission(free,packed_budget,dense_budget)
            streamed=importlib.import_module('deepseek41_1017.streamed_reference');packed=importlib.import_module('deepseek41_1017.packed_cuda');residency=importlib.import_module('deepseek41_1017.residency');portable=importlib.import_module('deepseek41_1017.portable_kernels');quant=importlib.import_module('deepseek41_1017.sm89_quant')
            graph=load_bundle_graph(bundle,portable);self.graph=graph
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(str(bundle.root),local_files_only=True,trust_remote_code=False)
            config={k:v for k,v in bundle.config.items()if k not in ('native_geometry_contract','runtime_asset_map','compact_mixed_source_provenance')};config.update(max_batch_size=1,max_seq_len=context,temperature=0)
            args=graph.ModelArgs(**config);self.args=args
            # Own exact bundle kernel module; do not call prepared loader's old path guard.
            kernel=next(a for a in bundle.config['runtime_asset_map']['items']if a['runtime_path']=='inference/kernel.py');path=bundle.root/kernel['output_name']
            need(digest(path)==kernel['output_sha256']=='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455','official native operator changed')
            spec=importlib.util.spec_from_file_location('_bundle_native_kernel_v8',path);kernels=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernels)
            owner=self
            class Ops(packed.PackedCudaOperations):
                def __init__(self):
                    streamed.NativeOperations.__init__(self,owner.store,'cuda:0',128,None)
                    self.row_decode_mode='cuda';self.activation_quantizer='sm89-fragment';self.weight_cache=residency.PackedTileCache(packed_budget,self.device);self.kernels=kernels;self.packed_linear_calls=0;self.active_layer=None;self.capture_sink=None;self.aux_dense={};self.aux_dense_bytes=0
                def require_layer(self,name):
                    need(not owner._closed and not owner._failed,'auxiliary closed/failed')
                    need(name.startswith(('mtp.0.','mtp.1.','mtp.2.','vision.','aligner.'))or name in ('head.weight','embed.weight'),'auxiliary names only; target/expert14 forbidden')
                    owner.store.info(name)
                def quantize_activation(self,x):
                    # Exact V3 Heretic fragment wrapper body after new typed-owner admission.
                    need(x.device==self.device and x.dtype==torch.bfloat16 and x.is_contiguous() and x.numel() and x.shape[-1]%32==0,'contiguous native BF16/K32 activation')
                    shape,width=x.shape,x.shape[-1];rows=x.numel()//width;flat=x.reshape(rows,width)
                    if rows%32:flat=torch.nn.functional.pad(flat,(0,0,0,-rows%32))
                    codes=torch.empty_like(flat,dtype=torch.float8_e4m3fn);scales=torch.empty(len(flat),width//32,device=x.device,dtype=torch.float8_e8m0fnu)
                    quant.quant_kernel(width)(flat,codes,scales)
                    return codes[:rows].reshape(shape).contiguous(),scales[:rows].reshape(*shape[:-1],width//32).contiguous()
                def rows(self,name,indices,dtype=torch.bfloat16):
                    if name in owner.shared:
                        ids=torch.as_tensor(np.asarray(indices,dtype=np.int64),device=self.device);return owner.shared[name].index_select(0,ids).to(dtype)
                    if name in self.aux_dense:
                        ids=torch.as_tensor(np.asarray(indices,dtype=np.int64),device=self.device);return self.aux_dense[name].index_select(0,ids).to(dtype)
                    return super().rows(name,indices,dtype)
                def linear(self,name,x,weight_dtype,bias_name=None):
                    self.require_layer(name)
                    if owner.head_mode=='selected-bf16-tc-f32' and name=='head.weight':
                        need(bias_name is None,'source head has no bias');flat=x.reshape(-1,5120).to(torch.bfloat16)
                        return torch.mm(flat,owner.shared[name].T,out_dtype=torch.float32).reshape(*x.shape[:-1],129280)
                    if owner.head_mode=='selected-bf16-tc-f32' and name=='mtp.2.markov_head.head.weight' and name in self.aux_dense:
                        return torch.nn.functional.linear(x.float(),self.aux_dense[name]).to(x.dtype)
                    return super().linear(name,x,weight_dtype,bias_name)
            self.shared={};self.ops=Ops()
            for which,name in ((0,'embed.weight'),(1,'head.weight')):
                lease=target.borrow_dense(which);self._leases.append(lease)
                self.shared[name]=torch.as_tensor(lease,device=self.device).view(torch.bfloat16)
            old=torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with torch.device('meta'):self.model=build_auxiliary(graph,torch,args,mtp=enable_mtp,vision=enable_vision)
            finally:torch.set_default_dtype(old)
            bind_auxiliary(self,args,streamed)
            if enable_mtp:
                # Bounded protected-only cache; NEVER experts/FP8/MX4 expanded.
                names=(('mtp.2.markov_head.embed.weight',torch.bfloat16),('mtp.2.markov_head.head.weight',torch.float32))
                cost=sum(math.prod(self.store.info(n)['shape'])*(2 if t==torch.bfloat16 else 4)for n,t in names)
                if dense_budget:
                    need(cost<=dense_budget,'protected Markov cache exceeds explicit budget')
                    for name,dtype in names:
                        info=self.store.info(name);matrix=self.ops.rows(name,np.arange(info['shape'][0]),dtype);self.ops.aux_dense[name]=matrix;self.ops.aux_dense_bytes+=matrix.numel()*matrix.element_size()
                def embed(this,ids):return torch.nn.functional.embedding(ids,self.shared['embed.weight'])
                self.model.embed.forward=types.MethodType(embed,self.model.embed)
            # Literal original committed-window helper, no full model/reference constructor.
            path=Q.parent/'deepseek41/Strata/backend/deepseek41/speculative.py';tree=ast.parse(source(path));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='commit_main_windows');scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);self._commit=scope['commit_main_windows']
            self.model.eval();self.metadata={'schema':'deepseek1017-compact-auxiliary/8','target_layers_instantiated':0,'draft_layers':len(self.model.mtp),'vision_enabled':enable_vision,'head_mode':head_mode,'activation_mode':'native-source-fp8-mxfp4-k32-fragment-v1','packed_cache_budget':packed_budget,'protected_dense_bytes':self.ops.aux_dense_bytes,'expert_BF16_bank_bytes':0,'source_quantized_weights_unchanged':True,'graph_capture_supported':False,'quality_admitted':False}
        except BaseException:self.close();raise
    def _open(self):need(not self._closed and not self._failed,'closed/failed auxiliary')
    def _tensor(self,main):
        import numpy as np
        t=self.torch;array=np.asarray(main);need(array.dtype==np.uint16 and array.ndim==2 and array.shape[1]==15360 and 1<=len(array)<=128,'raw BF16 committed target rows')
        return t.from_numpy(np.ascontiguousarray(array)).view(t.bfloat16).to(self.device).unsqueeze(0)
    def commit(self,main,first,*,clear=False):
        self._open();need(len(self.model.mtp)==3 and type(clear)is bool,'constructed DSpark and explicit clear flag required');need(type(first)is int and first>=0 and first+len(main)<=self.context,'absolute committed row extent')
        try:
            with self.torch.inference_mode(),self.torch.device(self.device):self._commit(self.model,self.graph,self._tensor(main),first,clear=clear)
        except BaseException:self._failed=True;self.torch.cuda.synchronize(self.device);raise
    def forward_spec(self,known,main,position):
        self._open();need(len(self.model.mtp)==3 and len(main)==1 and type(position)is int and position>=1 and position+6<=self.context,'bounded actual DSpark proposal')
        t=self.torch
        need(known.dtype==t.long and known.device==self.device and tuple(known.shape)==(1,),'one CUDA0 integer known token required')
        try:
            need(not t.cuda.is_current_stream_capturing(),'eager source prototype; graph capture not admitted')
            with t.inference_mode(),t.device(self.device):return self.model.forward_spec(known,self._tensor(main),position)
        except BaseException:self._failed=True;t.cuda.synchronize(self.device);raise
    def forward(self,known,main,position):return self.forward_spec(known,main,position)
    def propose(self,pending,main,position):
        self._open();need(type(pending)is int and 0<=pending<129280,'pending target token')
        t=self.torch
        try:
            known=t.tensor([pending],dtype=t.long,device=self.device);out=self.forward_spec(known,main,position)
            need(out is not None and tuple(out[0].shape)==(1,6)and int(out[0][0,0])==pending and tuple(out[2].shape)==(1,5),'source DSpark output shape')
            need(bool(t.isfinite(out[1]).all()) and bool(t.isfinite(out[2]).all()),'nonfinite draft output')
            self.raw_confidence=[float(v)for v in out[2][0].tolist()];self.confidence=[1/(1+math.exp(-max(-80.,min(80.,v))))for v in self.raw_confidence]
            return [int(v)for v in out[0][0,1:].tolist()]
        except BaseException:self._failed=True;t.cuda.synchronize(self.device);raise
    def image_features(self,images,prompt_length):
        self._open();need(self.model.vision is not None,'vision not constructed')
        try:return features_module().encode_image_features(self,images,prompt_length)
        except BaseException:self._failed=True;self.torch.cuda.synchronize(self.device);raise
    def close(self):
        if getattr(self,'_closed',False):return
        t=getattr(self,'torch',None)
        if t is not None:t.cuda.synchronize(getattr(self,'device',None))
        if self.ops is not None:
            self.ops.weight_cache.clear();self.ops.aux_dense.clear()
        graph=getattr(self,'graph',None)
        if graph is not None:
            for name in ('precompute_freqs_cis','get_dspark_topk_idxs'):
                fn=getattr(graph,name,None)
                if hasattr(fn,'cache_clear'):fn.cache_clear()
            fn=getattr(getattr(graph,'ViT',None),'forward',None)
            cache=getattr(fn,'__globals__',{}).get('get_vision_cos_sin')
            if hasattr(cache,'cache_clear'):cache.cache_clear()
        self.model=self.ops=None;self.shared={}
        import gc
        gc.collect() # Drop original bound-method/module cycles before releasing native leases.
        for lease in reversed(self._leases):lease.close()
        self._leases=[];self._closed=True
        if t is not None:t.cuda.empty_cache() # Explicit end-of-auxiliary phase only.
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
