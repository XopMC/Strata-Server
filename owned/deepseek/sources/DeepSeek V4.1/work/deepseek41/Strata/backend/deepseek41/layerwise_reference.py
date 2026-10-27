"""Bounded layerwise prefill scheduling over the chunked numerical graph.

HC streams, Engram hashes and sparse IDs live on CPU; only a token tile is
computed on the selected device. This is still capped at4096 by the adapter.
Packed expert residency, production kernels and checkpoint parity are pending.
Optional DSpark capture retains only the last attention window per target layer,
never the complete prompt's target activations. Draft execution is separate.
"""
from contextlib import nullcontext
import types
import torch
import torch.nn.functional as F


class LayerwisePrefill:
    def __init__(self, adapter, *, tile_sizes=(128,),host_budget_bytes=512*1024**2,span=None,on_progress=None,collect_mtp_tail=False):
        if not tile_sizes or any(type(n) is not int or n < 1 for n in tile_sizes):
            raise ValueError('Layerwise tile sizes must be positive integers')
        if host_budget_bytes < 1:
            raise ValueError('Invalid layerwise host budget')
        self.adapter,self.graph,self.model = adapter,adapter.graph,adapter.model
        self.tile_sizes,self.host_budget_bytes,self.span = tile_sizes,host_budget_bytes,span
        self.on_progress = on_progress
        if type(collect_mtp_tail) is not bool:raise ValueError('MTP tail capture must be boolean')
        self.collect_mtp_tail=collect_mtp_tail
        self.mtp_tail_start=None
        self.indices = self.candidates = self.hashes = None
        self.index_rows_spilled,self.candidate_rows_spilled = 0,0
        self.layer_order = []
        self.last_plan = None

    def tiles(self,length):
        start,index = 0,0
        while start < length:
            stop = min(start+self.tile_sizes[index%len(self.tile_sizes)],length)
            yield start,stop
            start,index = stop,index+1

    def context(self,start,length):
        return self.span(start,length) if self.span is not None else nullcontext()

    def plan(self,length,full_logits):
        model = self.model
        dim,hc = model.layers[0].ffn.dim,model.hc_mult
        index_width = max((min(a.indexer.index_topk,length//a.compress_ratio)
                           for a in self.adapter.attentions if a.is_index_source),default=0)
        candidate_width = max((min(a.indexer.candidate_topk_blocks,
                                   (length//a.compress_ratio+a.indexer.candidate_block_size-1)//a.indexer.candidate_block_size)
                               for a in self.adapter.attentions if a.is_index_source and a.indexer.is_candidate_source),default=0)
        layout = model.engram_layout
        hashes = length*len(layout.layer_ids)*(layout.max_ngram_size-1)*layout.n_heads*8 if layout else 0
        plan = {'hc_double_buffer_bytes':2*length*hc*dim*2,'pre_mix_bytes':length*hc*4,
                'engram_hash_bytes':hashes,'sparse_index_bytes':length*index_width*4,
                'candidate_block_bytes':length*candidate_width*4,
                'full_logits_bytes':length*model.head.vocab_size*4 if full_logits else 0}
        if self.collect_mtp_tail:
            plan['mtp_target_tail_bytes']=min(length,max(a.window_size for a in self.adapter.attentions))*len(model.target_layer_ids)*dim*2
        plan['total_workspace_bytes'] = sum(plan.values())
        return plan

    def _compressed(self,original,attention,x,qr,start,offset):
        graph = self.graph
        end = start+x.shape[1]
        if attention.is_index_source:
            indexer = attention.indexer
            if start == 0:
                graph.shared_attn.topk_idxs = None
                self.indices = None
                width = min(indexer.index_topk,self.length//attention.compress_ratio)
                self.indices = torch.empty(1,self.length,width,dtype=torch.int32,device='cpu')
                if indexer.is_candidate_source:
                    graph.shared_attn.compact_candidates = None
                    self.candidates = None
                    count = min(indexer.candidate_topk_blocks,
                                (self.length//attention.compress_ratio+indexer.candidate_block_size-1)//indexer.candidate_block_size)
                    self.candidates = torch.empty(1,self.length,count,dtype=torch.int32,device='cpu')
            if indexer.uses_candidates:
                if self.candidates is None:
                    raise ValueError('Layerwise candidate source has not completed')
                graph.shared_attn.compact_candidates = self.candidates[:,start:end].to(x.device)
        else:
            if self.indices is None:
                raise ValueError('Layerwise index source has not completed')
            ids = self.indices[:,start:end].to(x.device)
            graph.shared_attn.topk_idxs = torch.where(ids >= 0,ids+offset,-1)
        result = original(x,qr,start,offset)
        if attention.is_index_source:
            ids = result[1]
            self.indices[:,start:end] = torch.where(ids >= 0,ids-offset,-1).cpu()
            self.index_rows_spilled += end-start
            if attention.indexer.is_candidate_source:
                blocks = graph.shared_attn.compact_candidates
                if blocks.shape[-1] < self.candidates.shape[-1]:
                    blocks = F.pad(blocks,(0,self.candidates.shape[-1]-blocks.shape[-1]),value=-1)
                self.candidates[:,start:end] = blocks.cpu()
                self.candidate_rows_spilled += end-start
        return result

    @torch.inference_mode()
    def run(self,tokens,*,images=None,token_types=None,full_logits=False):
        if (tokens.ndim != 2 or tokens.shape[0] != 1 or not tokens.shape[1]
                or tokens.shape[1] > self.model.max_seq_len):
            raise ValueError('Layerwise input outside admitted single-batch numerical context')
        if token_types is not None and token_types.shape != tokens.shape:
            raise ValueError('Layerwise token types disagree')
        length = tokens.shape[1]
        targets=list(self.model.target_layer_ids) if self.collect_mtp_tail else []
        if self.collect_mtp_tail and (not targets or len(set(targets))!=len(targets) or
                                     any(i<0 or i>=len(self.model.layers) for i in targets)):
            raise ValueError('MTP capture requires valid unique target layers')
        plan = self.plan(length,full_logits)
        if plan['total_workspace_bytes'] > self.host_budget_bytes:
            raise ValueError('Layerwise workspace exceeds its host budget')
        self.last_plan,self.length = plan,length
        self.layer_order = []
        self.index_rows_spilled = self.candidate_rows_spilled = 0
        self.adapter.begin(length,images=images)
        device = self.model.layers[0].attn.window_kv_cache.device
        dim,hc = self.model.layers[0].ffn.dim,self.model.hc_mult
        h_a = torch.empty(1,length,hc,dim,dtype=torch.bfloat16,device='cpu')
        h_b = torch.empty_like(h_a)
        mixes = torch.empty(1,length,hc,dtype=torch.float32,device='cpu')
        originals = []
        tail_length=min(length,max(a.window_size for a in self.adapter.attentions)) if targets else 0
        self.mtp_tail_start=length-tail_length if targets else None
        main_tail=torch.empty(1,tail_length,len(targets)*dim,dtype=torch.bfloat16,device='cpu') if targets else None
        try:
            for attention in self.adapter.attentions:
                if not attention.compress_ratio:
                    continue
                original = attention._compress_kv
                originals.append((attention,original))
                def compressed(this,x,qr,start,offset,original=original):
                    return self._compressed(original,this,x,qr,start,offset)
                attention._compress_kv = types.MethodType(compressed,attention)
            for start,end in self.tiles(length):
                ids = tokens[:,start:end].to(device)
                types_tile = token_types[:,start:end].to(device) if token_types is not None else None
                self.adapter.position,self.adapter.call_active = start,True
                with self.context(start,end-start):
                    embedded = self.model.embed(ids)
                self.adapter.call_active = False
                if embedded.dtype != torch.bfloat16:
                    raise ValueError('Layerwise baseline expects native BF16 embedding outputs')
                h = embedded.unsqueeze(2).repeat(1,1,hc,1)
                h_a[:,start:end] = h.cpu()
                mixes[:,start:end] = self.graph.make_identity_pre_mix(h,hc).cpu()
                if self.model.engram_hash is not None:
                    hashes = self.model.engram_hash(ids,start,None if types_tile is None else types_tile < 0)
                    if self.hashes is None:
                        self.hashes = torch.empty(1,length,*hashes.shape[2:],dtype=hashes.dtype,device='cpu')
                    self.hashes[:,start:end] = hashes.cpu()
            for layer in self.model.layers:
                self.layer_order.append(layer.attn.layer_id)
                for start,end in self.tiles(length):
                    h = h_a[:,start:end].to(device)
                    pre_mix = mixes[:,start:end].to(device)
                    types_tile = token_types[:,start:end].to(device) if token_types is not None else None
                    image_mask = types_tile >= 0 if types_tile is not None else None
                    with self.context(start,end-start):
                        if layer.engram is not None:
                            hashes = self.hashes[:,start:end,layer.engram.layer_hash_index].to(device)
                            h = layer.engram(h,hashes,None if image_mask is None else ~image_mask)
                        if layer.attn.layer_id in targets and end>self.mtp_tail_start:
                            first=max(start,self.mtp_tail_start)
                            slot=targets.index(layer.attn.layer_id)
                            main_tail[:,first-self.mtp_tail_start:end-self.mtp_tail_start,slot*dim:(slot+1)*dim]=h[:,first-start:].mean(2).cpu()
                        h,pre_mix = layer(h,start,pre_mix,image_mask)
                    h_b[:,start:end] = h.cpu()
                    mixes[:,start:end] = pre_mix.cpu()
                    if self.on_progress is not None:
                        self.on_progress({'phase':'layerwise_prefill','layer':layer.attn.layer_id,
                            'processed_layer_tokens':end,'prompt_tokens':length,'layer_complete':end == length})
                h_a,h_b = h_b,h_a
            if full_logits:
                logits = torch.empty(1,length,self.model.head.vocab_size,dtype=torch.float32,device=device)
                for start,end in self.tiles(length):
                    with self.context(start,end-start):
                        h = self.model.layers[-1].hc_pre(h_a[:,start:end].to(device),mixes[:,start:end].to(device))
                        logits[:,start:end] = self.model.head(self.model.norm(h),full_logits=True)
            else:
                h = self.model.layers[-1].hc_pre(h_a[:,-1:].to(device),mixes[:,-1:].to(device))
                logits = self.model.head(self.model.norm(h))
            self.adapter.position = length
            self.adapter.failed = False
            # CPU slice views must not retain the entire host spill matrices.
            if self.graph.shared_attn.topk_idxs is not None:
                self.graph.shared_attn.topk_idxs = self.graph.shared_attn.topk_idxs.clone()
            if self.graph.shared_attn.compact_candidates is not None:
                self.graph.shared_attn.compact_candidates = self.graph.shared_attn.compact_candidates.clone()
            return self.graph.sample(logits,self.model.temperature),logits,main_tail.to(device) if main_tail is not None else None
        except Exception:
            self.adapter.failed = True
            raise
        finally:
            self.adapter.call_active = False
            for attention,original in originals:
                attention._compress_kv = original
            self.indices = self.candidates = self.hashes = None
