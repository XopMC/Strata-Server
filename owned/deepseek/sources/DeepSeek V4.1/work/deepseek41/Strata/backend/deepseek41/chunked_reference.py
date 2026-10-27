"""Numerical chunked-prefill adapter, not a production long-context engine.

Keeps the exact pinned graph's projections, vision, Engram and HyperConnections.
Replaces attention cache scheduling/index allocation with tested bounded CPU
oracles. This still executes every backbone layer for every chunk, rather than
the layerwise production schedule. Admission remains capped at4096 tokens until
actual checkpoint/component/quality comparisons pass. Native torch.topk tie
selection is retained within bounded full-prompt-width query score tiles.
"""
import types
import torch
import torch.nn.functional as F

from .long_context import CompressionTail, SlidingWindowTail, select_index_tiled


class ChunkedReference:
    def __init__(self, graph, model, *, query_tile=16, key_tile=256, head_tile=1,
                 tie_policy='native', score_budget_bytes=64*1024**2,score_function=None,
                 candidate_scratch_budget_bytes=64*1024**2,experimental_long_context=False):
        if (model.max_seq_len > 4096 and not experimental_long_context) or graph.world_size != 1:
            raise ValueError('Chunked numerical prototype is single-device and limited to4096')
        if type(experimental_long_context) is not bool or model.max_seq_len > 1048576:
            raise ValueError('Experimental chunked capacity exceeds1048576')
        if min(query_tile, key_tile, head_tile) < 1:
            raise ValueError('Index tiles must be positive')
        self.graph, self.model = graph, model
        self.query_tile, self.key_tile, self.head_tile = query_tile, key_tile, head_tile
        if tie_policy not in ('native','stable') or score_budget_bytes < 1:
            raise ValueError('Invalid index tie policy or score budget')
        self.tie_policy, self.score_budget_bytes = tie_policy, score_budget_bytes
        if score_function is not None and (not callable(score_function) or tie_policy != 'native'):
            raise ValueError('Optional index score function requires native tie policy')
        self.score_function,self.candidate_scratch_budget_bytes = score_function,candidate_scratch_budget_bytes
        self.attentions = [layer.attn for layer in model.layers]
        self.originals = []
        self.position, self.prompt_length = 0, None
        self.images = None
        self.failed = False
        self.sequence_id, self.next_snapshot_id, self.valid_snapshots = 0, 0, {}
        self.call_active, self.full_logits = False, False
        self._patch(model.head, 'forward', self._head)
        self.embedding_hook = model.embed.register_forward_hook(self._embedding)
        for attention in self.attentions:
            self._patch(attention, '_window_kv', self._window)
            if attention.compress_ratio:
                self._patch(attention, '_compress_kv', self._compressed)
                kv_type=getattr(model,'_deepseek41_kv_cache_type','bf16')
                if kv_type in('q4_0','native-fp4'):
                    if kv_type=='q4_0':from .q4_cache import attention_forward
                    else:from .native_fp4_cache import attention_forward
                    self.originals.append((attention,'forward',attention.forward))
                    def packed_forward(this,x,start_pos,graph=graph,forward=attention_forward):
                        return forward(graph,this,x,start_pos)
                    attention.forward=types.MethodType(packed_forward,attention)
        self.reset_states()

    def _patch(self, module, name, function):
        original = getattr(module, name)
        self.originals.append((module, name, original))
        if name == 'forward':
            def head(this, x, full_logits=False):
                return original(x, full_logits=full_logits or (self.call_active and self.full_logits))
            replacement = head
        else:
            def replacement(this, *args):
                return function(this, *args)
        setattr(module, name, types.MethodType(replacement, module))

    def _head(self, *args):
        raise AssertionError('Head replacement is installed by _patch')

    def reset_states(self):
        self.windows = {a.layer_id: SlidingWindowTail(a.window_size) for a in self.attentions}
        self.compressors = {a.layer_id: CompressionTail(a.compress_ratio) for a in self.attentions if a.is_kv_source}
        self.graph.shared_attn.compact_candidates = None

    @torch.inference_mode()
    def begin(self, prompt_length, images=None):
        if prompt_length < 1 or prompt_length > self.model.max_seq_len:
            raise ValueError('Prompt outside admitted numerical context')
        prepared = None
        if images is not None:
            if self.model.vision is None:
                raise ValueError('Model has no image encoder')
            prepared = []
            for sample in images:
                spans, last_end = [], 0
                for image in sorted(sample or (), key=lambda value: value.start):
                    token_types = image.types.to(self.model.image_start.device)
                    end = image.start + token_types.numel()
                    if image.start < last_end or image.start < 0 or end > prompt_length:
                        raise ValueError('Overlapping or out-of-prompt image spans')
                    known = (token_types >= self.graph.IMAGE_START) & (token_types <= self.graph.IMAGE_END)
                    if not known.all():
                        raise ValueError('Unknown image marker type')
                    features = self.model.encode_image(image.patches.to(self.model.image_start.device),
                                                       image.n_vit_h, image.n_vit_w)
                    if int((token_types == self.graph.IMAGE).sum()) != features.shape[0]:
                        raise ValueError('Image span and aligner feature counts disagree')
                    values = self.model.image_start.new_empty(token_types.numel(), self.model.image_start.numel())
                    values[token_types == self.graph.IMAGE_START] = self.model.image_start
                    values[token_types == self.graph.IMAGE_END] = self.model.image_end
                    values[token_types == self.graph.IMAGE_NEW_LINE] = self.model.image_newline
                    values[token_types == self.graph.IMAGE] = features.to(values.dtype)
                    spans.append((image.start, values, token_types.clone()))
                    last_end = end
                prepared.append(spans)
        self.reset_states()
        self.sequence_id += 1
        self.valid_snapshots.clear()
        self.position, self.prompt_length, self.images, self.failed = 0, prompt_length, prepared, False

    def _embedding(self, module, args, h):
        if not self.call_active or self.images is None:
            return h
        if len(self.images) != h.shape[0]:
            raise ValueError('Image batch size disagrees with input tokens')
        for batch, spans in enumerate(self.images):
            for image_start, values, _ in spans:
                first = max(image_start, self.position)
                last = min(image_start+values.shape[0], self.position+h.shape[1])
                if first < last:
                    h[batch, first-self.position:last-self.position] = values[first-image_start:last-image_start].to(h.dtype)
        return h

    def _window(self, attention, x, frequencies, start):
        graph = self.graph
        kv = attention.kv_norm(attention.wkv(x))
        graph.apply_rotary_emb(kv[..., -attention.rope_head_dim:], frequencies)
        graph.act_quant(kv, graph.fp8_block_size, graph.scale_fmt, graph.scale_dtype, True)
        state = self.windows[attention.layer_id]
        values, ids = state.append(kv, start)
        # Keep source prefill padding boundaries for the BF16 sparse kernel's
        #64-position probability blocks, even when a short chunk has fewer KV.
        width = min(attention.window_size, self.prompt_length) if start < self.prompt_length else attention.window_size
        if ids.shape[-1] < width:
            ids = F.pad(ids, (0, width-ids.shape[-1]), value=-1)
        attention.window_kv_cache[:x.shape[0]] = state.ring()
        return values, ids

    def _compressed(self, attention, x, qr, start, offset):
        graph = self.graph
        batch, length, _ = x.shape
        ratio, end = attention.compress_ratio, start+x.shape[1]
        compressed_length = end//ratio
        first_group, latent = start//ratio, None
        if attention.is_kv_source:
            comp = attention.compressor
            state = self.compressors[attention.layer_id]
            if ratio == 1:
                latent, first_group = state.append(comp.norm(comp.wkv(x)), None, start)
            else:
                pooled, first_group = state.append(comp.wkv(x.float()), comp.wgate(x.float()), start)
                latent = comp.norm(pooled.to(x.dtype)) if pooled.shape[1] else None
                if state.kv.shape[1]:
                    comp.kv_state[:batch,:state.kv.shape[1]] = state.kv
                    comp.score_state[:batch,:state.gate.shape[1]] = state.gate
            graph.shared_attn.compress_kv = getattr(attention,'_deepseek41_native_fp4_kv',
                getattr(attention,'_deepseek41_q4_kv',attention.compress_kv_cache))
        if attention.is_index_source:
            indexer = attention.indexer
            if indexer.owns_k:
                if latent is not None:
                    keys = indexer.k_norm(indexer.wk(latent))
                    positions = torch.arange(first_group, first_group+keys.shape[1],device=x.device)*ratio
                    graph.apply_rotary_emb(keys[..., -indexer.rope_head_dim:], attention.freqs_cis[positions])
                    graph.fp4_act_quant(keys, graph.fp4_block_size, True)
                    indexer.k_cache[:batch, first_group:first_group+keys.shape[1]] = keys
                graph.shared_attn.index_k = indexer.k_cache
            if compressed_length:
                q = indexer.wq_b(qr).unflatten(-1, (indexer.n_local_heads, indexer.index_head_dim))
                graph.apply_rotary_emb(q[..., -indexer.rope_head_dim:], attention.freqs_cis[start:end])
                graph.fp4_act_quant(q, graph.fp4_block_size, True)
                weights = indexer.weights_proj(x)*(indexer.softmax_scale*indexer.n_heads**-.5)
                visible = (torch.arange(start+1,end+1,device=x.device)//ratio).expand(batch,-1)
                block_size = indexer.candidate_block_size or 8
                selection = select_index_tiled(q,graph.shared_attn.index_k[:batch,:compressed_length],weights,
                    visible,indexer.index_topk,offset=offset,
                    candidate_topk=indexer.candidate_topk_blocks if indexer.is_candidate_source else 0,
                    candidate_blocks=graph.shared_attn.compact_candidates if indexer.uses_candidates else None,
                    block_size=block_size,query_tile=self.query_tile,key_tile=self.key_tile,head_tile=self.head_tile,
                    tie_policy=self.tie_policy,score_budget_bytes=self.score_budget_bytes,
                    score_function=self.score_function,candidate_scratch_budget_bytes=self.candidate_scratch_budget_bytes,
                    selection_width=max(self.prompt_length,end)//ratio if self.tie_policy == 'native' and start < self.prompt_length else None)
                ids = selection.positions
                if indexer.is_candidate_source:
                    graph.shared_attn.compact_candidates = selection.blocks
            else:
                ids = torch.empty(batch,length,0,dtype=torch.int32,device=x.device)
                if indexer.is_candidate_source:
                    graph.shared_attn.compact_candidates = torch.empty(batch,length,0,dtype=torch.int32,device=x.device)
            width = min(indexer.index_topk, (self.prompt_length if start < self.prompt_length else end)//ratio)
            if ids.shape[-1] < width:
                ids = F.pad(ids, (0,width-ids.shape[-1]),value=-1)
            graph.shared_attn.topk_idxs = ids
        else:
            ids = graph.shared_attn.topk_idxs
        if latent is not None:
            positions = torch.arange(first_group,first_group+latent.shape[1],device=x.device)*ratio
            graph.apply_rotary_emb(latent[..., -attention.rope_head_dim:],attention.freqs_cis[positions])
            native_cache=getattr(attention,'_deepseek41_native_fp4_kv',None)
            if native_cache is not None:native_cache.write(first_group,latent,graph.fp4_act_quant)
            else:
                graph.fp4_act_quant(latent,16,True,scale_dtype=torch.float8_e4m3fn)
                packed_cache=getattr(attention,'_deepseek41_q4_kv',None)
                if packed_cache is not None:packed_cache.write(first_group,latent)
                else:attention.compress_kv_cache[:batch,first_group:first_group+latent.shape[1]] = latent
        cache=graph.shared_attn.compress_kv
        if getattr(self.model,'_deepseek41_kv_cache_type','bf16')in('q4_0','native-fp4'):
            return cache.prefix(compressed_length),ids
        return cache[:batch,:compressed_length],ids

    @torch.inference_mode()
    def forward(self, tokens, start_pos, *, token_types=None, full_logits=False):
        if self.prompt_length is None or self.failed:
            raise ValueError('Begin a sequence or restore its snapshot before forwarding')
        if tokens.ndim != 2 or tokens.shape[0] != 1 or not tokens.shape[1] or start_pos != self.position:
            raise ValueError('Chunked reference requires contiguous nonempty single-batch inputs')
        if start_pos+tokens.shape[1] > self.model.max_seq_len:
            raise ValueError('Input outside admitted numerical context')
        if token_types is not None and token_types.shape != tokens.shape:
            raise ValueError('Image token types disagree with input shape')
        if self.images is not None:
            for batch, spans in enumerate(self.images):
                for image_start, values, expected_types in spans:
                    first, last = max(image_start,start_pos),min(image_start+values.shape[0],start_pos+tokens.shape[1])
                    if first < last:
                        if token_types is None or not torch.equal(
                            token_types[batch,first-start_pos:last-start_pos].to(expected_types.device),
                            expected_types[first-image_start:last-image_start]):
                            raise ValueError('Image spans require matching token types for Engram/routing')
        self.call_active, self.full_logits = True, full_logits
        try:
            result = self.model(tokens,start_pos,token_types=token_types)
        except Exception:
            self.failed = True
            raise
        else:
            self.position += tokens.shape[1]
            return result
        finally:
            self.call_active, self.full_logits = False, False

    def snapshot(self):
        if self.prompt_length is None or self.failed:
            raise ValueError('Cannot snapshot an uninitialized or failed sequence')
        self.next_snapshot_id += 1
        self.valid_snapshots[self.next_snapshot_id] = self.position
        return {'sequence_id':self.sequence_id,'snapshot_id':self.next_snapshot_id,
                'position':self.position,'prompt_length':self.prompt_length,'images':self.images,
                'windows':{key:value.snapshot() for key,value in self.windows.items()},
                'compressors':{key:value.snapshot() for key,value in self.compressors.items()}}

    def restore(self, snapshot):
        # Every compressed/index write begins at floor(position/ratio), beyond
        # the visible prefix. Restore tails/cursors and rings; stale future cache
        # entries are masked until overwritten. Engram cache likewise writes
        # only future positions. This is backbone rollback, not complete DSpark.
        if (snapshot.get('sequence_id') != self.sequence_id or
                self.valid_snapshots.get(snapshot.get('snapshot_id')) != snapshot['position']):
            raise ValueError('Snapshot belongs to an old sequence or discarded speculative branch')
        if set(snapshot['windows']) != set(self.windows) or set(snapshot['compressors']) != set(self.compressors):
            raise ValueError('Snapshot belongs to a different attention graph')
        for key, state in self.windows.items():
            state.restore(snapshot['windows'][key])
        for key, state in self.compressors.items():
            state.restore(snapshot['compressors'][key])
        for attention in self.attentions:
            state = self.windows[attention.layer_id]
            if state.history is None:
                attention.window_kv_cache.zero_()
            else:
                attention.window_kv_cache[:state.history.shape[0]] = state.ring()
            if attention.is_kv_source and attention.compress_ratio > 1:
                tail = self.compressors[attention.layer_id]
                if tail.kv is not None and tail.kv.shape[1]:
                    attention.compressor.kv_state[:tail.kv.shape[0],:tail.kv.shape[1]] = tail.kv
                    attention.compressor.score_state[:tail.gate.shape[0],:tail.gate.shape[1]] = tail.gate
        self.position, self.prompt_length, self.images = snapshot['position'],snapshot['prompt_length'],snapshot['images']
        self.valid_snapshots = {key:position for key,position in self.valid_snapshots.items() if position <= self.position}
        self.failed = False

    def close(self):
        self.embedding_hook.remove()
        for module, name, original in reversed(self.originals):
            setattr(module,name,original)
        self.originals.clear()
