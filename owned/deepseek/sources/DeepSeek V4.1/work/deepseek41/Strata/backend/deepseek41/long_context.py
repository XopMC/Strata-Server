"""Bounded CPU oracles for native two-level index selection and chunked pooling.

This module does not provide a production prefill engine. It avoids the source's
quadratic-sized score/mask allocations, while retaining exact index quantization
inputs and BF16 head-product/reduction boundaries. Compute before the candidate
filter is still quadratic. Native selection retains one bounded query tile of
full-width scores, preserving torch.topk's tie behavior. The optional online
policy uses smaller IDs first and requires separate checkpoint quality checks.
"""
from dataclasses import dataclass
import torch


@dataclass
class IndexSelection:
    positions: torch.Tensor
    blocks: torch.Tensor | None
    peak_score_elements: int


def _merge_topk(scores, ids, new_scores, new_ids, count):
    """Keep bounded score/ID pairs, with a deterministic ID tie break."""
    if scores is not None:
        new_scores = torch.cat((scores, new_scores), -1)
        new_ids = torch.cat((ids, new_ids), -1)
    by_id = new_ids.argsort(dim=-1, stable=True)
    new_scores, new_ids = new_scores.gather(-1, by_id), new_ids.gather(-1, by_id)
    by_score = new_scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
    return new_scores.gather(-1, by_score), new_ids.gather(-1, by_score)


def _score_tile(query, keys, weights, head_tile):
    """Native: BF16 dot -> relu -> BF16 weight product -> float sum -> BF16."""
    # keys are [B,K,D] or query-specific [B,Q,K,D]. Never materialize all heads.
    result = torch.zeros(query.shape[:2] + (keys.shape[-2],), device=query.device, dtype=torch.float32)
    for start in range(0, query.shape[2], head_tile):
        stop = min(start + head_tile, query.shape[2])
        if keys.ndim == 3:
            dots = torch.einsum('bqhd,bkd->bqhk', query[:, :, start:stop], keys)
        else:
            dots = torch.einsum('bqhd,bqkd->bqhk', query[:, :, start:stop], keys)
        product = dots.relu() * weights[:, :, start:stop].unsqueeze(-1)
        result += product.float().sum(2)
    return result.to(query.dtype)


def _select_native(query, keys, weights, visible_lengths, topk, *, offset,
                   candidate_topk, candidate_blocks, block_size, query_tile,
                   key_tile, head_tile, width, score_budget_bytes,score_function,
                   candidate_scratch_budget_bytes):
    """Source topk semantics, with full-width scores only for bounded query rows.

    Future positions occupy -inf slots even when their keys have not been
    computed. This retains prefill ties, including source selection of visible
    masked IDs when candidate consumers have fewer finite scores than topk.
    """
    batch, length, heads, _ = query.shape
    cache_width = keys.shape[1]
    scan_width = (candidate_blocks.shape[-1]*block_size if candidate_blocks is not None else cache_width)
    row_bytes = batch * (width+(scan_width if score_function is not None else 0)) * query.element_size()
    if row_bytes > score_budget_bytes:
        raise ValueError('Native topk score budget cannot hold one query row')
    query_tile = min(query_tile, score_budget_bytes // max(row_bytes, 1))
    if score_function is not None and candidate_blocks is not None and scan_width:
        gather_bytes = batch*scan_width*keys.shape[-1]*keys.element_size()
        if gather_bytes > candidate_scratch_budget_bytes:
            raise ValueError('Native candidate gather cannot hold one query within its scratch budget')
        query_tile = min(query_tile,candidate_scratch_budget_bytes//gather_bytes)
    count = min(topk, width)
    block_count = min(candidate_topk, (width+block_size-1)//block_size)
    positions = torch.full((batch, length, count), -1, dtype=torch.int32, device=query.device)
    blocks = torch.full((batch, length, block_count), -1, dtype=torch.int32, device=query.device) if candidate_topk else None
    peak = 0
    if not width:
        return IndexSelection(positions, blocks, peak)
    for first in range(0, length, query_tile):
        last = min(first+query_tile, length)
        q, w = query[:,first:last], weights[:,first:last]
        visible = visible_lengths[:,first:last].unsqueeze(-1)
        scores = query.new_full((batch,last-first,width), -torch.inf)
        peak = max(peak, scores.numel())
        selected = candidate_blocks[:,first:last].long() if candidate_blocks is not None else None
        scan_width = selected.shape[-1]*block_size if selected is not None else cache_width
        if score_function is not None and scan_width and cache_width:
            if selected is None:
                ids = torch.arange(cache_width,device=query.device).expand(batch,last-first,-1)
                values = score_function(q,keys,w,score_budget_bytes=score_budget_bytes-scores.numel()*scores.element_size())
            else:
                relative = torch.arange(scan_width,device=query.device)
                chosen = selected[...,relative//block_size]
                ids = (chosen*block_size+relative%block_size).masked_fill(chosen < 0,-1)
                gathered = keys[torch.arange(batch,device=query.device)[:,None,None],ids.clamp(0,cache_width-1)]
                flat_q = q.reshape(batch*(last-first),1,heads,q.shape[-1])
                flat_w = w.reshape(batch*(last-first),1,heads)
                flat_keys = gathered.reshape(batch*(last-first),scan_width,keys.shape[-1])
                values = score_function(flat_q,flat_keys,flat_w,
                    score_budget_bytes=score_budget_bytes-scores.numel()*scores.element_size())
                if values.shape != (batch*(last-first),1,scan_width):
                    raise ValueError('Native score function contract shape mismatch')
                values = values.reshape(batch,last-first,scan_width)
            if values.shape != (batch,last-first,scan_width) or values.dtype != query.dtype or values.device != query.device:
                raise ValueError('Native score function contract shape/dtype/device mismatch')
            values.masked_fill_((ids < 0) | (ids >= visible) | (ids >= cache_width),-torch.inf)
            peak = max(peak,scores.numel()+values.numel())
            if selected is None:
                scores[...,:cache_width] = values
            else:
                scores.scatter_reduce_(-1,ids.clamp(0,width-1),values,reduce='amax',include_self=True)
                del gathered,flat_keys,flat_q,flat_w
            del values,ids
        for kfirst in range(0,scan_width if score_function is None else 0,key_tile):
            klast = min(kfirst+key_tile,scan_width)
            if selected is None:
                ids = torch.arange(kfirst,klast,device=query.device).expand(batch,last-first,-1)
                tile_keys = keys[:,kfirst:klast]
            else:
                relative = torch.arange(kfirst,klast,device=query.device)
                chosen = selected[...,relative//block_size]
                ids = (chosen*block_size+relative%block_size).masked_fill(chosen < 0,-1)
                if not cache_width:
                    continue
                tile_keys = keys[torch.arange(batch,device=query.device)[:,None,None],
                                 ids.clamp(0,cache_width-1)]
            values = _score_tile(q,tile_keys,w,head_tile)
            values.masked_fill_((ids < 0) | (ids >= visible) | (ids >= cache_width), -torch.inf)
            peak = max(peak,batch*(last-first)*(klast-kfirst)*min(heads,head_tile))
            if selected is None:
                scores[...,kfirst:klast] = values
            else:
                # Invalid candidate slots may clamp onto a valid slot: max with
                # -inf avoids overwriting its already computed score.
                scores.scatter_reduce_(-1,ids.clamp(0,width-1),values,reduce='amax',include_self=True)
        if blocks is not None:
            block_scores = torch.nn.functional.pad(scores,(0,-width%block_size),value=-torch.inf)
            block_scores = block_scores.unflatten(-1,(-1,block_size)).amax(-1)
            newest = (visible-1)//block_size
            block_scores.masked_fill_(torch.arange(block_scores.shape[-1],device=query.device) == newest,torch.inf)
            top = block_scores.topk(block_count,dim=-1)
            blocks[:,first:last] = top.indices.masked_fill(top.values == -torch.inf,-1).sort(-1).values.int()
        ids = scores.topk(count,dim=-1,sorted=False).indices.sort(-1).values
        positions[:,first:last] = torch.where(ids < visible,ids+offset,-1).int()
    return IndexSelection(positions,blocks,peak)


def select_index_tiled(query, keys, weights, visible_lengths, topk, *, offset=0,
                       candidate_topk=0, candidate_blocks=None, block_size=8,
                       query_tile=16, key_tile=256, head_tile=1, tie_policy='stable',
                       selection_width=None, score_budget_bytes=64*1024**2,
                       score_function=None,candidate_scratch_budget_bytes=64*1024**2):
    """Return sparse positions and optional compact block IDs, never a dense mask.

    query/keys must already contain the native FP4-quantized values after RoPE.
    visible_lengths is [B,Q] and counts COMPLETE compression groups. Source
    selection streams every key; consumers gather only previously selected
    blocks. -1 marks unreachable position/block slots. Output is position-sorted.
    Native tie policy uses the full prompt selection_width during chunked prefill,
    bounded by score_budget_bytes per query tile. Stable policy streams top-k.
    """
    if query.ndim != 4 or keys.ndim != 3 or weights.shape != query.shape[:3]:
        raise ValueError('Index query/key/weight shapes disagree')
    batch, length, heads, dim = query.shape
    width = keys.shape[1]
    if keys.shape[0] != batch or keys.shape[2] != dim or visible_lengths.shape != (batch, length):
        raise ValueError('Index key/visibility shapes disagree')
    if query.dtype not in (torch.bfloat16, torch.float32) or keys.dtype != query.dtype or weights.dtype != query.dtype:
        raise ValueError('Index values require matching BF16 or float32 dtypes')
    if min(query_tile, key_tile, head_tile, block_size, topk) < 1 or candidate_topk < 0:
        raise ValueError('Tile sizes and topk must be positive')
    if key_tile % block_size:
        raise ValueError('Key tiles must align to candidate blocks')
    if candidate_topk and candidate_blocks is not None:
        raise ValueError('A candidate source cannot consume candidate blocks')
    if tie_policy not in ('stable','native'):
        raise ValueError('Unknown index tie policy')
    if score_function is not None and (not callable(score_function) or tie_policy != 'native'):
        raise ValueError('An optional score function requires native tie policy')
    if candidate_scratch_budget_bytes < 1:
        raise ValueError('Invalid candidate gather scratch budget')
    selection_width = width if selection_width is None else selection_width
    if selection_width < width or score_budget_bytes < 1:
        raise ValueError('Selection width or score budget outside allowed bounds')
    if tie_policy == 'stable' and selection_width != width:
        raise ValueError('Full prompt selection width requires native tie policy')
    if torch.any(visible_lengths < 0) or torch.any(visible_lengths > width):
        raise ValueError('Visibility outside the available compressed cache')
    if candidate_blocks is not None:
        if candidate_blocks.ndim != 3 or candidate_blocks.shape[:2] != (batch, length):
            raise ValueError('Candidate block shape disagrees')
        if torch.any(candidate_blocks < -1) or torch.any(candidate_blocks >= (width+block_size-1)//block_size):
            raise ValueError('Candidate block outside compressed cache')
        sorted_blocks = candidate_blocks.sort(-1).values
        if torch.any((sorted_blocks[..., 1:] == sorted_blocks[..., :-1]) & (sorted_blocks[..., 1:] >= 0)):
            raise ValueError('Duplicate candidate block would duplicate attention positions')
    if tie_policy == 'native':
        return _select_native(query,keys,weights,visible_lengths,topk,offset=offset,
                              candidate_topk=candidate_topk,candidate_blocks=candidate_blocks,
                              block_size=block_size,query_tile=query_tile,key_tile=key_tile,
                              head_tile=head_tile,width=selection_width,score_budget_bytes=score_budget_bytes,
                              score_function=score_function,candidate_scratch_budget_bytes=candidate_scratch_budget_bytes)
    count = min(topk, width)
    block_count = min(candidate_topk, (width+block_size-1)//block_size)
    positions = torch.full((batch, length, count), -1, dtype=torch.int32, device=query.device)
    blocks = torch.full((batch, length, block_count), -1, dtype=torch.int32, device=query.device) if candidate_topk else None
    peak = 0
    if width == 0:
        return IndexSelection(positions, blocks, peak)
    for qstart in range(0, length, query_tile):
        qend = min(qstart + query_tile, length)
        q, w = query[:, qstart:qend], weights[:, qstart:qend]
        visible = visible_lengths[:, qstart:qend].unsqueeze(-1)
        scores = ids = block_scores = block_ids = None
        if candidate_blocks is None:
            scan_width = width
        else:
            selected = candidate_blocks[:, qstart:qend].long()
            scan_width = selected.shape[-1] * block_size
        for kstart in range(0, scan_width, key_tile):
            kend = min(kstart+key_tile, scan_width)
            if candidate_blocks is None:
                tile_ids = torch.arange(kstart, kend, device=query.device).expand(batch, qend-qstart, -1)
                tile_keys = keys[:, kstart:kend]
            else:
                relative = torch.arange(kstart, kend, device=query.device)
                tile_blocks = selected[..., relative // block_size]
                tile_ids = tile_blocks * block_size + relative % block_size
                tile_ids = tile_ids.masked_fill(tile_blocks < 0, -1)
                tile_keys = keys[torch.arange(batch, device=query.device)[:, None, None], tile_ids.clamp(0, max(width-1, 0))]
            values = _score_tile(q, tile_keys, w, head_tile)
            valid = (tile_ids >= 0) & (tile_ids < visible) & (tile_ids < width)
            values.masked_fill_(~valid, -torch.inf)
            peak = max(peak, batch*(qend-qstart)*(kend-kstart)*min(heads, head_tile))
            scores, ids = _merge_topk(scores, ids, values, tile_ids, count)
            if candidate_topk:
                padding = -(kend-kstart) % block_size
                padded = torch.nn.functional.pad(values, (0, padding), value=-torch.inf)
                local_scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
                local_ids = torch.arange(kstart//block_size, (kend+block_size-1)//block_size, device=query.device)
                local_ids = local_ids.expand_as(local_scores)
                last = (visible - 1) // block_size
                local_scores.masked_fill_((local_ids == last) & (visible > 0), torch.inf)
                block_scores, block_ids = _merge_topk(block_scores, block_ids, local_scores, local_ids, block_count)
        if scores is not None:
            # Filtering -inf also guards candidate consumers with fewer than k
            # valid positions. Source may emit masked-but-visible IDs in this
            # degenerate setting; production top2048*8 vs index512 avoids it.
            final_ids = ids.masked_fill(~torch.isfinite(scores), width)
            final_ids = final_ids.sort(-1).values
            positions[:, qstart:qend, :final_ids.shape[-1]] = torch.where(final_ids < width, final_ids+offset, -1).int()
        if blocks is not None and block_scores is not None:
            final_blocks = block_ids.masked_fill(block_scores == -torch.inf, -1).sort(-1).values
            blocks[:, qstart:qend, :final_blocks.shape[-1]] = final_blocks.int()
    return IndexSelection(positions, blocks, peak)


class CompressionTail:
    """Carry projected KV/gate tails across arbitrary contiguous token chunks.

    Pooling operates in float32 before the native RMS norm and quantization.
    Callers supply projected KV and gate values, preserving native linear dtypes.
    Snapshot/restore support is a component of future speculative rollback.
    """
    def __init__(self, ratio):
        if ratio < 1:
            raise ValueError('Compression ratio must be positive')
        self.ratio = ratio
        self.position = 0
        self.kv = self.gate = None

    def append(self, kv, gate, start_pos):
        if start_pos != self.position or kv.ndim != 3:
            raise ValueError('Compression chunks must be contiguous [B,S,D]')
        if self.ratio > 1 and (gate is None or gate.shape != kv.shape):
            raise ValueError('Compression gate shape disagrees')
        first_group = start_pos // self.ratio
        if self.ratio == 1:
            self.position += kv.shape[1]
            return kv, first_group
        values, scores = kv.float(), gate.float()
        if self.kv is not None:
            values = torch.cat((self.kv, values), 1)
            scores = torch.cat((self.gate, scores), 1)
        cutoff = values.shape[1] // self.ratio * self.ratio
        self.kv, self.gate = values[:, cutoff:].clone(), scores[:, cutoff:].clone()
        self.position += kv.shape[1]
        pooled = (values[:, :cutoff].unflatten(1, (-1, self.ratio)) *
                  scores[:, :cutoff].unflatten(1, (-1, self.ratio)).softmax(2)).sum(2)
        return pooled.to(kv.dtype), first_group

    def snapshot(self):
        return (self.position, None if self.kv is None else self.kv.clone(),
                None if self.gate is None else self.gate.clone())

    def restore(self, state):
        position, kv, gate = state
        if position < 0 or (kv is None) != (gate is None):
            raise ValueError('Invalid compression snapshot')
        if kv is not None and (kv.shape != gate.shape or kv.shape[1] != position % self.ratio):
            raise ValueError('Compression snapshot tail shape disagrees')
        self.position, self.kv, self.gate = position, None if kv is None else kv.clone(), None if gate is None else gate.clone()


class SlidingWindowTail:
    """Bounded chronological window across arbitrary prefill/decode chunks.

    Inputs are the native post-RoPE, FP8-quantized KV values. Returned sparse IDs
    index only [previous window tail + current chunk], and remain causal within
    the chunk. ring() reproduces the source's final decode cache layout.
    """
    def __init__(self, window):
        if window < 1:
            raise ValueError('Sliding window must be positive')
        self.window, self.position, self.history = window, 0, None

    def append(self, kv, start_pos):
        if kv.ndim != 3 or kv.shape[1] == 0 or start_pos != self.position:
            raise ValueError('Window chunks must be nonempty and contiguous [B,S,D]')
        if self.history is not None and (self.history.shape[0] != kv.shape[0] or
                                        self.history.shape[-1] != kv.shape[-1] or
                                        self.history.dtype != kv.dtype or self.history.device != kv.device):
            raise ValueError('Window KV layout changed within a sequence')
        previous = self.history[:, -(self.window-1):] if self.history is not None and self.window > 1 else kv[:, :0]
        values = torch.cat((previous, kv), 1)
        end = torch.arange(previous.shape[1], values.shape[1], device=kv.device).unsqueeze(-1)
        idxs = (end-self.window+1).clamp_min(0) + torch.arange(min(self.window,values.shape[1]), device=kv.device)
        idxs = idxs.masked_fill(idxs > end, -1).int()
        idxs = idxs.unsqueeze(0).expand(kv.shape[0], -1, -1).contiguous()
        self.history = values[:, -self.window:].clone()
        self.position += kv.shape[1]
        return values, idxs

    def ring(self):
        if self.history is None:
            raise ValueError('Window has no KV values yet')
        history = self.history
        cache = history.new_zeros(history.shape[0],self.window,history.shape[-1])
        positions = torch.arange(self.position-history.shape[1],self.position,device=history.device) % self.window
        cache[:, positions] = history
        return cache

    def snapshot(self):
        return self.position, None if self.history is None else self.history.clone()

    def restore(self, state):
        position, history = state
        if position < 0 or (history is None and position != 0) or (history is not None and
                (history.ndim != 3 or history.shape[1] != min(position,self.window))):
            raise ValueError('Invalid window snapshot')
        self.position, self.history = position, None if history is None else history.clone()
