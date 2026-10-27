from .cuda_admission import cuda0_mask_admitted
"""Bounded native-weight adapter for the pinned reference graph.

This slow baseline is for numerical comparisons. It is not the final Strata
engine and deliberately refuses contexts above4096 until production scheduling
and checkpoint quality pass. Optional chunked attention is a numerical adapter.
No full expert/Engram table is ever dequantized.
"""
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .native import bf16
from .portable_kernels import act_quant
from .reference_graph import load_graph, publish_index_owners
from .weights import NativeStore


class NativeOperations:
    def __init__(self, store, device='cpu', row_chunk=128, observer=None):
        if row_chunk < 1 or row_chunk > 4096:
            raise ValueError('Invalid bounded row chunk')
        self.store, self.device, self.row_chunk = store, torch.device(device), row_chunk
        self.rows_read = 0
        self.linear_calls = 0
        self.observer = observer

    def _row_chunk(self, name):
        info = self.store.info(name)
        budget = getattr(self.store,'read_limit_bytes',None)
        if budget is None:
            return self.row_chunk
        row_bytes = info['bytes']//info['shape'][0]
        if info['dtype']=='D41Q':
            row_bytes=max(row_bytes,info['shape'][1]*4)
        if row_bytes < 1 or row_bytes > budget:
            raise ValueError('Single tensor row exceeds the bounded read budget')
        return min(self.row_chunk,budget//row_bytes)

    def rows(self, name, indices, dtype=torch.bfloat16):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        chunks = []
        row_chunk = self._row_chunk(name)
        for start in range(0, len(indices), row_chunk):
            selected = indices[start:start+row_chunk]
            values = self.store.float_rows(name, selected)
            chunks.append(torch.from_numpy(values).to(device=self.device, dtype=dtype))
            self.rows_read += len(selected)
        shape = self.store.info(name)['shape']
        logical_width = shape[1] * (2 if self.store.info(name)['dtype'] == 'I8' else 1)
        return torch.cat(chunks) if chunks else torch.empty(0, logical_width, device=self.device, dtype=dtype)

    def linear(self, name, x, weight_dtype, bias_name=None):
        if self.observer is not None:
            self.observer(name, x.detach())
        info = self.store.info(name)
        quantized = info['dtype'] in ('I8', 'F8_E4M3', 'D41Q')
        output_dtype = torch.bfloat16 if quantized else x.dtype
        if quantized:
            q, scale = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
            activation = (q.float().unflatten(-1, (-1, 32)) * scale.float().unsqueeze(-1)).flatten(-2)
            compute_dtype = torch.float32
        else:
            compute_dtype = weight_dtype
            activation = x.to(compute_dtype)
        # Native nn.Linear in ViT/aligner fuses bias before BF16 output
        # rounding. A separate BF16 addition introduces a second rounding.
        bias = self.small(bias_name,compute_dtype) if bias_name is not None and not quantized else None
        if not quantized and name.startswith(('vision.','aligner.')):
            # Preserve the source nn.Linear GEMM shape. Splitting its output
            # channels changes CUDA BF16 reduction algorithms and compounds
            # differences through the32 ViT blocks. Each source row read stays
            # bounded; only this one dense matrix is materialized (<=128MiB).
            matrix_bytes = int(np.prod(info['shape']))*torch.empty((),dtype=compute_dtype,device='meta').element_size()
            if matrix_bytes > 128*1024**2:
                raise ValueError('Single native vision matrix exceeds128MiB materialization budget')
            weight = self.rows(name,np.arange(info['shape'][0]),compute_dtype)
            output = F.linear(activation,weight,bias).to(output_dtype)
            self.linear_calls += 1
            return output
        output = torch.empty(*x.shape[:-1], info['shape'][0], device=x.device, dtype=output_dtype)
        row_chunk = self._row_chunk(name)
        for start in range(0, info['shape'][0], row_chunk):
            indices = np.arange(start, min(start + row_chunk, info['shape'][0]))
            weight = self.rows(name, indices, compute_dtype)
            output[..., start:start+len(indices)] = F.linear(activation, weight,
                bias[start:start+len(indices)] if bias is not None else None).to(output_dtype)
        if bias_name is not None and quantized:
            output += self.small(bias_name, output_dtype)
        self.linear_calls += 1
        return output

    def small(self, name, dtype):
        info = self.store.info(name)
        raw = self.store.raw(name)
        if info['dtype'] == 'BF16':
            array = bf16(raw).reshape(info['shape'])
        elif info['dtype'] == 'F32':
            array = np.frombuffer(raw, dtype='<f4').copy().reshape(info['shape'])
        else:
            raise ValueError('Unsupported small parameter representation: ' + name)
        return torch.from_numpy(array).to(self.device, dtype)


def _chunked_score_function(index_mode,device):
    if index_mode not in ('tiled','sm89') or (index_mode == 'sm89' and device.type != 'cuda'):
        raise ValueError('Invalid chunked index mode/device')
    if index_mode == 'sm89':
        from .sm89_index import score_index_with_small_fallback
        return score_index_with_small_fallback
    return None


class NativeReference:
    def __init__(self, project_root, device='cpu', context=4096, row_chunk=128,linear_mode='portable',attention_mode='portable',weight_cache_bytes=0,
                 row_decode_mode='cpu',activation_quantizer='sm89-fragment',bundle_path=None,borrowed_store=None):
        project_root = Path(project_root).resolve()
        if context < 2 or context > 4096:
            raise ValueError('Unvalidated production scheduling: numerical context currently limited to4096')
        device = torch.device(device)
        if linear_mode not in ('portable','native-cuda') or (linear_mode == 'native-cuda' and device.type != 'cuda'):
            raise ValueError('Invalid numerical linear mode/device')
        if attention_mode not in ('portable','sm89') or (attention_mode == 'sm89' and device.type != 'cuda'):
            raise ValueError('Invalid numerical attention mode/device')
        if type(weight_cache_bytes) is not int or weight_cache_bytes < 0 or (weight_cache_bytes and linear_mode != 'native-cuda'):
            raise ValueError('Packed weight residency requires nonnegative bytes and native-CUDA linear mode')
        if row_decode_mode not in ('cpu','cuda') or (row_decode_mode == 'cuda' and linear_mode != 'native-cuda'):
            raise ValueError('CUDA row decode requires native-CUDA operations')
        if activation_quantizer not in ('legacy','sm89-fragment'):
            raise ValueError('Unknown native activation quantizer')
        if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and device.index not in (None, 0)):
            raise ValueError('Reference device must be CPU or the masked CUDA0')
        # Guard before any GPU allocation or full-model work, including direct imports.
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location('deepseek41_admission', project_root / 'tools/run_guarded.py')
        guard = module_from_spec(spec)
        spec.loader.exec_module(guard)
        guard.admit('cpu' if device.type == 'cpu' else 'gpu0')
        if device.type == 'cuda':
            import os
            if not cuda0_mask_admitted():
                raise ValueError('GPU reference requires the explicit physical GPU0 UUID mask')
        self._owns_store=borrowed_store is None
        if borrowed_store is not None:
            if bundle_path is not None:
                raise ValueError('Choose an existing verified store or a bundle path')
            if device.type=='cuda':
                report=getattr(borrowed_store,'ram_preload_report',None)or{}
                manifest=getattr(borrowed_store,'manifest',{})
                expected={name:row for name,row in borrowed_store.verified.items()if row.get('kind')=='weights'}
                records=report.get('shards',[])
                if(manifest.get('format')!='deepseek41-packed-bundle-v1'or report.get('phase')!='all_candidate_shards_sha256_verified_in_ram'or
                   not expected or set(borrowed_store.ram_shards)!=set(expected)or {v['name']for v in records}!=set(expected)or len(records)!=len(expected)):
                    raise ValueError('GPU borrowing requires the complete SHA-verified candidate RAM owner')
                for row in records:
                    array=borrowed_store.ram_shards[row['name']];entry=expected[row['name']]
                    if(array.dtype!=np.uint8 or not array.flags.c_contiguous or array.flags.writeable or array.nbytes!=row['bytes']or array.nbytes!=entry['size']or row['sha256']!=entry['digest']or ('buffer_address'in row and row['buffer_address']!=array.ctypes.data)):
                        raise ValueError('Borrowed candidate RAM identity changed')
                if sum(a.nbytes for a in borrowed_store.ram_shards.values())!=report.get('resident_bytes')or report['resident_bytes']!=manifest['payload_bytes']:
                    raise ValueError('Borrowed candidate RAM byte inventory differs')
            report=getattr(borrowed_store,'ram_preload_report',None)
            if not report or report.get('source_revision')!=guard.REVISION or not report.get('resident_bytes'):
                raise ValueError('Borrowed source RAM lacks complete checkpoint integrity proof')
            self.store=borrowed_store
            if getattr(borrowed_store,'manifest',{}).get('format')=='deepseek41-packed-bundle-v1':
                runtime_config=borrowed_store.source/'runtime-config.json';tokenizer_root=borrowed_store.source
            else:runtime_config=project_root/'reference/inference/config.json';tokenizer_root=project_root/'source-metadata'
        elif bundle_path is None:
            status = json.loads((project_root / 'state/download-status.json').read_text())
            self.store = NativeStore(status['destination'], project_root / 'state/download-status.json')
            runtime_config = project_root/'reference/inference/config.json'
            tokenizer_root = project_root/'source-metadata'
        else:
            from .bundle import BundleStore
            self.store = BundleStore(bundle_path)
            runtime_config = self.store.source/'runtime-config.json'
            tokenizer_root = self.store.source
        if linear_mode == 'native-cuda':
            from .packed_cuda import PackedCudaOperations
            self.ops = PackedCudaOperations(self.store,project_root/'reference/inference',device,row_chunk,
                                           weight_cache_bytes=weight_cache_bytes,row_decode_mode=row_decode_mode,
                                           activation_quantizer=activation_quantizer)
        else:
            self.ops = NativeOperations(self.store, device, row_chunk)
        self.linear_mode = linear_mode
        self.runtime_config_path=runtime_config
        self.context, self.device = context, device
        self.graph = graph = load_graph(project_root / 'reference/inference')
        if attention_mode == 'sm89':
            from .sm89_attention import sparse_attn
            graph.sparse_attn = sparse_attn
        self.attention_mode = attention_mode
        config = json.loads(runtime_config.read_text())
        config.update(max_batch_size=1, max_seq_len=context, temperature=0)
        args = graph.ModelArgs(**config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device('meta'):
                self.model = graph.Transformer(args, self.tokenizer)
        finally:
            torch.set_default_dtype(old)
        self._bind(args)
        self.runtime_corrections = {'index_cache_owner_publication_layers':publish_index_owners(graph,self.model)}
        self.runtime_corrections['linear_mode'] = linear_mode
        self.runtime_corrections['attention_mode'] = attention_mode
        self.runtime_corrections['weight_cache_bytes'] = weight_cache_bytes
        self.runtime_corrections['row_decode_mode'] = row_decode_mode
        self.runtime_corrections['activation_quantizer'] = activation_quantizer if linear_mode == 'native-cuda' else 'portable'
        self.chunked = None
        self.model.eval()

    def _bind(self, args):
        graph, model, ops = self.graph, self.model, self.ops
        streamed_parameters = set()
        for name, module in model.named_modules():
            weight_name = name + '.weight'
            if isinstance(module, (graph.ParallelEmbedding, graph.ParallelEngramEmbedding)):
                def embed_forward(this, indices, weight_name=weight_name):
                    values = ops.rows(weight_name, indices.detach().cpu().numpy().reshape(-1))
                    return values.reshape(*indices.shape, values.shape[-1])
                module.forward = types.MethodType(embed_forward, module)
                streamed_parameters.update(name + '.' + p for p in module._parameters)
            elif isinstance(module, (graph.Linear, torch.nn.Linear, graph.ParallelHead)):
                if name.endswith('wo_a'):
                    # Only grouped output projections access .weight directly. Keep
                    # one materialized tensor for its attention call, then release.
                    parent = model.get_submodule(name.rsplit('.', 1)[0])
                    def materialize(this, inputs, module=module, weight_name=weight_name):
                        n = ops.store.info(weight_name)['shape'][0]
                        module.weight = torch.nn.Parameter(ops.rows(weight_name, np.arange(n)), requires_grad=False)
                    def release(this, inputs, output, module=module):
                        module.weight = torch.nn.Parameter(torch.empty_like(module.weight, device='meta'), requires_grad=False)
                    parent.register_forward_pre_hook(materialize)
                    parent.register_forward_hook(release, always_call=True)
                else:
                    dtype = module.weight.dtype
                    bias_name = name + '.bias' if getattr(module, 'bias', None) is not None else None
                    is_head = isinstance(module, graph.ParallelHead)
                    def linear_forward(this, x, full_logits=False, weight_name=weight_name,
                                       dtype=dtype, bias_name=bias_name, is_head=is_head):
                        if is_head and not (full_logits or getattr(this, '_return_full_logits', False)):
                            x = x[:, -1]
                        if is_head:
                            x = x.float()
                        return ops.linear(weight_name, x, dtype, bias_name)
                    module.forward = types.MethodType(linear_forward, module)
                streamed_parameters.update(name + '.' + p for p in module._parameters)
        for name, parameter in list(model.named_parameters()):
            if name in streamed_parameters:
                continue
            parent, leaf = name.rsplit('.', 1) if '.' in name else ('', name)
            model.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(ops.small(name, parameter.dtype), requires_grad=False)
        # Rebuild the small hashing state on CPU; meta construction intentionally
        # retained no token-map values. It is the same pinned tokenizer algorithm.
        with torch.device('cpu'):
            model.engram_hash = graph.NgramHashState(args, model.engram_layout, self.tokenizer)
        model.engram_hash.to(self.device)
        graph.precompute_freqs_cis.cache_clear()
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(self.device):
                for module in model.modules():
                    if isinstance(module, graph.Attention):
                        ratio = module.compress_ratio
                        module.freqs_cis = graph.precompute_freqs_cis(args.rope_head_dim, args.max_seq_len,
                            args.original_seq_len if ratio else 0,
                            args.compress_rope_theta if ratio else args.rope_theta,
                            args.rope_factor, args.beta_fast, args.beta_slow)
                    for name, buffer in list(module._buffers.items()):
                        if buffer is not None and buffer.device.type == 'meta':
                            fill = -torch.inf if name == 'score_state' else 0
                            module._buffers[name] = torch.full(buffer.shape, fill, dtype=buffer.dtype, device=self.device)
        finally:
            torch.set_default_dtype(old)
        # Cached vision RoPE tables must follow the actual computation device.
        for module in model.modules():
            if hasattr(module, 'rope_dim') and hasattr(module, 'patch_embed'):
                original = module.forward
                def vision_forward(this, *inputs, original=original):
                    with torch.device(self.device):
                        return original(*inputs)
                module.forward = types.MethodType(vision_forward, module)

    @torch.inference_mode()
    def forward(self, tokens, start_pos=0, images=None, token_types=None, full_logits=False):
        if tokens.shape[0] != 1 or tokens.shape[1] == 0 or start_pos + tokens.shape[1] > self.context:
            raise ValueError('Reference input outside admitted context')
        if start_pos and tokens.shape[1] != 1 and self.chunked is None:
            raise ValueError('Start a chunked sequence before multi-token continuation')
        self.model.head._return_full_logits = full_logits
        try:
            with torch.device(self.device):
                if self.chunked is not None:
                    if images is not None:
                        raise ValueError('Supply images when starting the chunked sequence')
                    return self.chunked.forward(tokens.to(self.device),start_pos,
                        token_types=token_types.to(self.device) if token_types is not None else None,
                        full_logits=full_logits)
                return self.model(tokens.to(self.device), start_pos, images=images,
                                  token_types=token_types.to(self.device) if token_types is not None else None)
        finally:
            self.model.head._return_full_logits = False

    def start_chunked_sequence(self, prompt_length, images=None, *, query_tile=16,key_tile=256,head_tile=None,
                               tie_policy='native',score_budget_bytes=64*1024**2,index_mode='tiled',
                               candidate_scratch_budget_bytes=64*1024**2):
        from .chunked_reference import ChunkedReference
        score_function = _chunked_score_function(index_mode,self.device)
        self.stop_chunked_sequence()
        if head_tile is None:
            head_tile = 32 if self.device.type == 'cuda' else 1
        self.chunked = ChunkedReference(self.graph,self.model,query_tile=query_tile,key_tile=key_tile,head_tile=head_tile,
                                       tie_policy=tie_policy,score_budget_bytes=score_budget_bytes,
                                       score_function=score_function,candidate_scratch_budget_bytes=candidate_scratch_budget_bytes)
        self.runtime_corrections['chunked_index_tie_policy'] = tie_policy
        self.runtime_corrections['chunked_index_score_budget_bytes'] = score_budget_bytes
        self.runtime_corrections['chunked_index_head_tile'] = head_tile
        self.runtime_corrections['chunked_index_mode'] = index_mode
        with torch.device(self.device):
            self.chunked.begin(prompt_length,images=images)

    def stop_chunked_sequence(self):
        if self.chunked is not None:
            self.chunked.close()
            self.chunked = None

    @torch.inference_mode()
    def prefill_layerwise(self, tokens, *, images=None, token_types=None,
                          tile_sizes=(128,), full_logits=False, span=None,
                          host_budget_bytes=512*1024**2,index_mode='tiled'):
        """Diagnostic layer schedule; its chunked state supports subsequent decode."""
        from .chunked_reference import ChunkedReference
        from .layerwise_reference import LayerwisePrefill
        self.stop_chunked_sequence()
        head_tile = 32 if self.device.type == 'cuda' else 1
        self.chunked = ChunkedReference(self.graph,self.model,head_tile=head_tile,
                                       score_function=_chunked_score_function(index_mode,self.device))
        schedule = LayerwisePrefill(self.chunked,tile_sizes=tile_sizes,
                                    host_budget_bytes=host_budget_bytes,span=span)
        cache = getattr(getattr(self,'ops',None),'weight_cache',None)
        self.runtime_corrections['layerwise_prefill'] = {
            'tile_sizes':list(tile_sizes),'host_budget_bytes':host_budget_bytes,
            'mtp_enabled':False,'packed_weight_residency':bool(cache and cache.budget_bytes),'index_head_tile':head_tile}
        self.runtime_corrections['layerwise_prefill']['index_mode'] = index_mode
        with torch.device(self.device):
            result = schedule.run(tokens,images=images,token_types=token_types,
                                  full_logits=full_logits)
        self.runtime_corrections['layerwise_prefill']['workspace'] = schedule.last_plan
        return result

    def close(self):
        self.stop_chunked_sequence()
        if hasattr(self.ops,'weight_cache'):
            self.ops.weight_cache.clear()
        if getattr(self,'_owns_store',True):self.store.close()
