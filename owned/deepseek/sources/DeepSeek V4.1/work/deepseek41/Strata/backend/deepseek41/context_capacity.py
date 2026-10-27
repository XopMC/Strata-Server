"""Explicit experimental capacity admission; full-context quality is separate."""
from contextlib import contextmanager
import torch

GIB = 1024**3
MAX_CONTEXT = 1048576


def context_plan(config,capacity,prompt_tokens,*,weight_cache_bytes=0,tile_tokens=128,kv_cache_type='bf16'):
    if (type(capacity) is not int or not 2 <= capacity <= MAX_CONTEXT
            or type(prompt_tokens) is not int or not 1 <= prompt_tokens < capacity
            or type(tile_tokens) is not int or not 1 <= tile_tokens <= 4096
            or type(weight_cache_bytes) is not int or weight_cache_bytes < 0 or kv_cache_type not in('bf16','q4_0','native-fp4')):
        raise ValueError('Invalid experimental context/tile/cache capacity')
    ratios = config['compress_ratios']
    positions = sum(capacity//ratios[i] for i in config['kv_source_layers'])
    sparse_width = min(config['index_topk'],prompt_tokens)
    candidate_width = min(config['candidate_topk_blocks'],
                          (prompt_tokens+config['candidate_block_size']-1)//config['candidate_block_size'])
    rows = prompt_tokens
    host = {'hc_double_buffer':2*rows*config['hc_mult']*config['dim']*2,
            'hc_pre_mix':rows*config['hc_mult']*4,
            'engram_hash_ids':rows*len(config['engram_layer_ids'])*(config['engram_max_ngram_size']-1)*config['engram_n_heads']*8,
            'sparse_ids':rows*sparse_width*4,'candidate_block_ids':rows*candidate_width*4}
    # Two distinct RoPE policies are shared by all attentions; never40 copies.
    device = {'compressed_kv_bf16':positions*config['head_dim']*2,
              'index_keys_bf16':positions*config['index_head_dim']*2,
              'rotary_tables':2*capacity*(config['rope_head_dim']//2)*8,
              'engram_token_history':capacity*8,
              'window_kv':(config['n_layers']+config['n_mtp_layers'])*config['window_size']*config['head_dim']*2,
              'expert_dense_weight_cache':weight_cache_bytes,
              'bounded_index_score_and_gather':128*1024**2,
              'tile_hc_double_buffer':2*tile_tokens*config['hc_mult']*config['dim']*2}
    if kv_cache_type in('q4_0','native-fp4'):
        if config['head_dim']%32:raise ValueError('Q4_0 latent width must align to32')
        del device['compressed_kv_bf16']
        device['compressed_kv_'+kv_cache_type.replace('-','_')]=positions*(config['head_dim']//32)*18
    return {'context_capacity':capacity,'actual_prompt_tokens':prompt_tokens,'host':host,'device':device,
            'host_workspace_bytes':sum(host.values()),'device_known_bytes':sum(device.values()),
            'device_additional_scratch_reserve_bytes':2*GIB,
            'validated_full_context':False,'kv_cache_type':kv_cache_type,
            'method':{'bf16':'BF16 native cache','q4_0':'GGML Q4_0 compressed latent cache; native BF16 index/window/tails',
                'native-fp4':'Native E2M1/E4M3 group16 compressed latent cache; native BF16 index/window/tails'}[kv_cache_type]+', layerwise host HC, bounded score/gather tiles; no full-prompt logits.'}


def cache_budget(cache,budget):
    """Change capacity only between serial calls, never evict live tensor users."""
    if type(budget) is not int or budget < 0:
        raise ValueError('Invalid packed cache budget')
    with cache.lock:
        if any(entry.leases for entry in cache.entries.values()):
            raise ValueError('Cannot resize a cache containing live leases')
        while cache.bytes > budget:
            _,entry = cache.entries.popitem(last=False)
            cache.bytes -= entry.bytes
            cache.evictions += 1
            del entry
        cache.budget_bytes = budget


@contextmanager
def experimental_capacity(reference,config,capacity,*,weight_cache_bytes=None,kv_cache_type='bf16'):
    """Replace only state buffers; retain verified packed weights and graph.

    Callers must run layerwise/chunked scheduling, and must not interpret this
    allocation as a successful long-context benchmark. Source state restored
    on failure; no shared buffers remain enlarged after a diagnostic.
    """
    if type(capacity) is not int or not 2 <= capacity <= MAX_CONTEXT:
        raise ValueError('Experimental capacity outside2–1048576')
    if reference.chunked is not None:
        raise ValueError('Close the previous sequence before capacity replacement')
    old_capacity = reference.context
    cache = getattr(reference.ops,'weight_cache',None)
    old_budget = cache.budget_bytes if cache is not None else 0
    if weight_cache_bytes is None:
        weight_cache_bytes = old_budget
    plan = context_plan(config,capacity,max(1,capacity-1),weight_cache_bytes=weight_cache_bytes,kv_cache_type=kv_cache_type)
    if kv_cache_type=='q4_0'and reference.device.type=='cuda':
        from .q4_cache import admit_component
        admit_component(reference)
    if kv_cache_type=='native-fp4'and reference.device.type=='cuda':
        from .native_fp4_cache import admit_component
        admit_component(reference)
    if reference.device.type == 'cuda':
        import gc
        report = getattr(reference.store,'ram_preload_report',None)
        if not report or report.get('source_revision') != '23da584e06e57ec43c896d286b44b822c9bb0ad6':
            raise ValueError('Large-context CUDA requires the independently verified live packed RAM owner')
        from .weights import available_memory_bytes
        if available_memory_bytes() < plan['host_workspace_bytes']+80*GIB:
            raise MemoryError('Full layerwise workspace would violate80GiB free RAM reserve')
        if cache is not None:
            # An old fragmented16GiB cache can otherwise pass an incremental
            # check and later OOM when the new context fills its entire budget.
            # Retire ONLY our unleased weight cache, then admit the whole new
            # budget together with new state and scratch. No other owner is
            # queried or modified.
            cache_budget(cache,0)
        gc.collect()
        torch.cuda.synchronize(reference.device)
        torch.cuda.empty_cache()
        free,_ = torch.cuda.mem_get_info(reference.device)
        needed = plan['device_known_bytes']
        plan['allocation_admission']={'driver_free_after_own_cache_release':free,
            'requested_complete_state_and_weight_cache_bytes':needed,
            'torch_allocated_bytes':torch.cuda.memory_allocated(reference.device),
            'torch_reserved_bytes':torch.cuda.memory_reserved(reference.device),
            'other_owners_touched':False}
        if free < needed+plan['device_additional_scratch_reserve_bytes']:
            if cache is not None:
                cache_budget(cache,old_budget)
            raise MemoryError('Native long-context state would violate2GiB GPU scratch reserve')
        if cache is not None:cache_budget(cache,weight_cache_bytes)
    previous,old_index_freqs,temporary_attributes = [],[],[]
    old_flag = getattr(reference,'experimental_long_context',False)
    graph = reference.graph
    old_shared = vars(graph.shared_attn).copy() if hasattr(graph,'shared_attn') else None
    graph.precompute_freqs_cis.cache_clear()
    def replace(module,name,value):
        previous.append((module,name,getattr(module,name)))
        setattr(module,name,value)
    def attribute(module,name,value):
        exists=hasattr(module,name)
        temporary_attributes.append((module,name,exists,getattr(module,name,None)))
        setattr(module,name,value)
    try:
        with torch.device(reference.device):
            for layer in [*reference.model.layers,*reference.model.mtp]:
                attention = layer.attn
                ratio = attention.compress_ratio
                frequencies = graph.precompute_freqs_cis(config['rope_head_dim'],capacity,
                    config['original_seq_len'] if ratio else 0,
                    config['compress_rope_theta'] if ratio else config['rope_theta'],
                    config['rope_factor'],config['beta_fast'],config['beta_slow'])
                replace(attention,'freqs_cis',frequencies)
                if attention.is_kv_source:
                    if kv_cache_type in('q4_0','native-fp4'):
                        if kv_cache_type=='q4_0':
                            from .q4_cache import Q4Cache as Storage
                            name='_deepseek41_q4_kv'
                        else:
                            from .native_fp4_cache import NativeFP4Cache as Storage
                            name='_deepseek41_native_fp4_kv'
                        attribute(attention,name,Storage(capacity//ratio,config['head_dim'],reference.device))
                        replace(attention,'compress_kv_cache',torch.empty(1,0,config['head_dim'],dtype=torch.bfloat16,device=reference.device))
                    else:
                        replace(attention,'compress_kv_cache',torch.zeros(1,capacity//ratio,config['head_dim'],
                            dtype=attention.compress_kv_cache.dtype,device=reference.device))
                if attention.indexer is not None:
                    indexer = attention.indexer
                    old_index_freqs.append((indexer,indexer.freqs_cis))
                    indexer.freqs_cis = None
                    if indexer.owns_k:
                        replace(indexer,'k_cache',torch.zeros(1,capacity//ratio,config['index_head_dim'],
                            dtype=indexer.k_cache.dtype,device=reference.device))
            if reference.model.engram_hash is not None:
                replace(reference.model.engram_hash,'cache',torch.empty(1,capacity,dtype=torch.int64,device=reference.device))
        reference.context,reference.model.max_seq_len = capacity,capacity
        attribute(reference.model,'_deepseek41_kv_cache_type',kv_cache_type)
        reference.experimental_long_context = True
        yield plan
    finally:
        reference.stop_chunked_sequence()
        for module,name,value in reversed(previous):
            setattr(module,name,value)
        for module,value in old_index_freqs:
            module.freqs_cis = value
        for module,name,exists,value in reversed(temporary_attributes):
            if exists:setattr(module,name,value)
            else:delattr(module,name)
        reference.context,reference.model.max_seq_len = old_capacity,old_capacity
        reference.experimental_long_context = old_flag
        graph.precompute_freqs_cis.cache_clear()
        if old_shared is not None:
            vars(graph.shared_attn).clear()
            vars(graph.shared_attn).update(old_shared)
        if cache is not None:
            cache_budget(cache,old_budget)
