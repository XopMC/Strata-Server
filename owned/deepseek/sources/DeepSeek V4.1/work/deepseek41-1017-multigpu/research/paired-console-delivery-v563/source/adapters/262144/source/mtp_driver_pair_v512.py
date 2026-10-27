"""Use the measured GPU0 DSpark graphs in the ordinary API runtime."""
from contextlib import contextmanager
import os,time
import torch
from proposal_gap_v512 import ReferenceDSpark
from resident_source_stream_pair_v512 import resident_draft_moe
from resident_dense_stream_pair_v512 import resident_draft_dense
from native_cpp_speculative_cost_v44 import NativeGreedySpeculator
from factory_cache_pair_v512 import native_launch_cache


def initial_costs(context):
    if context == 131072:
        from mtp_cost128_pair_v512 import initial_costs as original
    elif context == 262144:
        from mtp_cost256_pair_v512 import initial_costs as original
    elif context == 524288:
        import json,hashlib
        from pathlib import Path
        raw=(Path(__file__).resolve().parent/'cost_prior_512_v40.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!='e98d988aa26970dff8ad46613e1a97cb7e1d5343c917be1cfdf19e6364b698e2':raise ValueError('Private512 cost estimate changed')
        prior=json.loads(raw)
        return dict(verify_seconds_by_inputs=prior['verify_seconds_by_inputs'],commit_seconds=prior['commit_seconds'])
    else:
        raise ValueError('Only the three explicit finalist cost priors are admitted')
    return original(context)


def prepare_mtp_context(reference, context):
    from mtp_binding_1017 import ATTRIBUTE, ModelBinding
    binding = getattr(reference, ATTRIBUTE, None)
    if not isinstance(binding, ModelBinding):
        raise ValueError('Attach the explicit new1017 owner binding before preparing proposal RoPE')
    binding.check_store(reference.store, require_ram=False)
    from mtp_setup_pair_v512 import prepare_mtp_context as original
    return original(reference, context)


@contextmanager
def greedy_driver(reference,engine,*,mode,context,pending,arena_upload=False,gap_control386=None,replay_proof386=False):
    if mode!='native4' or context not in (131072,262144,524288):
        raise ValueError('Explicit GPU0 draft mode and context up to512k required')
    prepare_mtp_context(reference,context)
    from mtp_binding_1017 import binding_for_store
    binding_for_store(reference.store)
    # Reserve the Torch cuBLAS workspace after target prefill, before draft weights,
    # as in the filled-context benchmark; preserves the prefill memory geometry.
    torch.cuda.current_blas_handle()
    if arena_upload is not False:raise ValueError('NEW1017 binds protected native4 only; no old own3 bank')
    moe_context=resident_draft_moe
    with moe_context(reference,bits=0 if mode=='native4' else 3)as weights, \
         resident_draft_dense(reference,engine)as dense,native_launch_cache(reference):
        driver=NativeGreedySpeculator(engine,ReferenceDSpark(reference,gap_control386=gap_control386,replay_proof386=replay_proof386),
            eos_token_id=reference.tokenizer.eos_token_id,context=context,
            max_draft_tokens=5,cost_profile=initial_costs(context))
        try:
            started=time.perf_counter();driver.seed()
            driver.drafter.prepare_commits(driver.last_main,driver.position-1)
            driver.seed();driver.drafter.propose(pending,driver.last_main,driver.position-1)
            driver.seed();torch.cuda.synchronize()
            metadata=dict(mode=mode,draft_load=weights,draft_dense=dense,
                warmup_seconds=time.perf_counter()-started,cost_prior='unchanged numerical v128/v136/v198 scheduling estimates;512 constants privately pinned; warm EMA; NEW1017 unmeasured',
                host_call_count_excludes_graph_replays=True,
                model_binding=binding_for_store(reference.store).configuration(),
                previous_recipe_cost_prior_reused=True, new1017_cost_prior_measured=False)
            if gap_control386 is not None:
                if driver.drafter.cuda_graph is None:raise RuntimeError('386 must warm proposal graph before arm')
                metadata['adaptive386_before_generation']=gap_control386.arm()
            yield driver,metadata
            if gap_control386 is not None:
                metadata['adaptive386_after_generation']=gap_control386.snapshot()
                metadata['adaptive386_gap_calls']=gap_control386.record()
            metadata.update(statistics=driver.statistics,graphs=driver.drafter.metadata)
        finally:
            # Captured graphs retain these weights and target borrowed tensors.
            # Destroy graphs before the two enclosing residency contexts exit.
            driver.drafter.close()
