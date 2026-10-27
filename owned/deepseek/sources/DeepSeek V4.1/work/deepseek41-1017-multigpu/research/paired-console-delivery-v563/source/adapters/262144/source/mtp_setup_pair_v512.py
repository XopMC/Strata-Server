import torch

def prepare_mtp_context(reference,context):
    """Prepare immutable proposal RoPE before target prefill, including serial runs.

    The filled-context benchmark allocates these tables for both target-only
    and MTP configurations. Allocating them later changes the target chunk
    geometry when the full prompt exceeds the available GPU scratch.
    """
    from native_mtp_capture_stream_v92 import capture_stream
    capture_stream(reference)
    if context not in(4096,16384,131072,262144,524288):raise ValueError('Explicit context up to512k required')
    if getattr(reference,'native_mtp_prepared_context',None)==context:return
    reference.model.max_seq_len=context
    import json
    settings=json.loads(reference.runtime_config_path.read_text())
    with torch.device(reference.device):
        frequencies=reference.graph.precompute_freqs_cis(settings['rope_head_dim'],context,0,
            settings['rope_theta'],settings['rope_factor'],settings['beta_fast'],settings['beta_slow'])
    for layer in reference.model.mtp:layer.attn.freqs_cis=frequencies
    reference.native_mtp_prepared_context=context

