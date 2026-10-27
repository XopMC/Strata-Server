"""Import the pinned graph with portable kernels instead of importing TileLang.

The caller chooses meta/CPU explicitly. Loading this module never selects a GPU.
"""
import importlib.util
from pathlib import Path
import sys
import types

from . import portable_kernels


def publish_index_owners(graph, model):
    """Correct the pinned graph's stale shared-index pointer during partial groups.

    Original Indexer.forward publishes its own key cache only when a new latent
    arrives. On intervening decode steps the global still points to a later KV
    source from the preceding forward. Publish the owner every time; no weights
    or numerical operations change. Dense prefill is unaffected. The defect and
    correction are reproduced on a tiny graph, not asserted as checkpoint parity.
    """
    corrected = []
    for layer in model.layers:
        indexer = layer.attn.indexer
        if indexer is None or not indexer.owns_k:
            continue
        corrected.append(layer.attn.layer_id)
        if getattr(indexer, '_deepseek41_always_publish', False):
            continue
        original = indexer.forward
        def forward(this, *args, original=original, **kwargs):
            graph.shared_attn.index_k = this.k_cache
            return original(*args, **kwargs)
        indexer.forward = types.MethodType(forward,indexer)
        indexer._deepseek41_always_publish = True
    return corrected


def load_graph(reference_directory):
    reference_directory = Path(reference_directory).resolve()
    aliases = ['kernel', 'engram', 'vision', 'image_processor', 'model']
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules['kernel'] = portable_kernels
        for name in ['engram', 'vision', 'image_processor', 'model']:
            spec = importlib.util.spec_from_file_location(name, reference_directory / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules['model']
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
