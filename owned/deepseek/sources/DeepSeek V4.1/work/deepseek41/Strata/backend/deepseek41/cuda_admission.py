"""CUDA0 numerical helpers may coexist with explicitly admitted expert GPUs.

Tensor device checks remain CUDA0. This only verifies physical visibility before
allocations; it never enables peer access, NCCL or additional devices itself.
"""
import sys
from pathlib import Path


def cuda0_mask_admitted():
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / 'tools')) if str(root / 'tools') not in sys.path else None
    from stage_devices import validate_visible_mask
    try:
        validate_visible_mask()
    except (ValueError, OSError, KeyError):
        return False
    return True
