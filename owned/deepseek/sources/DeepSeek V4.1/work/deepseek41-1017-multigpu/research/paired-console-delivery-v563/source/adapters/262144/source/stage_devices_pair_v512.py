"""Strict physical pair admission, called before importing torch/CUDA."""
import os
import sys
import subprocess
from pair_runtime_policy_v512 import PAIR_MASK, PAIR_UUIDS, PAIR_FLAG


def validate_visible_pair():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PAIR_MASK:
        raise ValueError("Pair runtime requires the exact GPU0,GPU1 UUID mask in logical order")
    if os.environ.get("DS41_EXPECTED_PAIR_UUIDS") != PAIR_MASK:
        raise ValueError("Native pair snapshot must equal fresh index0/1 selection")
    if os.environ.get(PAIR_FLAG) != "1":
        raise ValueError("Pair runtime native admission flag is missing")
    if os.environ.get("DEEPSEEK41_GPU_STAGE") != "0":
        raise ValueError("Pair runtime keeps frozen Engine stage=0; old prefix stage2 is forbidden")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("Pair runtime requires PCI_BUS_ID ordering")
    return PAIR_UUIDS


def activate_pair():
    torch = sys.modules.get("torch")
    if torch is not None and (torch.cuda.is_initialized() or torch.cuda._is_in_bad_fork()):
        raise RuntimeError("Choose the pair mask before CUDA initialization")
    current = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if current not in ("", PAIR_MASK):
        raise ValueError("Refusing to replace another GPU mask; launch a fresh pair child")
    os.environ["CUDA_VISIBLE_DEVICES"] = PAIR_MASK
    os.environ["DS41_EXPECTED_PAIR_UUIDS"] = PAIR_MASK  # overwrite inherited token, never authorize from it
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["DEEPSEEK41_GPU_STAGE"] = "0"
    os.environ[PAIR_FLAG] = "1"
    return validate_visible_pair()


def require_gpu1_no_foreign_compute():
    """Bounded health guard for only the admitted GPU1; GPU0 is checked by frozen health."""
    result = subprocess.run(
        ["nvidia-smi", "-i", PAIR_UUIDS[1], "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15)
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split(",")]
        if len(fields) != 2 or fields[1] != PAIR_UUIDS[1] or int(fields[0]) != os.getpid():
            raise RuntimeError("GPU1 has another compute owner; pair child refuses cache allocation")
