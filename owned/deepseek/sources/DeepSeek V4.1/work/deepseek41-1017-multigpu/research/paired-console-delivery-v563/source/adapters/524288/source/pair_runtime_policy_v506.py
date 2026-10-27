"""Private dual-RTX4090 runtime policy; CPU/source-only import."""
from __future__ import annotations

from index_selection_v506 import current_pair
CURRENT_PAIR = current_pair()
PAIR_UUIDS = tuple(r["uuid"] for r in CURRENT_PAIR)
PAIR_MASK = ",".join(PAIR_UUIDS)
PAIR_FLAG = "DS41_NATIVE_PAIR_MASK"
GPU0 = PAIR_UUIDS[0]
GPU1 = PAIR_UUIDS[1]
CONTEXTS = (262144, 524288)
CACHE_GIB = (0,)


def cache_devices(settings):
    mode = getattr(settings, "pair_cache_mode", "off")
    amount = getattr(settings, "expert_cache_gib", 0)
    mtp = getattr(settings, "mtp", "off")
    context = getattr(settings, "context", None)
    if type(context) is not int or context not in CONTEXTS or type(amount) is not int or amount not in CACHE_GIB:
        raise ValueError("K14 pair adapter admits only R7 256/512k contexts and cache zero")
    if mtp not in ("off", "native4"):
        raise ValueError("Pair adapter retains only the selected native4 MTP or MTP-off modes")
    if mode != "off":
        raise ValueError("K14 pair adapter requires cache-off mode")
    return (1,)  # retain selected zero-budget native cache configuration


def cache_settings_admitted(settings):
    try:
        cache_devices(settings)
        return True
    except (TypeError, ValueError):
        return False


def environment_for(settings):
    devices = cache_devices(settings)
    # NativeGpuExperts takes a per-visible-device cap. With native4 MTP, device0
    # is never an expert-cache arena; only the current physical GPU1 may be used.
    if getattr(settings, "mtp", "off") == "native4" and 0 in devices:
        raise ValueError("GPU0 MTP requires GPU0 expert-cache budget to remain zero")
    return {
        "CUDA_VISIBLE_DEVICES": PAIR_MASK,
        "DS41_EXPECTED_PAIR_UUIDS": PAIR_MASK,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "DEEPSEEK41_GPU_STAGE": "0",  # preserve the frozen Engine's GPU0 target path; never stage2
        PAIR_FLAG: "1",
        "DS41_NATIVE_EXPERT_DEVICES": ",".join(map(str, devices)),
        "DS41_NATIVE_PAIR_CACHE_MODE": getattr(settings, "pair_cache_mode", "off"),
        "DS41_NATIVE_PAIR_CACHE_GIB": str(getattr(settings, "expert_cache_gib", 0)),
        "DS41_NATIVE_PAIR_MTP": getattr(settings, "mtp", "off"),
    }


def runtime_record(settings):
    devices = cache_devices(settings)
    return {
        "phase": "PRIVATE_PAIR_CONFIGURATION_QUALITY_UNACCEPTED",
        "cuda_visible_devices": PAIR_MASK,
        "logical_devices": {"0": {"uuid": GPU0, "pci_bus_id": CURRENT_PAIR[0]["pci_bus_id"], "system_index": 0, "role": "target_native4_mtp_and_original_prefill_pass0_local_K14"},
                            "1": {"uuid": GPU1, "pci_bus_id": CURRENT_PAIR[1]["pci_bus_id"], "system_index": 1, "role": "original_prefill_pass1_headless_local_K14_then_GPU1_whole370"}},
        "pair_prefill_kind": "original_two_pass_layer_wavefront_v325_local_K14",
        "wavefront_actual_per_rank_and_state_join_evidence_required": True,
        "context": settings.context,
        "mtp": settings.mtp,
        "expert_cache_mode": getattr(settings, "pair_cache_mode", "off"),
        "expert_cache_devices": list(devices),
        "expert_cache_gib_per_selected_device": settings.expert_cache_gib,
        "gpu0_cache_bytes": 0,
        "cache_arithmetic": "generic GPU expert cache disabled; exact whole37020 replaces partialGU147",
        "expert_cache_assignment": "one (layer, expert) entry belongs to one arena; no duplicated expert entry",
        "device_telemetry_required": True,
    }
