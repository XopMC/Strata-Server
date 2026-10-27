"""Private pair-only CUDA0 helper visibility admission."""
def cuda0_mask_admitted():
 from stage_devices_pair_v506 import validate_visible_pair
 try:validate_visible_pair()
 except (ValueError,RuntimeError,OSError,KeyError):return False
 return True
