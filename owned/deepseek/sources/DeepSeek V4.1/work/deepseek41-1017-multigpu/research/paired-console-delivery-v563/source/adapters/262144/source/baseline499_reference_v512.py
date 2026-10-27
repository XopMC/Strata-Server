"""Current completed499 raw is numerical reference only; no borrowed old DSO proof."""
from pathlib import Path
import json
from port_binding_v512 import check,need
SIX=('generated_token_ids','prefill_logits_sha256','encoded_prompt_sha256','encoded_prompt_token_ids_sha256','checkpoint','actual_prefill_shape')
def validate(actual,*,complete):
 v=check();ref=json.loads(Path(v['baseline499_raw']['path']).read_text());keys=SIX if complete else SIX[1:]
 for k in keys:need(actual.get(k)==ref.get(k),'new512 differs from completed499 '+k)
 return dict(actual499_raw=v['baseline499_raw'],fields=list(keys),all_fields_exact=True,numerical_reference_only=True,new511_native_model_qualification=False,quality_admitted=False)
