"""Literal449/456 numerical objects, new469 matched host-only closure."""
from fixed_binding_v506 import check as fixed_check,call_prior,need,pin
FLAGS={'DS1017_VERIFY_WOA_QUANT_MANY_V456':'1','DS1017_VERIFY_WOA_MANY_V449':'1','DS41_NATIVE_WOA_MANY':'0'}
def check(require_ready=True):
 v=fixed_check();base=call_prior('verify_woa_binding')
 need(base['flag']==FLAGS,'literal qualified449/456 flags')
 return dict(base,contexts=v['contexts'],build=v['build'],source_pins=v['source_pins'],recipe=v['recipe'],historical456=base,selected_host_overlay=469)
