"""Actual33724 words: separate attention publication and full-layer completion."""
import ctypes as C
NAMES=('version','ON_constructor_latches','GPU0_attention_events','GPU1_attention_events','GPU0_early_publications','GPU1_early_publications','GPU0_full_layer_syncs','GPU1_full_layer_syncs','GPU0_FFN_after_ACK','GPU1_FFN_after_ACK','GPU0_callback_ACKs','GPU1_callback_ACKs','refusals','active_callbacks','borrowed_early_events','drained_borrow_aborts','live_early_events','source_publications','nonsource_publications','reserved19','GPU0_last_attention_layer_plus1','GPU1_last_attention_layer_plus1','GPU0_last_full_layer_plus1','GPU1_last_full_layer_plus1')
def need(v,m):
 if not v:raise RuntimeError('342 actual337 counter proof: '+m)
def words(v):
 need(isinstance(v,dict)and set(v)==set(NAMES),'literal24fields');need(all(type(v[k])is int and 0<=v[k]<1<<64 for k in NAMES),'uint64words');return[v[k]for k in NAMES]
def read(lib):
 f=lib.ds337_early_attention_info;f.argtypes=[C.POINTER(C.c_uint64),C.c_int];f.restype=C.c_int
 out=(C.c_uint64*24)();need(f(out,24)==0,'actual getter failed');return dict(zip(NAMES,map(int,out)))
def validate_before(v):need(words(v)==[337,1]+[0]*22,'one headful constructor latch, no prior pass')
def validate_after(before,after,wave):
 validate_before(before);need(words(after)==[337,2]+[40]*10+[0]*5+[8,72,0]+[40]*4,'both early boundaries plus all full layers/ACKs/drained lifetimes')
 a=wave['after_prefill'];need(a['GPU0_layers']==a['GPU1_layers']==40,'actual original325 full-layer census')
 return dict(before_prefill=before,after_prefill=after,scope='actual337 early attention callbacks/ACKs plus separate full FFN completion; original32548 and31712 retained; no overlap-duration or speed inference',performance_eligible=False)
def unchanged(before,after):need(words(before)==words(after),'generation/close changed completed337 prefill counters')
