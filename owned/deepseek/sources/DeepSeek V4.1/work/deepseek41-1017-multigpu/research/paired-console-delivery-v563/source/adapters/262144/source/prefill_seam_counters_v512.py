"""Read actual317 free C ABI; pure validators do not initialize native runtime."""
import ctypes as C
NAMES=('version','begun','layers','finished','aborted','live_states','last_c0','last_S','last_p0','last_suffix','last_next_layer','recorded_layer_events')
def need(value,message):
    if not value:raise RuntimeError('321 serial-prefill proof: '+message)
def read(library):
    fn=library.ds317_prefill_seam_info
    fn.argtypes=[C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int
    values=(C.c_uint64*12)();rc=fn(values,12)
    need(rc==0,'actual ds317 getter rc='+str(rc))
    return dict(zip(NAMES,map(int,values)))
def words(value):
    need(isinstance(value,dict) and set(value)==set(NAMES),'literal12 named fields')
    need(all(type(value[k]) is int and value[k]>=0 for k in NAMES),'unsigned integer counters')
    return [value[k] for k in NAMES]
def validate_before(before):
    need(words(before)==[317]+[0]*11,'fresh child must have no prior317 passes/events')
def validate_after(before,after,*,prompt_tokens,context,chunk,batch,expert_batch,shape):
    validate_before(before)
    need((context,chunk,batch,expert_batch)==(262144,131072,2048,4096),'original512 B1024/E2048/cap262144')
    need(type(prompt_tokens)is int and prompt_tokens==261568,'two nonempty original passes')
    need(shape==dict(passes=2,chunk_tokens=131072,attention_batch=2048),'actual original full512 shape')
    last=prompt_tokens-131072
    suffix=((last-4096)//2048)*2048 if last>8192 else 0
    expected=[317,2,80,2,0,0,131072,last,131072,suffix,40,80]
    need(words(after)==expected,'actual317 pass/layer/event/lifetime geometry differs: '+str(words(after)))
    return dict(before_prefill=dict(before),after_prefill=dict(after),expected_after_prefill=dict(zip(NAMES,expected)),
        actual_context=context,prompt_tokens=prompt_tokens,actual_prefill_shape=dict(shape),
        scope='Actual335 original256 pass arithmetic on BOTH ranks; real335/325 transport and actualserial352 parity required separately',
        model_parity_pending=True,performance_eligible=False)
def unchanged(before,after):
    need(words(before)==words(after),'generation/close changed prefill-only counters')
