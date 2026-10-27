"""Actual33116word per-rank observer, not a requested-env execution claim."""
import ctypes as C
NAMES=('version','ON_constructor_flag_latches','GPU0_pool_starts','GPU1_pool_starts','GPU0_copy_jobs','GPU1_copy_jobs','GPU0_parallel_copy_jobs','GPU1_parallel_copy_jobs','GPU0_copy_bytes','GPU1_copy_bytes','GPU0_copy_errors','GPU1_copy_errors','pool_acquisition_or_identity_failures','GPU0_active_copies','GPU1_active_copies','reserved15')
def need(v,m):
 if not v:raise RuntimeError('333 actual331 counter proof: '+m)
def words(v):
 need(isinstance(v,dict)and set(v)==set(NAMES),'literal16fields');need(all(type(v[k])is int and 0<=v[k]<1<<64 for k in NAMES),'uint64words');return[v[k]for k in NAMES]
def read(lib):
 f=lib.ds331_rank_stage_copy_info;f.argtypes=[C.POINTER(C.c_uint64),C.c_int];f.restype=C.c_int
 out=(C.c_uint64*16)();need(f(out,16)==0,'actualgetterfailed');return dict(zip(NAMES,map(int,out)))
def validate_before(v):need(words(v)==[331,1]+[0]*14,'one actualheadful flaglatch, no previouscopy/pool beforeprefill')
def validate_after(before,after):
 validate_before(before);a=words(after)
 need(a[:4]==[331,2,1,1],'actualheadful+headless flaglatches andoneprivatepool/rank')
 need(a[4]>0 and a[5]>0 and a[4]==a[6]and a[5]==a[7],'bothranks actualprivateparalleljobs; no serialfallback')
 need(a[8]==a[4]*6635520 and a[9]==a[5]*6635520,'literalpaired139projectionbytes/rank')
 need(all(x==0 for x in a[10:]),'copy errors/acquisitionfailure/active/reserved mustzero')
 return dict(before_prefill=before,after_prefill=after,scope='actual331 ON-only privatepool constructor/completedjobs/bytes; noCUDAkernel or speed claim',performance_eligible=False)
def validate_projection_totals(after,info):
 a=words(after);need(a[4]+a[5]==info['prefill_compact_projection_uploads']==info['prefill_compact_FP8_encodes'],'actualboth-rank copiedjobs ==325merged252uploads/encodes')
 need(a[8]+a[9]==info['prefill_compact_H2D_bytes'],'actualboth-rank copiedbytes ==325merged252H2D')
 need(info['paired145_upload_delta'][2]==2 and info['paired145_upload_delta'][3]==a[4]+a[5]and info['paired145_upload_delta'][4]==a[8]+a[9],'actualshared145 twoconsumers/jobs/bytes')
def unchanged(before,after):need(words(before)==words(after),'generation/close changed completedprefillcopy counters')
