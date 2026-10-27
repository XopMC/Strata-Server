"""Real already-loaded Task489 idle counters; decisions are NOT GPU execution."""
import ctypes as C,os
from generation_off_binding_v506 import check,need
NAMES='schema words mode entered successful failed successful_rows positive_routes contains raw_groups raw_routes forced_groups forced_routes formed_groups formed_routes active'.split()
def read(phase):
 v=check();lib=C.CDLL(v['task']['path'],mode=os.RTLD_NOW|os.RTLD_NOLOAD)
 fn=lib.ds489_moe_gpu1_generation_info;fn.argtypes=[C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int
 out=(C.c_uint64*16)();rc=int(fn(out,16));return dict(phase=phase,native_rc=rc,words=list(map(int,out)),scope='actualTask489 cumulative idle decision counters; actualOwner/391 counters prove generationcompute separately')
def validate(r):
 w=r['words'];need(r['native_rc']==0 and len(w)==16 and w[:3]==[489,16,0],'actualTask489 mode0/idle getter')
 need(w[3]==w[4]and w[5]==w[15]==0,'Task489 successful only/error0/idle')
 need(w[9]==w[13]and w[10]==w[14]and w[11]==w[12]==0,'all real present groups/routes formedGPU commands; no forcedCPUallmiss')
 need(w[9]<=w[8]<=w[7]and w[10]<=w[7]and w[4]<=w[6]<=6*w[4]and w[7]<=6*w[6],'actual decision count bounds')
 return r
def delta(before,after,bulk_delta):
 validate(before);validate(after);a=before['words'];b=after['words'];d=[y-x for x,y in zip(a,b)]
 need(all(x>=0 for x in d[3:])and d[3]==d[4]==bulk_delta[6]>0 and d[5]==0,'same actual successfulCPUbulk generationjobs')
 need(d[8]>0 and d[9]==d[13]>0 and d[10]==d[14]>0 and d[11]==d[12]==0,'generation actual resident detection/formedGPU commands; no forcedCPUallmiss')
 return dict(words=d,fields=dict(zip(NAMES,d)),scope='true after-driverwarmup to postgenerate; hardware work separately checked through Owner/Operator0delta')
