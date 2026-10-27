"""Actual335256 wavefront+325 transport; no old326 resource shortcut."""
import ctypes as C
from prefill_seam_binding_v512 import need
def read(library):
 out={}
 for name,symbol,count in(('geometry335','ds335_prefill_geometry_info',32),('wavefront325','ds325_prefill_wavefront_info',48)):
  fn=getattr(library,symbol);fn.argtypes=[C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int
  values=(C.c_uint64*count)();need(fn(values,count)==0,'actual '+symbol+' failed');out[name]=list(map(int,values))
 return out
def validate_before(v):need(v==dict(geometry335=[335]+[0]*31,wavefront325=[325]+[0]*47),'fresh335/325 counters before wavefrontPF')
def validate_after(before,after):
 validate_before(before);a=after['geometry335'];w=after['wavefront325'];need(len(a)==32 and len(w)==48,'actual335/325 words')
 expected={0:335,1:1,2:2,3:262144,4:261568,5:131072,6:131072,7:2048,8:4096,9:1,10:1,11:0,12:0,13:2,14:80,15:2,16:0,17:0,18:40,19:40,28:0,29:0,30:0,31:0}
 need(all(a[k]==v for k,v in expected.items()),'actual335 wavefront256 mode/geometry/real317/error census')
 need(min(a[20:24])>0 and min(a[26:28])>=512<<20,'actual BOTH requestedscratch/features and sampledfree')
 need(0<a[24]<a[25]<=20<<30 and a[21]+a[23]+a[24]<=20<<30,'actual256headless startup/persistent/scratch/feature budget; no old512resource proof')
 ex={0:325,1:1,2:1,3:1,4:0,5:0,6:40,7:40,8:40,9:40,10:40,11:40,12:424673280,13:422830080,14:849346560,15:845660160,16:670138368,17:1,18:1,19:2,20:3932160,21:10240,22:1,23:1,24:261568,25:4,26:4,27:0,28:0,29:0,30:0,31:0,34:1,35:20<<30,38:1,39:0,41:0,42:0,43:126,44:252,45:0,46:0,47:0}
 need(all(w[k]==v for k,v in ex.items()),'actual325 all40 transport/ACK/fulljoin/geometry/zeroerrors')
 need(min(w[32:34])>=512<<20 and 0<w[37]<w[36]==a[24] and w[40]>0,'actual headless256 ledger and both reserve samples')
 return dict(before_prefill=before,after_prefill=after,scope='Actual335256 two-GPU role/allocation/carry/ACK/join census. Samples are not peak guarantee; actual serial352 logits/all512IDs enforced separately.',old326_used_as_256_proof=False,model_parity_pending=True,performance_eligible=False)
def unchanged(before,after):need(before==after,'generation/close changed wavefront-prefill335/325 counters')
def helper_off(before,after,policy,settings):
 policy.assert_current(settings);need(dict(policy.environment).get('DS1017_PAIR_INDEXER_PREFILL_V1')=='0','oldhelper immutableOFF')
 need(tuple(before)==tuple(after)==(1,)+(0,)*19,'old helper wrapper/device/work muststayzero')
 return dict(policy='OFF; localK14 BOTH ranks during wavefront',before=list(before),after=list(after),old_helper_used=False)
