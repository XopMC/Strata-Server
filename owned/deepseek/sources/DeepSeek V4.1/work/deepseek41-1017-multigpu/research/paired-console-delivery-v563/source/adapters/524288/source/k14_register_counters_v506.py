"""Read ONCE per boundary: DSO-global atomics, never per-rank attribution."""
import ctypes as C
from k14_register_binding_v506 import need
NAMES=('version','latched_ON','ordinary_public_dispatches_after_S8_return','selected_public_calls','selected_public_rows','OFF_fallback_calls','geometry_fallback_calls','new_score_host_launches','new_score_grid_tiles','maximum_live_keys','shared_bytes','query_tile','key_tile','head_group','threads','raw_fixture_calls')
def read(lib):
 a=(C.c_uint64*16)();fn=lib.ds334_score_info;fn.argtypes=[C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int
 need(fn(a,16)==0,'actual16word getter');x=list(map(int,a));need(x[0:2]==[334,1]and x[10:15]==[49280,16,128,4,512],'actual ON/resources geometry')
 return dict(zip(NAMES,x))
def words(x):return [int(x[n])for n in NAMES]
def before(x):
 a=words(x);need(not any(a[2:10])and a[15]==0,'fresh DSO no prefill work before model')
def validate(a,b,*,prompt_tokens,shape,wavefront):
 need(prompt_tokens==523712 and shape==dict(passes=2,chunk_tokens=262144,attention_batch=1024),'exact328 geometry')
 x,y=words(a),words(b);need(all(y[i]>=x[i]for i in range(2,10)),'monotonic global counters')
 delta=[y[i]-x[i]for i in range(16)]
 # Three ordinary ratio2 index sources2/8/14; source20 with candidate blocks is
 # original fallback only before its original S8 threshold. S8/compact functions
 # return before this dispatch counter. Source20 prefix batches0..30 have
 # end<32*T; prefix batch31 onward and ALL suffix batches take originalS8.
 need((delta[2],delta[3],delta[4],delta[5],delta[6],delta[7],delta[15])==(1567,1536,1571136,0,31,6138,0),'actual full ordinary source census/selected fastpath, no raw fixture')
 need(y[9]==261856 and delta[8]>0,'actual causal live-key extent/grid work')
 after=wavefront['after_prefill'];need(after['GPU0_layers']==after['GPU1_layers']==40,'actual both rank layers; not fabricated per-device334 counters')
 return dict(scope='ONE DSO-global16 atomic array shared by both concurrent rank threads; read once, no2x sum',before_prefill=a,after_prefill=b,delta16=delta,actual_rank_layers={'0':after['GPU0_layers'],'1':after['GPU1_layers']},direct_per_rank_selected_counter='NOT_EXPOSED',source_inference='Both exact same-DSO rank actors use immutable334ON ordinary ratio2 dispatch at sources2/8/14; both-device component qualified this archive',denominator='ordinary dispatch counter excludes early ratio1 S8 and compact functions; 31early-prefix geometry fallbacks at source20 are intentional; selected1536/ordinary1567 is not all-indexer or wall fraction',model_parity_pending=True)
def unchanged(a,b):need(words(a)==words(b),'generation/close changed prefill-only334 aggregate')
