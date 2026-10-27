"""Actual325 DSO48 counters; no DSO loading and no inferred execution claims."""
import ctypes as C,json
from pathlib import Path
from prefill_seam_binding_v506 import admission,FLAGS,need
NAMES=('version325','entered_enabled','requests_started','requests_completed','failure_reports','active_request','GPU0_layers','GPU1_layers','forward_published','forward_H2D_consumed','reverse_published','reverse_H2D_consumed','forward_payload_bytes','reverse_payload_bytes','forward_two_leg_DMA_bytes','reverse_two_leg_DMA_bytes','pinned_transport_frames_bytes','GPU0_issuer_starts','GPU1_issuer_starts','issuer_joins','MTP_tail_bytes','final_BF16_row_bytes','actual_frequency_merges','history_published','committed_end','forward_source_frames','reverse_source_frames','carry_or_boundary_refusals','GPU1_actual_CPUowner','GPU1_actual_head_pointer','GPU1_actual_pairindexer','GPU0_actual_pairindexer','GPU0_min_sampled_free_bytes','GPU1_min_sampled_free_bytes','headless_context_destroyed','headless_actual_max_device_budget','headless_actual_persistent_owned_plus_Pack','headless_actual_streamed_Pack','GPU0_dense_restored','drained_headless_aborts','prefill_wall_nanoseconds','reserved41','reserved42','actual_GPU0_issuer_CPU','actual_GPU1_issuer_CPU','reserved45','reserved46','reserved47')
def words(v):
 need(isinstance(v,dict)and set(v)==set(NAMES),'literal32548 fields')
 need(all(type(v[k])is int and 0<=v[k]<1<<64 for k in NAMES),'uint64 actual325 counters')
 return [v[k]for k in NAMES]
def read(library):
 fn=library.ds325_prefill_wavefront_info;fn.argtypes=[C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int
 out=(C.c_uint64*48)();need(fn(out,48)==0,'actual ds325 getter failure')
 return dict(zip(NAMES,map(int,out)))
def validate_before(before):need(words(before)==[325]+[0]*47,'fresh child no previous325 request/activity')
def validate_after(before,after,*,prompt_tokens,context,chunk,batch,expert_batch,shape):
 validate_before(before);a=words(after);v=admission()
 need((prompt_tokens,context,chunk,batch,expert_batch)==(523712,524288,262144,1024,2048),'exact original523712/two-pass B1024/E2048/cap262144')
 need(shape==dict(passes=2,chunk_tokens=262144,attention_batch=1024),'actual original full512 prefill shape')
 contract=json.loads(Path(v['counter_contract']['path']).read_text())
 need(tuple(contract['fields'])==NAMES,'actual source counter contract layout')
 for index,wanted in contract['expected_success'].items():need(a[int(index)]==wanted,'actual325 '+NAMES[int(index)]+' differs: '+str(a[int(index)]))
 need(a[0]==325 and min(a[32],a[33])>=512<<20,'actual both-rank reserve samples')
 resource=json.loads(Path(v['resource_proof']['path']).read_text())['native']
 need(0<a[37]<a[36]<=resource['planned_startup_device_bytes']<=a[35]==20<<30,'actual headless persistent/streamed ledger within resource qualified startup budget')
 need(a[43]==int(FLAGS['DS1017_WAVEFRONT_GPU0_CPU_V325'])and a[44]==int(FLAGS['DS1017_WAVEFRONT_GPU1_CPU_V325']),'actual issuer CPUs differ from immutable Root topology')
 need(a[40]>0 and all(a[i]==0 for i in(41,42,45,46,47)),'actual elapsed and reserved words')
 return dict(before_prefill=before,after_prefill=after,source_criteria=v['counter_contract'],resource_only_proof=v['resource_proof'],actual_context=context,prompt_tokens=prompt_tokens,actual_prefill_shape=shape,
  local_K14_both_ranks=True,old_pair_indexer_helper_used=False,state_transport_scope='actual payload bytes; two DMA legs counted separately; no routed activation relay',
  memory_scope='actual rank1 persistent+streamed Pack at pass begin; reserve samples on both devices, not graph peak',
  timing_scope='actual325 outer prefill excludes postprefill GU147 build; runtime NEWPP includes entire Engine.prefill call',
  model_parity_pending=True,performance_eligible=False)
def unchanged(before,after):need(words(before)==words(after),'generation/close changed prefill-only325 counters')
def validate_old_helper_off(before,after,policy,settings):
 policy.assert_current(settings);need(dict(policy.environment).get('DS1017_PAIR_INDEXER_PREFILL_V1')=='0','old helper flag must be immutableOFF')
 need(tuple(before)==tuple(after)==(1,)+(0,)*19,'old pair-helper owner/work/bytes must all remain zero')
 return dict(policy='OFF; local K14 inside both original passes',before=list(before),after=list(after),old_helper_used=False,wavefront_counters_required=True)
