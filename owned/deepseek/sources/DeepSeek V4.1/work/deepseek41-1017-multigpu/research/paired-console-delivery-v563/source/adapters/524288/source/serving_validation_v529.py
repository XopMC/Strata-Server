"""Request-local native528 geometry and preserved numerical-generation checks."""
import ctypes as C,importlib,json,os,re
from pathlib import Path

def need(value,message):
 if not value:raise RuntimeError('serving529: '+message)

class Request528:
 NAMES=('schema','prepare_epoch','armed','mode','context','N','position','split','cap','B','E','first_rows','second_rows','carry_rows','live_Impls','active_prefill')
 def __init__(self,path,settings,binding):
  need(binding['ready']is True and binding['context']==settings.context,'Root528 serving binding required')
  need(binding['prepare_symbol']=='ds528_prepare_request_v1' and binding['getter_symbol']=='ds528_prefill_plan_info','explicit native528 ABI')
  self.settings=settings;self.binding=binding
  self.library=C.CDLL(str(path),mode=os.RTLD_NOW|os.RTLD_LOCAL)
  self.prepared=self.prefill=None;self.transport_last=None
 def read(self):
  f=getattr(self.library,self.binding['getter_symbol']);f.argtypes=[C.POINTER(C.c_uint64),C.c_int];f.restype=C.c_int
  data=(C.c_uint64*16)();need(f(data,16)==0,'actual528 getter')
  row=dict(zip(self.NAMES,map(int,data)))
  need(row['schema']==528 and row['armed']==1 and row['prepare_epoch']>0 and row['mode']in(1,2,3,4),'armed request geometry')
  need(row['active_prefill']==0 and row['context']==self.settings.context,'idle prefill and allocated context')
  return row
 def prepare(self,rows,*,images=False,nll=False):
  need(self.prepared is None and type(rows)is int and 0<rows<=self.settings.context,'prepare once after actual encoding')
  control=self.binding.get('request_prepare_flags',0);need(type(control)is int and control in(0,4),'typed same-partition oracle flag')
  flags=control|int(images)|(int(nll)<<1)
  f=getattr(self.library,self.binding['prepare_symbol']);f.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint32];f.restype=C.c_int
  need(f(self.settings.context,rows,0,flags)==0,'idle-only528 prepare rejected')
  plan=self.read();need(plan['N']==rows and plan['position']==0 and plan['live_Impls']==0,'real rows bound before any Engine')
  self.validate_plan(plan)
  need(plan['mode']in(2,3,4),'mandatory fastpath: this request still requires an unimplemented accelerated variant; refusing silent ordinary prefill')
  self.prepared=plan
  self.transport_before=read_transport391(self.library);transport_snapshot391(self.transport_before,live=0)
  self.direct531_before=read531(self.library,enabled=self.binding['features']['direct_fp8'])
  self.direct531_info=None
  self.fast534_before=read534(self.library);self.fast534_info=None
  self.pass554_before=read554(self.library);self.pass554_info=None
  need(self.pass554_before==[554,24]+[0]*22,'558 legacy554 owner must be inactive')
  self.pass558_before=read558(self.library);self.pass558_info=None
  return dict(plan)
 def validate_plan(self,plan):
  s=self.settings;need(0<plan['cap']<=s.chunk_tokens and 0<plan['B']<=s.prefill_batch and 0<plan['E']<=s.expert_batch,'bounded actual prefill scratch independent of context')
  if plan['mode']in(2,3):
   a,b=plan['first_rows'],plan['second_rows']
   need(plan['split']==a and a%128==0 and 128<=a<=plan['cap'] and 128<=b<=plan['cap'] and a+b==plan['N'],'two real complete token spans')
   need(plan['cap']%128==0 and plan['B']==min(s.prefill_batch,plan['cap']) and plan['carry_rows']==plan['N']%2,'aligned capacity and truthful final carry')
  else:
   need(plan['split']==plan['first_rows']==plan['second_rows']==plan['carry_rows']==0,'one-span path has no invented two-rank spans')
   if plan['mode']==4:need(self.binding['features']['direct_fp8']is True and plan['E']>=16 and plan['E']%16==0,'accelerated one-span requires direct531 and bounded aligned FP8 workspace')
  if plan['mode']in(2,3):need(plan['E']==s.expert_batch,'two-rank original profile expert batch unchanged')
 def after_prefill(self,*,rows,shape):
  value=self.read();self.validate_plan(value)
  need(self.prepared is not None and value['live_Impls']==1,'sole headful Engine after prefill')
  need(all(value[k]==self.prepared[k]for k in self.NAMES if k not in('live_Impls','active_prefill')),'immutable prepared geometry')
  need(rows==value['N'] and set(shape)=={'passes','chunk_tokens','attention_batch'},'actual prompt and shape')
  if value['mode']in(2,3):need(shape==dict(passes=2,chunk_tokens=value['cap'],attention_batch=value['B']),'same actual two-span geometry')
  else:need(0<shape['chunk_tokens']<=value['cap'] and 0<shape['attention_batch']<=value['B'] and shape['passes']==(rows+shape['chunk_tokens']-1)//shape['chunk_tokens'],'ordinary actual capacity/pass conservation')
  self.direct531_info=validate531(self.direct531_before,read531(self.library,enabled=self.binding['features']['direct_fp8']),bank=self.binding['bank_format_binding'],mode=value['mode'])
  self.fast534_info=validate534(self.fast534_before,read534(self.library),plan=value,shape=shape,direct=self.direct531_info,bank=self.binding['bank_format_binding'])
  need(read554(self.library)==self.pass554_before,'558 cannot attribute execution to554')
  self.pass558_info=validate558(self.pass558_before,read558(self.library),plan=value,direct=self.direct531_info)
  self.prefill=value;return value
 def transport_request(self,raw):
  rebased=transport_rebase391(self.transport_before,raw);self.transport_last=dict(raw=list(raw),request=list(rebased));return rebased
 def after_engine_close(self):
  value=self.read();need(value['prepare_epoch']==self.prepared['prepare_epoch'] and value['live_Impls']==0,'all Engines closed')
  need(all(value[k]==self.prepared[k]for k in self.NAMES if k not in('live_Impls','active_prefill')),'close preserves immutable plan')
  need(self.direct531_info is not None and read531(self.library,enabled=self.binding['features']['direct_fp8'])==self.direct531_info['after_prefill'],'generation/close cannot mutate completed prefill531 counters')
  need(read534(self.library)==self.fast534_info['after_prefill'],'Engine close cannot mutate completed534 request counters')
  need(read554(self.library)==self.pass554_before,'legacy554 remains inactive')
  need(self.pass558_info is not None and read558(self.library)==self.pass558_info['after_prefill'],'generation/close cannot mutate completed558 staging')
  print(json.dumps(dict(actual_pass558=self.pass558_info)),flush=True)
  return dict(prepared=self.prepared,prefill=self.prefill,after_engine_close=value,direct531=self.direct531_info,fast534=self.fast534_info,pass_upload558=self.pass558_info,transport391=dict(before_engine=self.transport_before,last_validated=self.transport_last),scope='actual native528 immutable request geometry, raw lifetime/error counters retained')

def ordinary_unchanged(before,after,label):
 need(before==after,'ordinary prefill changed wavefront-only '+label)
 return dict(before_prefill=before,after_prefill=after,scope='ordinary516 path; '+label+' remains inactive',performance_eligible=False)

def ordinary_geometry256(before,after):
 need(set(before)==set(after)=={'geometry335','wavefront325'},'ordinary256 typed335/325 field closure')
 a=after['geometry335'];w=after['wavefront325']
 need(before==dict(geometry335=[335]+[0]*31,wavefront325=[325]+[0]*47),'fresh ordinary256 request counters before prefill')
 need(len(a)==32 and len(w)==48 and all(type(v)is int and v>=0 for v in a+w),'ordinary256 unsigned335/325 words')
 need(w==before['wavefront325'] and all(a[i]==before['geometry335'][i] for i in range(32)if i not in(20,22,26)),'ordinary256 request/wavefront/headless/error counters must remain inactive')
 need(a[20]>0 and a[22]>0 and a[26]>=512<<20,'ordinary256 actual scratch/planned features and sampled512MiB free reserve')
 return dict(before_prefill=before,after_prefill=after,memory_observations=dict(scratch_bytes=a[20],planned_feature_reserve_bytes=a[22],sampled_min_free_bytes=a[26]),scope='ordinary256 native prefill memory accounting only; fixed335 request, original317/wavefront325 and headless remain inactive',performance_eligible=False)

def ordinary_k14(before,after,*,prompt_tokens):
 names=tuple(before);need(set(before)==set(after),'334 field closure')
 a=list(before.values());b=list(after.values());need(a[:2]==b[:2]==[334,1]and a[10:15]==b[10:15]==[49280,16,128,4,512],'334 latched resources')
 d=[y-x for x,y in zip(a,b)];need(all(x>=0 for x in d[2:9])and d[2]==d[3]+d[5]+d[6]and d[5]==d[15]==0,'334 ordinary dispatch conservation/no OFF/raw fixture')
 need(d[4]>=d[3]and(d[3]==0 or(d[7]>0 and d[8]>0))and a[9]<=b[9]<=max(a[9],prompt_tokens),'334 selected rows/grid/global maximum extent')
 return dict(before_prefill=before,after_prefill=after,delta16=d,scope='actual334 ordinary fallback request; no filled-wavefront census or per-device attribution')
def k14_before(value):
 a=list(value.values());need(a[:2]==[334,1]and a[10:15]==[49280,16,128,4,512]and a[5]==a[15]==0,'334 resources/no OFF or raw-fixture dispatch; cumulative counts allowed')
def filled_k14(before,after,*,prompt_tokens,shape,wavefront):
 k14_before(before);ordinary=ordinary_k14(before,after,prompt_tokens=prompt_tokens);d=ordinary['delta16']
 need(prompt_tokens==523712 and shape==dict(passes=2,chunk_tokens=262144,attention_batch=1024),'exact original filled512 geometry')
 need((d[2],d[3],d[4],d[5],d[6],d[7],d[15])==(1567,1536,1571136,0,31,6138,0)and d[8]>0,'unchanged exact filled334 additive census')
 need(wavefront['after_prefill']['GPU0_layers']==wavefront['after_prefill']['GPU1_layers']==40,'actual filled two-rank layers')
 ordinary.update(scope='exact filled334 additive census; maximum_live_keys is cumulative/nonadditive and not a current-request measurement',maximum_live_keys_scope='process-global maximum; after <= max(before, request rows)')
 return ordinary
def ordinary_output16(runtime,engine,*,pair145_before,shape):
 v=(C.c_uint64*16)();f=engine.lib.ds41_native_engine_output16_info_v252;f.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t];f.restype=C.c_int;need(f(engine.handle,v,16)==0,'typed252 output owner')
 keys=('schema','registered_matrices','registered_compact_bytes','compact_mode_enabled','prefill_compact_projection_uploads','prefill_compact_H2D_bytes','prefill_compact_FP8_encodes','last_staging_capacity','maximum_compact_triple_bytes','prefill_raw_projection_uploads','cache_compact_entries','cache_compact_H2D_bytes','cache_inverse_original_row_bytes','cache_max_device_temporary_bytes','last_actual_prefill_cap','last_actual_prefill_sub');info=dict(zip(keys,map(int,v)))
 need(list(v)[:4]==[252,0,0,1]and v[9]==0 and v[4]==v[6]and v[8]<=v[7]==37602048,'unchanged252 compact feeding/staging/no raw uploads')
 paired=runtime.registry.pair32_info();upload=runtime.registry.pair32_upload_info();bank=runtime.registry.store.bank_format_binding_v146
 need(paired==(145,139,46080,305764761600,bank['owner_generation'],bank['format_epoch'],1,0),'same immutable paired139 bank registry')
 need(len(pair145_before)==len(upload)==12 and tuple(upload[:2])==(145,139)and not any(upload[10:]),'actual145 ABI/zero failures')
 d=[upload[i]-pair145_before[i]for i in range(2,6)];need(all(x>=0 for x in d)and d[0]==shape['passes']and d[1]==v[4]and d[2]==d[3]==v[5],'actual ordinary pass consumers/jobs/byte conservation')
 need(upload[6]==upload[9]==19906560 and upload[7]==v[14]==shape['chunk_tokens']and upload[8]==v[15]==shape['attention_batch'],'actual compact cap/batch/scratch')
 info.update(paired145_registry=list(paired),paired145_upload=list(upload),paired145_upload_before=list(pair145_before),paired145_actual_prefill_consumers=d[0],serving_scope='ordinary516 GPU-native prefill; GPU1 generation remains selected')
 return info

def verify_delta(a,b):
 def valid(r):
  w=r['words'];rc=r['native_rc'];need(rc in(0,2),'verify getter status')
  if rc==2:need(not any(w),'absent workspace has no fabricated words');return None
  need(len(w)==12 and w[0]in(449,456)and w[1:3]==[1,1]and w[8]==w[10]==w[11]==0,'verify mode/workspace/idle/error')
  need(w[5]+w[6]==w[3]and w[3]<=w[4]<=6*w[3]and w[7]<=w[3],'verify successful-window conservation');return w
 x,y=valid(a),valid(b)
 if y is None:need(x is None,'live verifier workspace disappeared before Engine close');d=[0]*6
 else:d=[y[i]-(x[i]if x is not None else 0)for i in range(3,9)]
 need(all(v>=0 for v in d)and d[2]+d[3]==d[0]and d[0]<=d[1]<=6*d[0]and d[4]<=d[0]and d[5]==0,'API verifier generation conservation; zero valid for no target verify')
 return dict(zip(('successful_calls','rows','eligible_many_windows','scalar_windows','reused_successes','errors'),d),baseline_workspace_absent=x is None,workspace_absent_after=y is None)
def bulk_delta(a,b,validate):
 validate(a);validate(b);d=[y-x for x,y in zip(a,b)];need(all(v>=0 for v in d[6:13])and d[7]==d[8]==d[9]and d[10]==d[11]==d[12]and max(d[7:13])<=d[6],'API bulk job/phase conservation; zero allowed');return d
def epoch_delta(a,b,bulk,validate):
 validate(a);validate(b);d=[y-x for x,y in zip(a,b)];need(all(v>=0 for v in d[3:12])and d[3]==d[4]==d[5]==d[10]==d[11]==bulk[6],'API actual epoch/bulk conservation');return d
def task489_delta(a,b,bulk,validate):
 validate(a);validate(b);d=[y-x for x,y in zip(a['words'],b['words'])];need(all(v>=0 for v in d[3:])and d[3]==d[4]==bulk[6]and d[5]==d[11]==d[12]==d[15]==0 and d[9]==d[13]and d[10]==d[14],'API actual489 decision conservation, optional all-miss/zero jobs');return dict(words=d,scope='actual decision counters; no assertion every request hits GPU cache')
def transport_validate(a,whole,*,closed=False,startup=False):
 w=whole['whole_raw'];need(len(a)==16 and a[:2]==[391,16]and not any(a[14:])and a[2]==int(not closed)and a[3:6]==[1,0,1],'actual391 mode/lifetime/errors')
 n=a[8]+a[9];need(n>0 and a[8]==0 and a[9]==n and a[7]==a[6]and a[10]==a[13]==n*640 and a[11]==n*176896 and a[12]==a[11]-a[10],'actual391 DOWN branch/byte conservation')
 need(n==w[43]+1 and a[6]==w[39]+1,'actual391/Owner chunks plus startup prime')
 if w[17]==0:need(w[29]==0 and not any(w[i]for i in(18,19,39,43))and a[6]==n==1,'prime-only owner is valid for zero GPU work')
 else:need(w[29]==a[6]*736+n*(65920+640),'actual Owner H2D ledger')
 if startup:need(w[17]==0 and not closed,'actual no real work before generation')
def transport_delta(a,b):
 d=[y-x for x,y in zip(a,b)];need(a[:2]==b[:2]==[391,16]and all(v>=0 for v in d[3:])and not any(d[14:]),'actual391 monotonic/errors')
 n=d[8]+d[9];need(d[8]==0 and d[9]==n and d[7]==d[6]and d[10]==d[13]==n*640 and d[11]==n*176896 and d[12]==d[11]-d[10],'API actual391 delta bytes, zero valid');return d
def cutoff_after(before,after,*,prompt_tokens,verify_calls,validate):
 validate(before);validate(after);need(after[6]>=before[6],'candidate position cannot move backwards')
 if after[6]>before[6]:validate(after,completed=True);need(after[6]>=prompt_tokens,'actual source20 position precedes prompt')
 elif verify_calls:raise RuntimeError('actual successful verify did not advance candidate position')
 return after[6]>before[6]


def candidate_state375(v,*,completed=False):
 need(len(v)==24 and all(type(x)is int and 0<=x<1<<64 for x in v),'375 unsigned typed24words')
 need(v[0]==375 and v[1]==1 and v[3]in(262144,524288)and v[2]=={262144:33280,524288:66304}[v[3]]and v[4:6]==[8,2048],'375 explicit current geometry/workspace')
 need(v[13:17]==[0]*4 and v[17:20]==[0,1,0]and v[20:]==[0]*4,'375 idle copied headful GPU0 state/status/reserved')
 positions,blocks,remaining,prefix,mask,k,mode=v[6:13]
 need(positions<=v[3]and blocks==(positions+7)//8 and k==min(blocks,2048),'375 actual live positions/ceil-blocks/min-topk')
 expected=0 if positions==0 else(1 if k>=blocks or blocks<4096 else 3)
 need(mode==expected,'375 actual zero/literal/multiCTA dispatch mode')
 if mode in(0,1):need(prefix==mask==0 and remaining==k,'375 literal fallback or cleared/zero state must retain initialized header')
 else:need(mask==0xffffffff and 0<=prefix<=0xffffffff and 1<=remaining<=k,'375 completed multiCTA full32-bit cutoff and positive tie quota')
 if completed:need(positions>0,'375 completed candidate snapshot must have actual live positions')
 return dict(mode=mode,positions=positions,blocks=blocks,k=k,scope='last-call device state; mode0 cannot distinguish unused from zero-live, mode1 is literal fallback, mode3 proves completed multiCTA state, not launch count')

def read_transport391(library):
 f=library.ds391_whole_down_transport_info;f.argtypes=[C.c_void_p,C.c_int];f.restype=C.c_int
 out=(C.c_uint64*16)();need(f(out,16)==0,'391 actual process-global getter before Engine');return list(map(int,out))
def transport_snapshot391(a,*,live=None):
 need(len(a)==16 and all(type(v)is int and 0<=v<1<<64 for v in a)and a[:2]==[391,16]and not any(a[14:]),'391 unsigned global schema/errors/reserved')
 need(a[2]in(0,1)and(live is None or a[2]==live)and a[3]==a[4]+a[5],'391 sole Operator and cumulative constructor/mode conservation')
 n=a[8]+a[9];need(n>=a[6]and a[7]<=a[6]and a[10]==a[13]==176896*a[8]+640*a[9]and a[11]==176896*n and a[12]==a[11]-a[10],'391 cumulative completed branch/job/byte conservation')
 return a
def transport_rebase391(before,after):
 transport_snapshot391(before,live=0);transport_snapshot391(after)
 d=list(after)
 for i in range(3,14):d[i]-=before[i]
 need(all(v>=0 for v in d[3:14])and d[3:6]==[1,0,1],'391 exactly one new metadataON Operator per request; prior constructors/jobs are history')
 return d


def raw_words(library,symbol,count):
 f=getattr(library,symbol);f.argtypes=[C.POINTER(C.c_uint64),C.c_int];f.restype=C.c_int
 values=(C.c_uint64*count)();need(f(values,count)==0,'actual '+symbol);return list(map(int,values))

def frame_census(plan):
 a,b=plan['first_rows'],plan['second_rows'];carry=plan['carry_rows']
 need(plan['mode']in(2,3) and a+b==plan['N'] and a%128==0,'frame census actual partition')
 forward=[];reverse=[]
 for layer in range(40):
  ratio=2 if layer in(2,8,14) else(1 if layer==20 else 0)
  forward.append(131072+(a//ratio*1280 if ratio else 0))
  reverse.append(131072+(b//ratio*1280 if ratio else 0)+(4096*carry if ratio==2 else 0))
 return dict(forward=sum(forward),reverse=sum(reverse),pinned_capacity=2*(max(forward)+max(reverse)),source_frames=4,carry_bytes=carry*3*4096)

def generic317(before,after,*,plan,shape):
 names=('version','begun','layers','finished','aborted','live_states','last_c0','last_S','last_p0','last_suffix','last_next_layer','recorded_layer_events')
 need(set(before)==set(after)==set(names),'317 named schema')
 old=[before[k]for k in names];actual=[after[k]for k in names];need(old==[317]+[0]*11,'fresh317 request counters')
 if plan['mode']in(1,4):need(actual==old,'ordinary path must not execute317 seam');expected=old
 else:
  suffix=plan['second_rows'];last_suffix=((suffix-4096)//plan['B'])*plan['B'] if suffix>8192 else 0
  expected=[317,2,80,2,0,0,plan['split'],suffix,plan['split'],last_suffix,40,80]
  need(actual==expected,'actual two-span317 layer/order/error/lifetime census')
 return dict(before_prefill=before,after_prefill=after,expected_after_prefill=dict(zip(names,expected)),plan=dict(plan),actual_prefill_shape=shape,quality_admitted=False,performance_admitted=False)

def generic325(values,*,plan):
 need(len(values)==48 and values[0]==325 and all(type(x)is int and x>=0 for x in values),'actual325 unsigned schema')
 if plan['mode']!=2:need(values==[325]+[0]*47,'nonwave path cannot report hidden wavefront work');return
 expected={1:1,2:1,3:1,4:0,5:0,6:40,7:40,8:40,9:40,10:40,11:40,17:1,18:1,19:2,20:3932160,21:10240,22:1,23:1,24:plan['N'],25:4,26:4,27:0,28:0,29:0,30:0,31:0,34:1,35:20<<30,38:1,39:0,41:0,42:0,43:126,44:252,45:0,46:0,47:0}
 census=frame_census(plan);expected.update({12:census['forward'],13:census['reverse'],14:2*census['forward'],15:2*census['reverse'],16:census['pinned_capacity']})
 need(all(values[i]==n for i,n in expected.items()),'actual dynamic325 frames/carry/ACK/roles/close/error conservation')
 need(min(values[32:34])>=512<<20 and 0<values[37]<values[36] and values[40]>0,'headless reserve/owned ledger and actual elapsed time')

def generic335(before,after,*,plan):
 need(set(before)==set(after)=={'geometry335','wavefront325'},'actual335/325 fields')
 need(before==dict(geometry335=[335]+[0]*31,wavefront325=[325]+[0]*47),'fresh335/325 request counters')
 a=after['geometry335'];w=after['wavefront325'];need(len(a)==32 and a[0]==335 and all(type(v)is int and v>=0 for v in a),'335 schema')
 generic325(w,plan=plan)
 if plan['mode']in(1,4):
  ordinary_geometry256(before,after)
 else:
  expected={1:1,2:2 if plan['mode']==2 else 1,3:plan['context'],4:plan['N'],5:plan['split'],6:plan['cap'],7:plan['B'],8:plan['E'],9:1,10:1,11:0,12:0,13:2,14:80,15:2,16:0,17:0,18:40 if plan['mode']==2 else 0,19:40 if plan['mode']==2 else 0,28:0,29:0,30:0,31:0}
  need(all(a[i]==v for i,v in expected.items()),'actual dynamic335 execution/geometry/layer/error census')
  need(a[20]>0 and a[22]>0 and a[26]>=512<<20,'headful scratch/reserve')
  if plan['mode']==2:need(a[21]>0 and a[23]>0 and a[27]>=512<<20 and 0<a[24]<a[25]<=20<<30 and a[21]+a[23]+a[24]<=20<<30,'headless real memory budget')
  else:need(all(a[i]==0 for i in(21,23,24,25,27)),'serial must not invent a headless allocation')
 return dict(before_prefill=before,after_prefill=after,plan=dict(plan),frame_census=frame_census(plan)if plan['mode']in(2,3)else None,scope='actual immutable528 geometry and original absolute error/ownership counters',quality_admitted=False,performance_admitted=False)

def generic512(library,*,plan):
 w=raw_words(library,'ds325_prefill_wavefront_info',48);generic325(w,plan=plan)
 early=raw_words(library,'ds337_early_attention_info',24)
 copies=raw_words(library,'ds331_rank_stage_copy_info',16)
 need(early[0]==337 and copies[0]==331,'actual512 early/copy schemas')
 if plan['mode']==2:
  need(early[1]==2 and all(early[i]==40 for i in range(2,12)) and all(early[i]==0 for i in range(12,17)) and early[17:20]==[8,72,0] and early[20:24]==[40]*4,'actual337 callback/event/ACK/error census')
  need(copies[1:4]==[2,1,1] and min(copies[4:10])>0 and copies[4]==copies[6] and copies[5]==copies[7] and copies[8]==copies[4]*6635520 and copies[9]==copies[5]*6635520 and not any(copies[10:]),'actual331 two pools, copy progress, zero issuer/copy/live failures')
 else:
  need(early==[337,1]+[0]*22 and copies==[331,1]+[0]*14,'ordinary/serial disables private rank paths without fake unlatching')
 return dict(wavefront325=w,early337=early,copy331=copies,plan=dict(plan),frame_census=frame_census(plan)if plan['mode']in(2,3)else None)


def output16_generic(runtime,engine,*,pair145_before,shape,plan,direct):
 value=(C.c_uint64*16)();fn=engine.lib.ds41_native_engine_output16_info_v252;fn.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t];fn.restype=C.c_int
 need(fn(engine.handle,value,16)==0,'actual252 owner')
 v=list(map(int,value));keys=('schema','registered_matrices','registered_compact_bytes','compact_mode_enabled','prefill_compact_projection_uploads','prefill_compact_H2D_bytes','prefill_compact_FP8_encodes','last_staging_capacity','maximum_compact_triple_bytes','prefill_raw_projection_uploads','cache_compact_entries','cache_compact_H2D_bytes','cache_inverse_original_row_bytes','cache_max_device_temporary_bytes','last_actual_prefill_cap','last_actual_prefill_sub')
 need(v[:4]==[252,0,0,1] and v[9]==0 and v[4]==v[6] and v[8]<=v[7]==37602048,'252 actual compact uploads/FP8 encodes/no raw fallback')
 paired=runtime.registry.pair32_info();upload=runtime.registry.pair32_upload_info();bank=runtime.registry.store.bank_format_binding_v146
 need(tuple(paired)==(145,139,46080,305764761600,bank['owner_generation'],bank['format_epoch'],1,0),'same immutable139 bank')
 need(len(pair145_before)==len(upload)==12 and tuple(upload[:2])==(145,139) and not any(upload[10:]),'actual145 no failures')
 delta=[upload[i]-pair145_before[i]for i in range(2,6)]
 need(all(x>=0 for x in delta) and delta[0]==(2 if plan['mode']==2 else 1),'actual prefill Engine consumers, not token-span count')
 need(delta[1]+direct['projections']==v[4] and delta[2]+direct['logical_input_bytes']==delta[3]+direct['logical_input_bytes']==v[5],'actual145 inverses plus531 direct projections/bytes ==252 uploads; no fabricated inverse work')
 need(upload[6]==upload[9]==19906560 and upload[7]==v[14]==shape['chunk_tokens'] and upload[8]==v[15]==shape['attention_batch'],'actual dynamic compact cap/sub')
 return dict(zip(keys,v),paired145_registry=list(paired),paired145_upload=list(upload),paired145_upload_before=list(pair145_before),paired145_upload_delta=[upload[i]-pair145_before[i] if i in range(2,6) else upload[i] for i in range(12)],paired145_delta=delta,plan=dict(plan))


def read531(library,*,enabled):
 f=library.ds531_prefill_direct_info;f.argtypes=[C.c_int,C.POINTER(C.c_uint64),C.c_size_t];f.restype=C.c_int
 rows=[]
 for device in (0,1):
  out=(C.c_uint64*16)();need(f(device,out,16)==0,'actual531 typed per-device getter')
  row=list(map(int,out));need(row[:4]==[531,16,device,int(enabled)] and row[10]==row[11]==0 and row[14:]==[0,0],'actual531 policy/absolute errors/lifetime/reserved')
  rows.append(row)
 return rows

def validate531(before,after,*,bank,mode):
 need(len(before)==len(after)==2,'actual two logical-device531 snapshots')
 deltas=[]
 for device,(a,b)in enumerate(zip(before,after)):
  need(a[:4]==b[:4]==[531,16,device,1] and b[10]==b[11]==0 and b[14:]==[0,0],'mandatory531 policy and absolute error/lifetime state')
  d=[b[i]-a[i] for i in range(4,10)]
  need(all(x>=0 for x in d) and d[0]==d[1] and d[2]==3*d[1] and d[3]==19906560*d[1] and d[4]==35389440*d[1] and d[5]==38912*d[1],'actual531 Scope/triple/projection/logicalbyte conservation')
  if d[0]:need(b[12:14]==[bank['owner_generation'],bank['format_epoch']],'531 exact current bank identity, no historical rebinding')
  need(d[1]>0 if device==0 or mode==2 else d[1]==0,'actual531 required rank execution; no unscoped slow conversion')
  deltas.append(d)
 return dict(before_engine=before,after_prefill=after,delta_per_device=deltas,projections=sum(d[2]for d in deltas),logical_input_bytes=sum(d[3]for d in deltas),scope='actual enqueued531 work plus synchronized native prefill completion; not standalone model-quality admission')


def read534(library):
 row=raw_words(library,'ds534_fp8_fastpath_info',32)
 need(row[:2]==[534,32] and row[2]==row[17]==row[18]==row[21]==0 and all(row[i]==0 for i in(25,27,31)),'actual534 idle/absolute errors/lease/reserved')
 return row

def validate534(before,after,*,plan,shape,direct,bank):
 if plan['mode']!=4:
  need(before==after,'two-rank work must not inherit attribution from a previous one-span534 request')
  return dict(before_engine=before,after_prefill=after,executed_for_this_request=False)
 #534 is REQUEST-LOCAL, unlike cumulative531. Never subtract old534 counters.
 a=after
 need(a[:9]==[534,32,0,4,plan['context'],plan['N'],plan['position'],plan['prepare_epoch'],plan['E']],'actual534 current immutable request geometry')
 need(a[9]==a[16]==a[19]==a[20]==1 and a[17]==a[18]==a[21]==0,'one actual534 lease/request with checked successful close, no live/error state')
 need(a[26]==shape['passes'] and a[11]==40*a[26] and a[10]==a[11]-1,'actual534 one lease reused across all real layers/pass')
 need(a[12]==a[14]>0 and a[15]==3*a[14] and a[13]==19906560*a[14],'actual534 expert/projection/logical transfer bytes')
 need(a[22:25]==[bank['owner_generation'],bank['format_epoch'],0] and a[28:31]==[shape['chunk_tokens'],shape['attention_batch'],1],'actual534 owner/device/cap/batch/mandatory531')
 need(a[15]==direct['projections'] and a[13]==direct['logical_input_bytes'],'actual534 producer and531 encoder conservation')
 return dict(before_engine=before,after_prefill=after,executed_for_this_request=True,scope='actual request-local534 optimized FP8 one-span execution; no ordinary fallback or prior-request attribution')

def read554(library):
 row=raw_words(library,'ds554_pass_upload_info',24)
 need(row[:2]==[554,24] and all(row[i]==0 for i in (10,11,12,13,18,19,20,21)),'554 absolute idle/errors/device/pinned lifetime')
 return row

def validate554(before,after,*,plan,direct):
 need(len(before)==len(after)==24 and before[:2]==after[:2]==[554,24],'554 actual24-word schema')
 need(all(before[i]==after[i]==0 for i in (10,11,12,13,18,19,20,21)),'554 no live/error/pinned/device state at boundary')
 if plan['mode']==4:
  need(before==after,'mode4 must retain original534 owner, no554 attribution')
  return dict(before_engine=before,after_prefill=after,executed_for_this_request=False)
 need(plan['mode']in(2,3),'554 exact pass mode required')
 d=[b-a for a,b in zip(before,after)]
 need(all(d[i]>=0 for i in range(2,18)),'554 cumulative counters cannot decrease')
 count=(1,1)if plan['mode']==2 else(2,0)
 for dev,n in enumerate(count):
  need(d[2+dev]==d[8+dev]==n and d[4+dev]==d[6+dev]==40*n,'554 actual created/layers/drains/closed census')
  triples=direct['delta_per_device'][dev][1]
  need(d[14+dev]==triples and d[16+dev]==19906560*triples,'554 enqueued logical upload bytes match independent531 encoder')
  need(after[22+dev]==plan['prepare_epoch']if n else after[22+dev]==before[22+dev],'554 actual request epoch per active rank')
 return dict(before_engine=before,after_prefill=after,delta_words=d,executed_for_this_request=True,scope='actual per-rank pass staging; original per-layer positive drains retained')

def helper_off528(before,after,policy,settings,*,plan):
 policy.assert_current(settings)
 need(dict(policy.environment).get('DS1017_PAIR_INDEXER_PREFILL_V1')=='0','528 old pair helper must remain immutableOFF')
 need(len(before)==len(after)==20 and all(type(x)is int and 0<=x<1<<64 for x in list(before)+list(after)),'528 old helper unsigned20-word schema')
 need(before[0]==after[0]==1 and not any(before[2:]) and not any(after[2:]),'528 old helper actual owner/work/bytes/errors must be zero')
 delta=after[1]-before[1]
 need(plan['context']==settings.context and settings.context in(262144,524288),'528 helper context identity')
 need(delta==int(plan['mode']==4 and settings.context==524288),'528 exact context-specific disabled-constructor census; no old helper execution')
 return dict(policy='OFF; selected native localK14',before=list(before),after=list(after),disabled_constructor_delta=delta,old_helper_used=False,scope='field1 is disabled constructor history; fields2..19 actual owner/work/bytes/errors stay zero')

def read558(library):
 row=raw_words(library,'ds558_pass_upload_info',28)
 need(row[:2]==[558,28] and all(type(x)is int and 0<=x<1<<64 for x in row),'558 unsigned28-word schema')
 need(all(row[i]==0 for i in (6,7,10,11,12,13,18,19,20,21,26,27)),'558 absolute idle/errors/exception/abort/device/pinned boundary')
 return row

def validate558(before,after,*,plan,direct):
 need(len(before)==len(after)==28 and before[:2]==after[:2]==[558,28],'558 actual28-word schema')
 need(all(type(x)is int and 0<=x<1<<64 for x in before+after),'558 unsigned counters')
 need(all(before[i]==after[i]==0 for i in (6,7,10,11,12,13,18,19,20,21,26,27)),'558 completed boundary has no live/errors/exception/abort/device/pinned state')
 if plan['mode']==4:
  need(before==after,'mode4 retains literal534, no558 attribution')
  return dict(before_engine=before,after_prefill=after,executed_for_this_request=False)
 need(plan['mode']in(2,3),'558 exact pass mode required')
 d=[b-a for a,b in zip(before,after)]
 need(all(d[i]>=0 for i in range(2,28)),'558 cumulative counters cannot decrease')
 count=(1,1)if plan['mode']==2 else(2,0)
 for dev,n in enumerate(count):
  need(d[2+dev]==d[8+dev]==d[24+dev]==n and d[4+dev]==40*n,'558 owners/enqueue-releases/closed/positive-pass-drains census')
  need(d[6+dev]==d[26+dev]==0,'558 successful request has no exception drains/aborts')
  triples=direct['delta_per_device'][dev][1]
  need(d[14+dev]==triples and d[16+dev]==19906560*triples,'558 uploaded bytes match independent531 encoder')
  need(after[22+dev]==plan['prepare_epoch']if n else after[22+dev]==before[22+dev],'558 actual request epoch per active rank')
 return dict(before_engine=before,after_prefill=after,delta_words=d,executed_for_this_request=True,scope='successful layer release is enqueue-only; positive pass close witnesses GPU completion')
