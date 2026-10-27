"""Actual N1 Owner469 diagnostic; no Engine casts, no inferred GPU speed."""
import ctypes as C,time
from fixed_binding_v512 import check,need
NAMES=('Version Words Mode LiveOwner Generation FormatEpoch Catalog Physical Active Spares PermanentStatics WorkerCPU WorkerInitialized Armed TrainingForwards BeforeCalls AfterCalls LayerCursor PartialForward CurrentM TrainingRows Frozen PlanBuilt PlanMoves QueuedMoves Started Ready Published Cancelled Pending MaxPending BackpressureParks ProtectedReady NewRoutesUsed OldRoutesUsed ComputeBegins ComputeUploadPending H2DBytes InverseReadBytes InverseWriteBytes ValidationNs CopyNs PlannerNs PlannerThreadNs StageThreadNs WorkerParked WorkerJoined UploadDrained ComputeDrained LivePinnedBytes LiveDeviceBytes DeclaredPeakBytes Errors Poisoned ActiveTicket Borrowed Closed HookRefs StartupInverseCalls StartupInverseBytes StartupJobs ObserverErrors Reserved62 Reserved63').split()
assert len(NAMES)==64

def read(lib,*,bank,phase,completed=False,absent=False):
 v=check();fn=lib.ds469_fixed_plan_info;fn.argtypes=[C.POINTER(C.c_uint64),C.c_int];fn.restype=C.c_int
 started=time.monotonic();attempts=0
 while True:
  out=(C.c_uint64*64)();rc=int(fn(out,64));attempts+=1
  if rc!=4 or time.monotonic()-started>=0.02:break
  # Metadata queue can be briefly busy; never wait for a GPU event or plan.
  time.sleep(0.0001)
 words=list(map(int,out))
 if absent:
  need(rc==2 and not any(words),'closed owner absent and output zeroed')
  return dict(phase=phase,native_rc=rc,owner_absent=True,attempts=attempts)
 need(rc==0,'actual469 snapshot rc='+str(rc))
 r=dict(zip(NAMES,words));need(r['Version']==469 and r['Words']==64 and r['Mode']==v['mode']and r['LiveOwner']==1,'actual469 mode/lifetime')
 need([r[k]for k in ('Generation','FormatEpoch')]==[bank['owner_generation'],bank['format_epoch']],'same one-bank identity')
 need([r[k]for k in ('Catalog','Physical','Active','Spares','PermanentStatics','WorkerCPU','WorkerInitialized')]==[15360,1078,1070,8,0,253,1],'actual capacity/uploader')
 need(not any(r[k]for k in ('Errors','Poisoned','ActiveTicket','Borrowed','Closed','ObserverErrors','Reserved62','Reserved63')),'clean idle snapshot')
 need(r['TrainingForwards']<=32 and r['BeforeCalls']<=1280 and r['AfterCalls']<=1280 and r['MaxPending']<=8,'finite observation/credits')
 need(r['H2DBytes']==r['InverseReadBytes']==r['InverseWriteBytes']and r['H2DBytes']%6635520==0,'actual complete matrix traffic')
 if v['mode']==0:need(not any(r[k]for k in ('Started','Ready','Published','Pending','H2DBytes')),'matchedOFF nouploads/publication')
 if completed:
  need(r['Armed']==r['Frozen']==1 and r['TrainingForwards']==32 and r['BeforeCalls']==r['AfterCalls']==1280 and r['LayerCursor']==r['PartialForward']==0,'exact successful32-forward seal')
 # PlanBuild can still be pending on worker at final token: record, do not wait.
 return dict(phase=phase,native_rc=rc,attempts=attempts,words=words,fields=r,scope='actual469 cumulative snapshot; pending event is NOT simultaneousCTA or model speed proof')
