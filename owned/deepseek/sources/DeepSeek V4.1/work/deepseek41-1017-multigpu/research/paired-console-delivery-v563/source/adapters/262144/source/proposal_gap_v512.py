"""Private exact selected41_1017 propose seam; inherited45commit/capture unchanged."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

import math,time
import torch
from pathlib import Path
import hashlib,inspect
from native_mtp_commit_graph_v45_1017 import ReferenceDSpark as BaseCommit
_EXPECTED=_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-nvfp4/delivery/single-gpu-frozen-v1/source-snapshot/native_mtp_commit_graph_v45_1017.py')
if Path(inspect.getfile(BaseCommit)).resolve()!=Path(_EXPECTED) or hashlib.sha256(Path(_EXPECTED).read_bytes()).hexdigest()!='63201f33633e527da2a06f01e5c5aaf39c2e500880569b9905b1fb89c2f2c893':raise RuntimeError('386 selected1017 commit45 source binding')
class ReferenceDSpark(BaseCommit):
 def __init__(self,reference,*,gap_control386=None,replay_proof386=False):
  if type(replay_proof386)is not bool:raise ValueError('386 replayproof explicitbool')
  super().__init__(reference);self._gap_control386=gap_control386
  self._replay_event386=torch.cuda.Event(enable_timing=False)if replay_proof386 else None
  self._replay_event_primed386=False
  self.metadata['adaptive_replay386_proof']=dict(enabled=replay_proof386,eligible_armed_replays=0,completion_pending_at_gap_entry=0,completion_pending_after_gap=0,hooks_starting_promotions=0,promotions_started=0,gap_wall_ns=0,scope='nonblocking completionevent on ACTUAL replay/currentstream; pending is not GPUexecutionduration or TPSproof'if replay_proof386 else'NOT_OBSERVED optionalproofOFF')
 def propose(self,pending,main,position):
  if type(position)is not int or not 0<position<self.reference.model.max_seq_len-5:raise ValueError('Actual bounded DSpark position required')
  self.metadata.setdefault('first_position',position);self.metadata['last_position']=position;self.metadata['crossed_window_128']=position//128!=self.metadata['first_position']//128
  self.pending_tensor.fill_(pending);self.position_tensor.fill_(position);self.main_tensor.copy_(self.tensor(main))
  if self.cuda_graph is None:self._capture(pending,position)
  else:
   replay_stream386=torch.cuda.current_stream(self.reference.device)
   self.cuda_graph.replay()
   #Onlythe realgeneration replay can service: capture/warmup stayunarmed.
   if self._gap_control386 is not None and self._gap_control386.armed:
    try:
     if self._replay_event386 is not None:
      if not self._replay_event_primed386:raise RuntimeError('386 proofevent must materialize duringwarmup')
      self._replay_event386.record(replay_stream386)
      pending_before386=not self._replay_event386.query()
      started_before386=self._gap_control386.snapshot()['started']
      start386=time.perf_counter_ns()
     self._gap_control386.gap()
     if self._replay_event386 is not None:
      elapsed386=time.perf_counter_ns()-start386
      pending_after386=not self._replay_event386.query()
      started_after386=self._gap_control386.snapshot()['started']
      delta386=started_after386-started_before386
      if delta386<0:raise RuntimeError('386 actualpromotioncounter regressed')
      proof386=self.metadata['adaptive_replay386_proof']
      proof386['eligible_armed_replays']+=1;proof386['completion_pending_at_gap_entry']+=int(pending_before386)
      proof386['completion_pending_after_gap']+=int(pending_after386);proof386['gap_wall_ns']+=elapsed386
      proof386['hooks_starting_promotions']+=int(delta386>0);proof386['promotions_started']+=delta386
    except BaseException:
     #Exactstream captured BEFOREreplay; callbackcannot redirect ourdrain.
     replay_stream386.synchronize()
     raise
  if self._replay_event386 is not None and not self._replay_event_primed386:
   #Oneeventcreated in__init__, firstmaterialized ONLYunarmedwarmup. The
   #originalbool andgreedyenter sync complete it beforearm; no decodeallocation.
   if self._gap_control386 is not None and self._gap_control386.armed:raise RuntimeError('386 lateproofevent initialization')
   self._replay_event386.record(torch.cuda.current_stream(self.reference.device));self._replay_event_primed386=True
  self.metadata['replays']+=1
  if not bool(self.valid_output):raise ValueError('Nonfinite graphed DSpark output')
  ids=self.output[0][0].tolist()
  if ids[0]!=pending or len(ids)!=6 or any(not 0<=v<129280 for v in ids):raise ValueError('Graphed DSpark token geometry changed')
  self.raw_confidence=[float(v)for v in self.output[2][0].tolist()]
  self.confidence=[1/(1+math.exp(-max(-80.,min(80.,v))))for v in self.raw_confidence]
  return [int(v)for v in ids[1:]]

