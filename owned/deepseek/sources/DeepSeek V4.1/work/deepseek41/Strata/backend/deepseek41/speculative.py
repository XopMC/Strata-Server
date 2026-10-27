"""DSpark proposals, greedy target verification and complete bounded rollback.

The native draft head never determines an emitted token without target
verification. GPU/checkpoint parity and acceptance must be measured separately.
No new CUDA kernels are introduced here.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import torch


@contextmanager
def forced_rejection_audit(model,expected_tokens,*,draft_index=3):
    """Perturb one actual draft for a bounded rollback audit, never serving.

    The replacement differs from the serial target at the same position.
    Native proposal computation still runs and target verification is unchanged.
    """
    if (type(draft_index) is not int or not 1<=draft_index<=5 or
            len(expected_tokens)<=draft_index or model.head.vocab_size<2 or
            any(type(token) is not int or not 0<=token<model.head.vocab_size for token in expected_tokens)):
        raise ValueError('Forced rejection needs bounded serial target token IDs')
    original=model.forward_spec;had_override='forward_spec' in vars(model)
    saved=vars(model).get('forward_spec');record={'injections':0,'diagnostic_only':True,'draft_index':draft_index}
    def proposal(*args,**kwargs):
        result=original(*args,**kwargs)
        if not record['injections']:
            if result is None or result[0].ndim!=2 or result[0].shape[0]!=1 or result[0].shape[1]<=draft_index:
                raise ValueError('Actual draft proposal cannot admit the forced rejection audit')
            tokens=result[0].clone();expected=expected_tokens[draft_index]
            replacement=(expected+1)%model.head.vocab_size
            record.update(injections=1,expected_target_token=expected,
                original_draft_token=int(tokens[0,draft_index]),replacement_draft_token=replacement)
            tokens[0,draft_index]=replacement
            return (tokens,*result[1:])
        return result
    model.forward_spec=proposal
    try:yield record
    finally:
        if had_override:model.forward_spec=saved
        else:delattr(model,'forward_spec')


@dataclass
class PhysicalSnapshot:
    model_id:int
    capacity:int
    logical:dict
    tensors:list
    shared:dict
    bytes:int


def tensor_bytes(value):return value.numel()*value.element_size() if value is not None else 0


@torch.inference_mode()
def complete_snapshot(adapter,maximum_inputs,*,budget_bytes=32*1024**2):
    if type(maximum_inputs) is not int or not 1<=maximum_inputs<=16 or budget_bytes<1:
        raise ValueError('Invalid bounded speculative rollback budget')
    position=adapter.position;model=adapter.model;end=position+maximum_inputs
    if position<1 or end>model.max_seq_len or adapter.failed:
        raise ValueError('Speculative snapshot outside a live admitted sequence')
    views=[]
    for layer in [*model.layers,*model.mtp]:
        attn=layer.attn
        views.append((attn.window_kv_cache,attn.window_kv_cache))
        if attn.is_kv_source:
            first=position//attn.compress_ratio
            last=(end+attn.compress_ratio-1)//attn.compress_ratio
            packed=getattr(attn,'_deepseek41_native_fp4_kv',getattr(attn,'_deepseek41_q4_kv',None))
            if packed is not None:
                views.extend((view,view)for view in packed.raw_buffers(first,last))
            else:
                view=attn.compress_kv_cache[:,first:last]
                views.append((view,view))
            compressor=attn.compressor
            for name in ('kv_state','score_state'):
                value=getattr(compressor,name,None)
                if value is not None:views.append((value,value))
        indexer=attn.indexer
        if indexer is not None and indexer.owns_k:
            first=position//attn.compress_ratio;last=(end+attn.compress_ratio-1)//attn.compress_ratio
            value=indexer.k_cache[:,first:last];views.append((value,value))
    if model.engram_hash is not None:
        value=model.engram_hash.cache[:,position:end];views.append((value,value))
    logical_bytes=sum(tensor_bytes(state.history) for state in adapter.windows.values())
    logical_bytes+=sum(tensor_bytes(state.kv)+tensor_bytes(state.gate) for state in adapter.compressors.values())
    copied_bytes=logical_bytes+sum(tensor_bytes(value) for target,value in views)
    if copied_bytes>budget_bytes:raise MemoryError('Complete speculative state exceeds rollback budget')
    logical=adapter.snapshot()
    return PhysicalSnapshot(id(model),model.max_seq_len,logical,[(target,value.clone()) for target,value in views],
        vars(adapter.graph.shared_attn).copy(),copied_bytes)


@torch.inference_mode()
def restore_complete(adapter,snapshot):
    if not isinstance(snapshot,PhysicalSnapshot) or snapshot.model_id!=id(adapter.model) or snapshot.capacity!=adapter.model.max_seq_len:
        raise ValueError('Speculative state belongs to another model/capacity')
    adapter.restore(snapshot.logical)
    for target,value in snapshot.tensors:
        if target.shape!=value.shape or target.device!=value.device or target.dtype!=value.dtype:
            raise ValueError('Speculative state buffer changed during verification')
        target.copy_(value)
    vars(adapter.graph.shared_attn).clear();vars(adapter.graph.shared_attn).update(snapshot.shared)


@torch.inference_mode()
def commit_main_windows(model,graph,main_hidden,first_position,*,clear=False):
    """Seed/update all DSpark windows from verified target attention inputs.

    This handles multiple accepted target rows and a bounded last-window
    prefill tail, which native forward_spec(start>0) does not accept.
    """
    if (not len(model.mtp) or main_hidden is None or main_hidden.ndim!=3 or main_hidden.shape[0]!=1 or
        not main_hidden.shape[1] or first_position<0 or first_position+main_hidden.shape[1]>model.max_seq_len or
        main_hidden.shape[-1]!=model.mtp[0].main_proj.in_features):
        raise ValueError('DSpark needs verified target attention inputs at their true positions')
    if not bool(torch.isfinite(main_hidden).all()):raise ValueError('Nonfinite DSpark target hidden states')
    device=main_hidden.device
    with torch.device(device):
        first=model.mtp[0]
        main_x=first.main_norm(first.main_proj(main_hidden))
        for layer in model.mtp:
            attention=layer.attn;win=attention.window_size
            if clear:attention.window_kv_cache.zero_()
            hidden=main_x[:,-win:];start=first_position+main_x.shape[1]-hidden.shape[1]
            values=attention.kv_norm(attention.wkv(hidden))
            graph.apply_rotary_emb(values[...,-attention.rope_head_dim:],attention.freqs_cis[start:start+hidden.shape[1]])
            graph.act_quant(values,graph.fp8_block_size,graph.scale_fmt,graph.scale_dtype,True)
            if not bool(torch.isfinite(values).all()):raise ValueError('Nonfinite DSpark window cache')
            positions=torch.arange(start,start+hidden.shape[1],device=device)%win
            attention.window_kv_cache[:,positions]=values


class GreedySpeculator:
    def __init__(self,adapter,*,eos_token_id,verification='batch',rollback_budget_bytes=32*1024**2):
        if verification not in ('batch','sequential') or type(eos_token_id) is not int or not len(adapter.model.mtp):
            raise ValueError('DSpark requires native draft stages, EOS and an explicit verification mode')
        self.adapter,self.model,self.graph=adapter,adapter.model,adapter.graph
        self.eos=eos_token_id;self.verification=verification;self.budget=rollback_budget_bytes
        self.statistics={'proposals':0,'draft_tokens':0,'accepted_draft_tokens':0,'target_forward_calls':0,
            'target_input_tokens':0,'complete_rollbacks':0,'replay_schedule_fallbacks':0,'maximum_rollback_bytes':0,
            'verification':verification,'quality_gate_passed':False}
        self.last_main=None

    @torch.inference_mode()
    def seed(self,main_hidden,first_position):
        if first_position+main_hidden.shape[1]!=self.adapter.position:
            raise ValueError('DSpark seed must end at the verified target cursor')
        commit_main_windows(self.model,self.graph,main_hidden,first_position,clear=True)
        self.last_main=main_hidden[:,-1:].clone()

    def target(self,inputs):
        self.statistics['target_forward_calls']+=1;self.statistics['target_input_tokens']+=inputs.shape[1]
        with torch.device(inputs.device):
            _,logits,main=self.adapter.forward(inputs,self.adapter.position,full_logits=True)
        if logits.ndim!=3 or logits.shape[:2]!=inputs.shape or not bool(torch.isfinite(logits).all()):
            raise ValueError('DSpark target verification requires finite logits for every input')
        if main is None or main.shape[:2]!=inputs.shape or not bool(torch.isfinite(main).all()):
            raise ValueError('DSpark target did not return verified attention inputs')
        return logits.argmax(-1),main

    def rollback(self,snapshot):
        restore_complete(self.adapter,snapshot);self.statistics['complete_rollbacks']+=1

    @torch.inference_mode()
    def generate(self,pending_token,max_new_tokens,*,on_tokens=None,cancel=None):
        if self.last_main is None or self.adapter.position<2 or type(max_new_tokens) is not int or max_new_tokens<1:
            raise ValueError('Seed DSpark after a target prompt of at least two tokens')
        if type(pending_token) is not int or not 0<=pending_token<self.model.head.vocab_size:
            raise ValueError('Invalid target-selected pending token')
        if self.adapter.position+max_new_tokens>self.model.max_seq_len:
            raise ValueError('Speculative generation exceeds admitted context')
        emitted=[pending_token];device=self.last_main.device
        if on_tokens:on_tokens([pending_token])
        block=self.model.mtp[0].block_size
        while len(emitted)<max_new_tokens and pending_token!=self.eos:
            if cancel is not None and cancel.is_set():break
            start=self.adapter.position
            count=min(block,max_new_tokens-len(emitted)-1,self.model.max_seq_len-start-1)
            snapshot=complete_snapshot(self.adapter,count+1,budget_bytes=self.budget)
            self.statistics['maximum_rollback_bytes']=max(self.statistics['maximum_rollback_bytes'],snapshot.bytes)
            try:
                known=torch.tensor([pending_token],dtype=torch.long,device=device)
                drafts=[]
                if count:
                    with torch.device(device):proposal=self.model.forward_spec(known,self.last_main,start-1)
                    if proposal is None or proposal[0].shape!=(1,block+1) or not bool(torch.isfinite(proposal[1]).all()):
                        raise ValueError('DSpark proposal shape/logits invalid')
                    if int(proposal[0][0,0])!=pending_token:raise ValueError('DSpark changed the target-selected token')
                    drafts=proposal[0][0,1:count+1].tolist()
                    self.statistics['proposals']+=1;self.statistics['draft_tokens']+=len(drafts)
                inputs=torch.tensor([[pending_token,*drafts]],dtype=torch.long,device=device)
                if self.verification=='sequential':
                    predictions=[];main_parts=[];accepted=0;terminal=False
                    for index in range(inputs.shape[1]):
                        selected,hidden=self.target(inputs[:,index:index+1])
                        main_parts.append(hidden);token=int(selected[0,0]);predictions.append(token)
                        if token==self.eos:terminal=True;break
                        if index>=len(drafts) or token!=drafts[index]:break
                        accepted+=1
                    main=torch.cat(main_parts,1)
                    new=predictions
                    # The proposal touched only a previously verified window
                    # row. Recommit exactly the target rows actually consumed.
                    commit_main_windows(self.model,self.graph,main,start)
                else:
                    selected,main=self.target(inputs);predictions=selected[0].tolist()
                    accepted=0;terminal=False
                    for index,draft in enumerate(drafts):
                        if predictions[index]!=draft:break
                        if draft==self.eos:terminal=True;break
                        accepted+=1
                    new=drafts[:accepted]+[predictions[accepted]]
                    consumed=accepted+1
                    if accepted<len(drafts) or terminal:
                        self.rollback(snapshot)
                        replay,replayed_main=self.target(inputs[:,:consumed])
                        if replay[0,:accepted].tolist()!=drafts[:accepted]:
                            self.statistics['replay_schedule_fallbacks']+=1
                            self.rollback(snapshot)
                            replay,replayed_main=self.target(inputs[:,:1]);accepted=0;consumed=1
                            new=[int(replay[0,0])]
                        else:new[-1]=int(replay[0,-1])
                        main=replayed_main
                    commit_main_windows(self.model,self.graph,main[:,:consumed],start)
                # No tokens beyond the first EOS or the requested count escape.
                if self.eos in new:new=new[:new.index(self.eos)+1]
                self.statistics['accepted_draft_tokens']+=accepted
                accepted_output=new[:max_new_tokens-len(emitted)]
                emitted.extend(accepted_output)
                if on_tokens:on_tokens(accepted_output)
                pending_token=emitted[-1];self.last_main=main[:,-1:].clone()
            except BaseException:
                self.rollback(snapshot);raise
        self.statistics['acceptance_rate']=(self.statistics['accepted_draft_tokens']/self.statistics['draft_tokens']
            if self.statistics['draft_tokens'] else 0.)
        return emitted,dict(self.statistics)
