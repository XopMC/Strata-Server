"""Explicit failed-child bank revalidation175 (returned or no-returned runtime). No IO/native/hardware on import.
Never reaccepts a failed model result, repeats a workload, resets GPU or mutates weights.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from maintenance_v146 import epoch,canonical,physical_snapshot,require_ro,need,pinned


def _parent_clean():
    need('torch' not in sys.modules,'revalidation parent imported Torch')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','revalidation parent CUDA mask changed')
    rows=Path('/proc/self/maps').read_text().splitlines()
    need(not any(any(x in row for x in ('libcuda.so','libcudart.so','libnvrtc.so','libcublas.so')) for row in rows),'revalidation parent has accelerator runtime mappings')
    tasks=list(Path('/proc/self/task').iterdir())
    need(len(tasks)==1,'revalidation parent must be single threaded')
    for task in tasks:need(not(task/'children').read_text().strip(),'revalidation parent has live/unreaped child')


def classify_close_scope_v175(close):
    need(isinstance(close,dict) and close.get('schema')==174,'actual174 child receipt required')
    if close.get('phase')=='CHILD_RUNTIME_CLOSED_ZERO_LEASES':
        need(close.get('runtime_created') is True and close.get('runtime_close_completed') is True,'returned runtime requires actual close completion')
        return 'RETURNED_RUNTIME_CLOSED_ZERO_LEASES',True
    need(close.get('phase')=='CHILD_NO_RUNTIME_CREATED_ZERO_LEASES' and close.get('runtime_created') is False and close.get('runtime_close_completed') is False,'no-returned-runtime receipt must not claim native close')
    return 'NO_RETURNED_RUNTIME_NORMAL_EXIT_REAPED_ZERO_LEASES',False


def revalidate_after_reaped_child(manager,command,failed,clean_flag,*,allowed,outer_check):
    need(allowed is True,'failed-child revalidation disabled; Root fresh-start opt-in required')
    keys={'action','operation_id','expected_bank_format_binding','failed_child_pid','close_proof_sha256'}
    need(isinstance(command,dict) and set(command)==keys and command['action']=='revalidate_after_reaped_child','exact revalidation command required')
    ident=command['operation_id'];need(isinstance(ident,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}',ident),'invalid revalidation operation ID')
    need(isinstance(clean_flag,list) and clean_flag==[False] and manager.child_drain_proved is False,'only blocked failed-child state can revalidate')
    need(manager.phase=='CHILD_DRAIN_UNPROVEN_NO_REUSE' and manager.active is None and not manager.poisoned and not manager.threads and not manager.workspaces and not manager._pending(),'active/poisoned/scratch bank cannot revalidate')
    need(isinstance(failed,dict) and isinstance(failed.get('reaped_child'),dict),'actual parent-reaped failure evidence required')
    actual=failed['reaped_child'];reap_path=pinned(actual['reap_proof'])
    reap=json.loads(reap_path.read_text());need(reap==actual['record'],'parent reap receipt changed')
    need(reap.get('schema')==174 and reap.get('phase')=='CHILD_REAPED' and reap.get('parent_pid')==os.getpid(),'actual own-parent reap receipt required')
    need(reap.get('normal_exit') is True and reap.get('exit_code')==1 and reap.get('signal') is None,'only normally reaped failed child exit1 can revalidate')
    need(type(command['failed_child_pid']) is int and command['failed_child_pid']==reap['child_pid'] and reap['child_pid']!=os.getpid(),'failed child identity differs')
    current=manager.binding()
    need(canonical(command['expected_bank_format_binding'])==canonical(current)==canonical(reap.get('bank_format_binding')),'current bank owner/layout/format epoch/inventory differs')
    need(epoch()==manager.parent==manager.store._owner_epoch and manager.store.generation==manager.generation,'actual parent epoch/generation changed')
    need(manager.store._ready and not manager.store._closed and not manager.store.fixture and not manager.store._fds,'bank readiness/FD ownership changed')
    need(getattr(manager.store,'_native_leases',-1)==0,'parent native leases nonzero')
    close_path=Path(reap['job_output'])/'close-proof.json'
    need(command['close_proof_sha256']==actual.get('close_proof_sha256') and isinstance(command['close_proof_sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',command['close_proof_sha256']),'actual close-proof pin required')
    close=json.loads(pinned(dict(path=str(close_path),sha256=command['close_proof_sha256'])).read_text())
    need(not(close_path.parent/'close-failure.json').exists(),'close-failure marker forbids reuse')
    close_scope,native_close_claimed=classify_close_scope_v175(close)
    need(type(close.get('native_leases')) is int and close['native_leases']==0,'actual zero child leases required')
    for key in ('child_pid','job_id','runtime_dso_path','runtime_dso_sha256','bank_format_binding'):
        need(close.get(key)==reap.get(key),'close/reap actual binding differs: '+key)
    need(close.get('parent_pid')==os.getpid() and close.get('quality_admitted') is False and close.get('result_accepted') is False,'close proof cannot assert model acceptance')
    need(not Path('/proc',str(reap['child_pid'])).exists(),'failed child PID still exists/reused')
    outer_check() # exact controller leases/outer child; intentionally no clean-flag check
    _parent_clean();manager._proof_guard()
    need(manager._inventory()==manager.inventory,'immutable allocation inventory changed')
    require_ro(manager.inventory)
    need(physical_snapshot(manager.inventory)==manager.physical,'physical RO/private NUMA bank changed')
    manager.health() # exact pair idle + actual recovery0; no reset/no foreign process action
    _parent_clean();outer_check()
    event_path=manager.root/('child-revalidation-'+ident+'.json')
    need(not event_path.exists() and ident not in getattr(manager,'revalidated_ids174',set()),'duplicate/replayed revalidation ID')
    old=(manager.phase,manager.child_drain_proved,manager.store.bank_runtime_admissible_v146,manager.store.bank_format_binding_v146,clean_flag[0])
    event=dict(schema=175,close_scope=close_scope,native_close_claimed=native_close_claimed,normal_reaped_exit_required=True,phase='CHECKS_COMPLETE_PENDING_DURABLE_ACK',operation_id=ident,parent_epoch=epoch(),bank_format_binding=current,failed_child_reap=actual['reap_proof'],close_proof=dict(path=str(close_path),sha256=command['close_proof_sha256']),failed_job_id=reap['job_id'],failed_result_remains_unaccepted=True,automatic_replay=False,weight_bytes_changed=False,quality_admitted=False,performance_admitted=False)
    try:
        # Single command thread: no new child can run before durable ACK returns.
        manager.child_drain_proved=True;manager.phase='READY_OLD16' if manager.layout==16 else 'READY_PAIR139'
        manager._publish_field();clean_flag[0]=True
        manager._guard() # now circular drain/exclusion guards are rechecked in full
        need(manager.binding()==current,'revalidation changed bank epoch/format')
        event['phase']='OWNER_REVALIDATED_FAILED_RESULT_UNACCEPTED'
        manager.atomic(event_path,event) # file+directory fsync before successful command ACK
        event_sha=hashlib.sha256(event_path.read_bytes()).hexdigest()
        manager.revalidated_ids174=getattr(manager,'revalidated_ids174',set())|{ident}
    except BaseException:
        manager.phase,manager.child_drain_proved,manager.store.bank_runtime_admissible_v146,manager.store.bank_format_binding_v146,clean_flag[0]=old
        raise
    return dict(schema=175,close_scope=close_scope,native_close_claimed=native_close_claimed,phase=event['phase'],operation_id=ident,bank_format_binding=current,journal_path=str(event_path),journal_sha256=event_sha,failed_result_remains_unaccepted=True,automatic_replay=False)
