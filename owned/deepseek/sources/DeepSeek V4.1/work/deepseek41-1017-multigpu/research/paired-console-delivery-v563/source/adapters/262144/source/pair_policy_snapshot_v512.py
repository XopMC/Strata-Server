"""Immutable N1 pair policy; pure Python, no CUDA/Torch/model import."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import threading
from pair_runtime_policy_v512 import environment_for, runtime_record
from stage_devices_pair_v512 import validate_visible_pair


def settings_key(s):
    return (s.context, s.mtp, s.pair_cache_mode, s.expert_cache_gib, s.workers, s.coordinator_cpu, s.expert_coordinator_cpu)


@dataclass(frozen=True)
class PolicySnapshot:
    environment: tuple
    settings: tuple
    pid: int
    _lock: object = field(default_factory=threading.Lock, compare=False, repr=False)

    @classmethod
    def capture(cls, settings):
        from cpu_projection_binding_v512 import resolve_cpu_projection
        resolve_cpu_projection()
        from prefill_seam_binding_v512 import admission, FLAGS
        admission()
        if (settings.context,settings.chunk_tokens,settings.prefill_batch,settings.expert_batch)!=(262144,131072,2048,4096):raise RuntimeError('328 strict512 geometry')
        if os.environ.get('DS1017_PREFILL_PASS_SEAM_V317')!='1':raise RuntimeError('317 flag must be latched before Engine')
        expected = dict(environment_for(settings))
        expected['DS1017_PAIR_SERVING_V516']='1'
        expected.update(FLAGS)
        expected['DS1017_CPU_GROUP_SUM_V2']='0'
        if os.environ.get('DS1017_CPU_NODE_LOCAL_QUEUE_V80')!='0':raise RuntimeError('CPU104 queue must be literal0 before pool construction')
        expected['DS1017_CPU_NODE_LOCAL_QUEUE_V80']='0'
        for key in ('DS1017_PAIR_PREFILL_V1', 'DS41_PREFILL_GU_HIDDEN_V41', 'DS1017_PAIR_STAGE_COPY_V4', 'DS1017_PAIR_PREFILL_LOOKAHEAD_V5'):
            value = os.environ.get(key, '0')
            if value != '0':raise ValueError('K14 indexer requires legacy EP/stage/lookahead/GU off: ' + key)
            expected[key] = value
        if os.environ.get('DS1017_PAIR_INDEXER_PREFILL_V1') != '0':
            raise ValueError('325 local K14 requires explicit old helper OFF=0')
        expected['DS1017_PAIR_INDEXER_PREFILL_V1'] = '0'
        if os.environ.get('DS41_INDEXER_S8_R1') != '1':
            raise ValueError('V10 K14 requires explicit selected DS41_INDEXER_S8_R1=1')
        expected['DS41_INDEXER_S8_R1'] = '1'
        if os.environ.get('DS41_PF_PROFILE') != '1':
            raise ValueError('V10 diagnostic requires explicit DS41_PF_PROFILE=1, same as V8 baseline')
        expected['DS41_PF_PROFILE'] = '1'
        for key,value in {'DS1017_GPU_GU_OVERLAP_V6':'0','DS1017_GPU_GU_RESIDENT_OWNER_V19':'1','DS1017_GPU_GU_BYTEPERM_V13':'1','DS1017_GPU_GU_BUDGET_GIB_V19':'20'}.items():
            if os.environ.get(key)!=value:raise ValueError('Explicit exact GU profile required: '+key)
            expected[key]=value
        for key,value in {'DS1017_CPU_PAIR32_LAYOUT_V143':'139','DS1017_PAIR32_REGISTRY_V145':'139','DS1017_PAIR32_BYTE_INVERSE_V141':'1','DS1017_GPU_GU_PAIR32_OWNER_V147':'0','DS1017_GPU_GU_STARTUP_SCRATCH_BYTES_V147':'0','DS1017_COLD_FOOTER_MAX_V108':'0'}.items():
            if os.environ.get(key)!=value:raise RuntimeError('Explicit paired139 policy differs: '+key)
            expected[key]=value
        result = cls(tuple(sorted(expected.items())), settings_key(settings), os.getpid())
        result.assert_current(settings)
        return result

    @property
    def context(self): return self.settings[0]
    @property
    def mtp(self): return self.settings[1]
    @property
    def mode(self): return self.settings[2]
    @property
    def gib(self): return self.settings[3]

    def assert_current(self, settings=None):
        if self.pid != os.getpid():
            raise RuntimeError('Pair policy must be created in the fresh GPU child, not inherited from parent')
        validate_visible_pair()
        if os.environ.get('DS41_NATIVE_EXPERT_DEVICE') is not None:
            raise ValueError('Pair mode refuses ambiguous legacy singular expert-device setting')
        for key, expected in self.environment:
            actual = os.environ.get(key, '0') if key in ('DS1017_PAIR_PREFILL_V1', 'DS41_PREFILL_GU_HIDDEN_V41', 'DS1017_PAIR_STAGE_COPY_V4', 'DS1017_PAIR_PREFILL_LOOKAHEAD_V5') else os.environ.get(key)
            if actual != expected:
                raise RuntimeError('Immutable pair policy changed: ' + key)
        if settings is not None and settings_key(settings) != self.settings:
            raise RuntimeError('Pair settings differ from constructor policy')
        return self

    def record(self, settings):
        self.assert_current(settings)
        # Return fresh metadata, not a mutable dictionary retained as policy.
        result = runtime_record(settings)
        result['policy_snapshot_pid'] = self.pid
        result['policy_environment'] = dict(self.environment)
        result['pair_stage_copy_v4'] = dict(self.environment)['DS1017_PAIR_STAGE_COPY_V4']
        result['pair_prefill_lookahead_v5'] = dict(self.environment)['DS1017_PAIR_PREFILL_LOOKAHEAD_V5']
        result['pair_indexer_prefill_v1'] = dict(self.environment)['DS1017_PAIR_INDEXER_PREFILL_V1']
        from cpu_projection_binding_v512 import resolve_cpu_projection
        result['CPU_projection']=resolve_cpu_projection().record()
        result['CPU_group_sum']='0'
        from adaptive_binding_v512 import record as adaptive_record
        result['GPU1_whole_adaptive370']=adaptive_record()
        result['GPU1_GU147']=dict(enabled=False,native_zero_and_no_owner_receipt_required=True)
        from prefill_seam_binding_v512 import record
        result['prefill_wavefront_v325']=record()
        result['performance_eligible']=False
        result['model_candidate_only']=True
        from index_native_binding_v512 import record as index501_record
        result['index_native_admission501']=index501_record()
        result['profiler_enabled'] = True
        result['profiler_environment'] = {'DS41_PF_PROFILE': '1'}
        result['profiler_scope'] = 'GPU0 legacy phase timeline only; both-rank elapsed NEWPP includes startup/transport/final join/GU build; timeline sums are not wall time'
        result['policy_rechecked'] = True
        return result

    @contextmanager
    def request_scope(self, settings):
        self.assert_current(settings)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Pair runtime refuses concurrent/reentrant requests')
        try:
            self.assert_current(settings)
            yield self
            self.assert_current(settings)
        finally:
            self._lock.release()
