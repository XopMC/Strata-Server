from index_selection_v506 import uuids as _index_uuids
from pathlib import Path
"""Literal selected Engine/Hash/Options wrappers; new authenticated hash source only."""
import ctypes as C
import os
import numpy as np
from native_cpp_adapter import P,S,I,ROOT
class Hash(C.Structure):
    _fields_=[('token_map',P),('token_count',S),('multipliers',P),('multiplier_count',S),('primes',P),('offsets',P),('modulus_count',S),('pad',C.c_int64)]
class Options(C.Structure):
    _fields_=[('context',I),('cpus',P),('cpu_count',S),('task_library',C.c_char_p),('projection_library',C.c_char_p),('a8',I),('graph',I)]

class NativeCppEngine:
    def __init__(self,checkpoint,library,*,cpus,context=4096,a8=False,graph=False,projection_revision=4,pair_policy=None):
        if pair_policy is None:raise ValueError("Pair Engine requires immutable constructor policy")
        pair_policy.assert_current()
        self.pair_policy=pair_policy
        import sys
        from types import SimpleNamespace
        from transformers import AutoTokenizer
        engram=checkpoint.store.engram_module()
        self.handle=None;self.owner=checkpoint;self.lib=C.CDLL(str(library))
        self.lib.ds41_native_engine_error.argtypes=[];self.lib.ds41_native_engine_error.restype=C.c_char_p
        self.lib.ds41_native_engine_create.argtypes=[P,C.POINTER(Hash),C.POINTER(Options)];self.lib.ds41_native_engine_create.restype=P
        self.lib.ds41_native_engine_step.argtypes=[P,I,I,P,P,S,P,S,P,S,P,S];self.lib.ds41_native_engine_step.restype=I
        self.lib.ds41_native_engine_destroy.argtypes=[P];self.lib.ds41_native_engine_destroy.restype=None
        self.lib.ds41_native_engine_worker_rows.argtypes=[P,P,S];self.lib.ds41_native_engine_worker_rows.restype=I
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint.source,local_files_only=True)
        token_map,vocab=engram.build_compressed_token_map(self.tokenizer)
        if vocab!=checkpoint.config['engram_compressed_vocab_size']:raise ValueError('Native compressed tokenizer vocabulary differs')
        args=SimpleNamespace(**checkpoint.config);layout=engram.EngramLayout.from_args(args)
        token_map=np.array(token_map,np.int32);mult=np.array(engram.compute_hash_multipliers(layout.layer_ids,layout.max_ngram_size,vocab),np.int64)
        primes=np.array(layout.primes,np.int64).reshape(2,24);offsets=np.concatenate([np.zeros((2,1),np.int64),primes[:,:-1].cumsum(1)],axis=1)
        h=Hash(token_map.ctypes.data,token_map.size,mult.ctypes.data,mult.size,primes.ctypes.data,offsets.ctypes.data,primes.size,int(token_map[args.engram_pad_id]))
        from adaptive_binding_v506 import check as adaptive_check
        adaptive=adaptive_check();cpus=np.array(cpus,np.int32);tasks=adaptive['task']['path'].encode()
        self.cpus=cpus
        if type(projection_revision)is not int or projection_revision not in(1,2,4)or(a8 and projection_revision!=1):raise ValueError('Explicit native validation implementation required')
        from cpu_projection_binding_v506 import resolve_cpu_projection
        projection=resolve_cpu_projection()
        if not a8:raise ValueError('CPU96 retains only selected original A8 arithmetic')
        pool=str(projection.library).encode()
        options=Options(context,cpus.ctypes.data,cpus.size,tasks,pool,a8,graph)
        self.handle=self.lib.ds41_native_engine_create(checkpoint.checkpoint,C.byref(h),C.byref(options))
        if not self.handle:raise RuntimeError(self.lib.ds41_native_engine_error().decode())
    def step(self,token,position,*,probes=False):
        if not self.handle:raise ValueError('Native engine is closed')
        next_token=C.c_int();logits=np.empty(129280,np.float32);times=np.empty(4,np.float64)
        hidden=np.empty((40,4,5120),np.uint16)if probes else None
        routes=np.empty((40,6),np.int32)if probes else None
        rc=self.lib.ds41_native_engine_step(self.handle,token,position,C.byref(next_token),logits.ctypes.data,logits.size,
            hidden.ctypes.data if probes else None,hidden.size if probes else 0,routes.ctypes.data if probes else None,routes.size if probes else 0,times.ctypes.data,times.size)
        if rc:raise RuntimeError(self.lib.ds41_native_engine_error().decode())
        if not np.isfinite(logits).all():raise ValueError('Native engine returned nonfinite logits')
        return next_token.value,logits,hidden,routes,times

    def bulk418_info(self,*,completed=False):
        self.pair_policy.assert_current()
        from bulk_counters_v506 import read
        return read(self.lib,completed=completed)

    def active_epoch427_info(self,*,completed=False):
        self.pair_policy.assert_current()
        from epoch_counters_v506 import read
        return read(completed=completed)

    def configure_gu_v19(self):
        self.pair_policy.assert_current()
        fn=self.lib.ds41_native_engine_gu_cache_v19;fn.argtypes=[P,C.c_uint64];fn.restype=I
        if fn(self.handle,20<<30):raise RuntimeError(self.lib.ds41_native_engine_error().decode())
    def gu_statistics_v19(self):
        self.pair_policy.assert_current()
        values=(C.c_uint64*32)();fn=self.lib.ds41_native_engine_gu_statistics_v19;fn.argtypes=[P,P,I];fn.restype=I
        if fn(self.handle,values,32):raise RuntimeError(self.lib.ds41_native_engine_error().decode())
        return dict(zip(['version', 'resident_entries', 'requested_device_bytes', 'weight_device_bytes', 'upload_H2D_bytes', 'contains_hits', 'contains_misses', 'contains_invalid', 'begin_calls', 'submitted_commands', 'released_views', 'aborted_commands', 'failures', 'live_tickets', 'borrowed_views', 'command_H2D_bytes', 'result_D2H_bytes', 'submitted_groups', 'submitted_route_pairs', 'count1_groups', 'count2_groups', 'count6_groups', 'issuer_rebinds', 'closed', 'budget_bytes', 'staging_pinned_bytes', 'request_pinned_bytes', 'result_pinned_bytes', 'scratch_sentinel_bytes', 'upload_copies', 'VRAM_free_before', 'VRAM_free_after'],map(int,values)))

    def gu_upload_stats147(self,original_gu):
        # The151free CABI publishes only ONE live typed147 owner; original32
        # counters remain separate. Read while this Engine still owns its cache.
        if not self.handle:raise ValueError('Native Engine closed before147 receipt')
        self.pair_policy.assert_current()
        values=(C.c_uint64*24)();fn=self.lib.ds151_last_upload_stats147;fn.argtypes=[P,I];fn.restype=I
        if fn(values,24):raise RuntimeError('No unique live151 paired147 upload receipt')
        v=tuple(map(int,values));b=self.owner.bank_format;n=original_gu['resident_entries'];matrix=6635520
        if (v[:2]!=(147,139) or v[2]!=b['owner_generation']or v[3]!=b['format_epoch']or v[4]!=15360 or n!=1618 or v[5]!=n or v[6]!=(20<<30) or v[7]!=matrix or v[8]!=v[6]+v[7] or v[8]>(22<<30) or v[9]!=original_gu['requested_device_bytes']or v[9]>v[6] or v[10]!=v[9]+matrix or v[11]!=matrix or v[12]!=0 or v[13]!=2*n or v[14]!=v[13] or v[15]!=v[13]*matrix or v[16]!=v[15] or v[17]!=v[15] or v[18]!=n*2*(matrix//288)*16 or v[19]!=1 or v[20]or v[21]or v[22]or v[23]!=32 or original_gu['upload_H2D_bytes']!=v[17]+n*8):
            raise RuntimeError('Actual151/147 paired139 GPU1 initialization differs')
        keys=('schema','layout_tag','owner_generation','bank_format_epoch','validated_descriptors','resident_entries','cache_budget_bytes','startup_scratch_budget_bytes','declared_peak_budget_bytes','final_requested_device_bytes','peak_requested_device_bytes','startup_scratch_peak_bytes','startup_scratch_live_bytes','inverse_submitted','inverse_completed','inverse_input_bytes','inverse_output_bytes','paired_H2D_bytes','validated_footer_words','init_complete','closed','failures','last_inverse_status','original_stat_words')
        return dict(zip(keys,v))

    def require_no_gu147(self):
        values=(C.c_uint64*24)();fn=self.lib.ds151_last_upload_stats147;fn.argtypes=[P,I];fn.restype=I
        if any(self.gu_statistics_v19().values()) or fn(values,24)!=2:
            raise RuntimeError('387 requires actual legacyGU32 allzero and no147owner rc2')

    def pair_indexer_counters(self):
        if not self.handle:raise ValueError('Native engine is closed')
        from pair_indexer_counters_v506 import read_counters
        return read_counters(self.lib,self.pair_policy)

    def pair_gpu_cache_stats(self):
        if not self.handle:raise ValueError('Native engine is closed')
        global_values=(C.c_uint64*5)();device_values=(C.c_uint64*16)()
        fn=self.lib.ds41_native_engine_expert_cache_stats;fn.argtypes=[P,P,S];fn.restype=I
        if fn(self.handle,global_values,5):raise RuntimeError(self.lib.ds41_native_engine_error().decode())
        fn=self.lib.ds41_native_engine_expert_device_stats;fn.argtypes=[P,P,S];fn.restype=I
        if fn(self.handle,device_values,16):raise RuntimeError(self.lib.ds41_native_engine_error().decode())
        g=list(map(int,global_values));raw=list(map(int,device_values));by_device={}
        expected={'0':_index_uuids()[0],'1':_index_uuids()[1]}
        for slot,uuid in expected.items():
            i=int(slot);row=raw[4*i:4*i+4]
            if any(row) and row[0] != i:raise RuntimeError('Native cache stats returned an unexpected logical device index')
            by_device[slot]={'uuid':uuid,'hits':row[1],'arena_used_bytes':row[2],'arena_budget_bytes':row[3]}
        if sum(v['arena_used_bytes'] for v in by_device.values())!=g[3] or sum(v['arena_budget_bytes'] for v in by_device.values())!=g[4]:
            raise RuntimeError('Per-device expert arenas do not reconcile with native cache totals')
        self.pair_policy.assert_current()
        mode=self.pair_policy.mode;mtp=self.pair_policy.mtp
        if mtp=='native4' and (by_device['0']['arena_budget_bytes'] or by_device['0']['arena_used_bytes']):
            raise RuntimeError('GPU0 native4 MTP must retain the full expert-cache arena')
        if mode=='gpu1' and (by_device['0']['arena_budget_bytes'] or by_device['0']['arena_used_bytes']):
            raise RuntimeError('GPU1-only cache unexpectedly allocated on GPU0')
        if mode=='gpu1' and self.pair_policy.gib>0 and not by_device['1']['arena_budget_bytes']:
            raise RuntimeError('Requested GPU1 cache has no admitted native device arena')
        return {'schema':'ds1017-pair-device-cache-stats/1','requested_mode':mode,
                'requested_gib_per_selected_device':self.pair_policy.gib,
                'global':dict(zip(('hits','misses','entries','used_bytes','budget_bytes'),g)),
                'devices':by_device,'cache_math_quality':'EXPERIMENTAL_UNACCEPTED_GPU_EXPERT_GEMV'}

    def close(self):
        if self.handle:
            control=getattr(self,'_adaptive_control',None)
            if control is not None:self._adaptive_closed_receipt=control.close()
            self.lib.ds41_native_engine_destroy(self.handle);self.handle=None
    def worker_rows(self):
        if not self.handle:raise ValueError('Native engine is closed')
        out=np.empty(len(self.cpus),np.uint64)
        if self.lib.ds41_native_engine_worker_rows(self.handle,out.ctypes.data,out.size):raise RuntimeError(self.lib.ds41_native_engine_error().decode())
        return out.tolist()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
