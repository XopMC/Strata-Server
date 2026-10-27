"""Standalone bundle runtime, single GPU0, native vision/tools and DSpark.

Experimental launch is for local validation. Public production admission must
be backed by independent quality and filled-context evidence, not allocation.
"""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from contextlib import ExitStack
from dataclasses import dataclass
import base64
import copy
import importlib.util
import json
import math
import os
import hashlib
import sys
import subprocess
from pathlib import Path
import tempfile
import time
import urllib.request

from .api_codec import MODEL_ID,ByteLevelDecoder,NativeDeltaParser,native_request,encode_messages
from .fifo import RequestError,RequestCancelled

ROOT=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41/Strata'))
from index_selection_v512 import uuids as _index_uuids
UUID=_index_uuids()[0]


@dataclass(frozen=True)
class RuntimeSettings:
    model_path:Path
    context:int=262144
    tile_tokens:int=1024
    weight_cache_gib:int=14
    packed_ram_gib:int=512
    host_workspace_gib:int=96
    cpu_threads:int=8
    expert_dispatch:str='gpu-only'
    mtp:str='off'
    grouped_prefill:bool=True
    experimental:bool=False
    cpu_dense_threads:int=1
    persistent_cpu_workers:bool=False
    critical_decode_residency:bool=False
    packed_row_staging:bool=False
    native_launch_cache:bool=False
    hc_sinkhorn_mode:str='portable'
    expert_gemv_mode:str='native'
    selected_expert_batch:bool=False
    decode_weight_cache_gib:int|None=None
    kv_cache_type:str='bf16'
    own_rows_cuda:bool=False
    own_cpu_mode:str='numpy'
    cpu_batch_projections:bool=False
    cpu_wxa8:bool=False
    cpu_moe_graph_sync:str|None=None
    cpu_moe_graph_capture:bool=True
    cpu_moe_graph_wxa8:bool=False
    native_linear_graph:bool=False
    fork_fp8_gemv:bool=False
    head_gemv_splits:int|None=None


def physical_cpu_workers(settings):
    selected=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        identity=((topology/'physical_package_id').read_text().strip(),(topology/'core_id').read_text().strip())
        if identity not in seen:selected.append(cpu);seen.add(identity)
        if len(selected)==settings.cpu_threads:return selected
    raise ValueError('CPU workers exceed the physically distinct allowed cores')


def runtime_configuration(settings):
    """Exact tested runtime choices, including actual physical placement."""
    return {'context_capacity':settings.context,'tile_tokens':settings.tile_tokens,
        'weight_cache_gib':settings.weight_cache_gib,'packed_ram_gib':settings.packed_ram_gib,
        'host_workspace_gib':settings.host_workspace_gib,'cpu_threads':settings.cpu_threads,
        'cpu_dense_threads':settings.cpu_dense_threads,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'physical_cpu_workers':physical_cpu_workers(settings),'numa_policy':os.environ.get('DEEPSEEK41_NUMA_POLICY','unknown'),
        'omp_wait_policy':os.environ.get('OMP_WAIT_POLICY','default'),
        'gomp_spincount':os.environ.get('GOMP_SPINCOUNT','default'),'gpu_uuid':UUID,
        'expert_dispatch':settings.expert_dispatch,'mtp':settings.mtp,'grouped_prefill':settings.grouped_prefill,
        'persistent_cpu_workers':settings.persistent_cpu_workers,'critical_decode_residency':settings.critical_decode_residency,
        'packed_row_staging':settings.packed_row_staging,'native_launch_cache':settings.native_launch_cache,
        'hc_sinkhorn_mode':settings.hc_sinkhorn_mode,'expert_gemv_mode':settings.expert_gemv_mode,
        'selected_expert_batch':settings.selected_expert_batch,'decode_weight_cache_gib':settings.decode_weight_cache_gib,
        'kv_cache_type':settings.kv_cache_type,'kv_index_window_type':'bf16','own_rows_cuda':settings.own_rows_cuda,
        'own_cpu_mode':settings.own_cpu_mode,'cpu_batch_projections':settings.cpu_batch_projections,'cpu_wxa8':settings.cpu_wxa8,
        'cpu_moe_graph_sync':settings.cpu_moe_graph_sync,'cpu_moe_graph_capture':settings.cpu_moe_graph_capture,
        'cpu_moe_graph_wxa8':settings.cpu_moe_graph_wxa8,
        'native_linear_graph':settings.native_linear_graph,
        'fork_fp8_gemv':settings.fork_fp8_gemv,'head_gemv_splits':settings.head_gemv_splits,
        'complete_MoE_persistent_workers':settings.cpu_moe_graph_sync is not None}


def validate_settings(settings):
    if(type(settings.fork_fp8_gemv)is not bool or
       (settings.head_gemv_splits is not None and(type(settings.head_gemv_splits)is not int or settings.head_gemv_splits not in(1,4,8)))):
        raise ValueError('FP8 and head GEMV require explicit bounded runtime choices')
    if(type(settings.native_linear_graph)is not bool or
       (settings.native_linear_graph and(not settings.critical_decode_residency or settings.mtp=='batch'))):
        raise ValueError('Linear graphs require critical residency; batch MTP shape inventory is not yet admitted')
    if(settings.cpu_moe_graph_sync not in(None,'hostfunc','memops')or
       type(settings.cpu_moe_graph_capture)is not bool or type(settings.cpu_moe_graph_wxa8)is not bool or
       (settings.cpu_moe_graph_sync is None and(settings.cpu_moe_graph_wxa8 or not settings.cpu_moe_graph_capture))or
       (settings.cpu_moe_graph_sync is not None and(settings.expert_dispatch!='gpu-only'or
        not settings.critical_decode_residency or settings.selected_expert_batch or settings.cpu_wxa8 or
        settings.mtp=='batch'))):
        raise ValueError('Invalid complete CPU MoE graph configuration; batch MTP shape inventory is not yet admitted')
    if (type(settings.context) is not int or settings.context not in (4096,8192,16384,32768,65536,131072,262144,524288,1048576) or
        type(settings.tile_tokens) is not int or not 128<=settings.tile_tokens<=4096 or
        type(settings.weight_cache_gib) is not int or not 0<=settings.weight_cache_gib<=18 or
        (settings.decode_weight_cache_gib is not None and(type(settings.decode_weight_cache_gib)is not int or not 0<=settings.decode_weight_cache_gib<=18))or
        settings.kv_cache_type not in('bf16','q4_0','native-fp4')or
        settings.own_cpu_mode not in('numpy','pinned-simd','pinned-simd-v2')or
        (settings.own_cpu_mode!='numpy'and settings.expert_dispatch=='gpu-only')or
        type(settings.packed_ram_gib) is not int or not 1<=settings.packed_ram_gib<=640 or
        type(settings.host_workspace_gib) is not int or not 1<=settings.host_workspace_gib<=128 or
        settings.expert_dispatch not in ('gpu-only','cpu-routed','hybrid-cache') or settings.mtp not in ('off','sequential','batch') or
        settings.hc_sinkhorn_mode not in ('portable','native-tilelang','portable-graph') or settings.expert_gemv_mode not in ('native','experimental-serial') or
        type(settings.cpu_threads) is not int or not 1<=settings.cpu_threads<=256 or
        type(settings.cpu_dense_threads) is not int or not 1<=settings.cpu_dense_threads<=settings.cpu_threads or
        any(type(getattr(settings,key)) is not bool for key in ('grouped_prefill','experimental','persistent_cpu_workers',
            'critical_decode_residency','packed_row_staging','native_launch_cache','selected_expert_batch','own_rows_cuda','cpu_batch_projections','cpu_wxa8')) or
        (settings.cpu_wxa8 and not(settings.cpu_batch_projections and settings.persistent_cpu_workers))or
        (settings.cpu_batch_projections and(settings.expert_dispatch=='gpu-only'or not settings.persistent_cpu_workers))or
        (settings.selected_expert_batch and settings.expert_dispatch!='gpu-only')or
        (settings.persistent_cpu_workers and settings.expert_dispatch=='gpu-only')):
        raise ValueError('Invalid explicit runtime configuration')
    physical=set()
    for cpu in os.sched_getaffinity(0):
        topology=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        physical.add(((topology/'physical_package_id').read_text().strip(),(topology/'core_id').read_text().strip()))
    if settings.cpu_threads>len(physical):raise ValueError('CPU workers exceed the physically distinct allowed cores')


def admit_production(path,context,proof,configuration):
    if (not isinstance(configuration,dict) or configuration.get('context_capacity')!=context or configuration.get('gpu_uuid')!=UUID or
        proof.get('phase')!='independently_verified' or proof.get('quality_gate_passed') is not True or
        proof.get('standalone_source_hidden_restart_passed') is not True or context not in proof.get('validated_contexts',[]) or
        Path(proof.get('model_path','')).resolve()!=path or
        proof.get('model_bundle_sha256')!=hashlib.sha256((path/'model.bundle.json').read_bytes()).hexdigest() or
        configuration not in proof.get('validated_runtime_configurations',[])):
        raise ValueError('Production requires matching independent quality/standalone/filled-context proof')


def admit_single_gpu_owner():
    """Read only GPU0; never unload another application or query other GPUs."""
    result=subprocess.run(['nvidia-smi','-i',UUID,'--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],
        check=True,capture_output=True,text=True,timeout=15)
    for line in result.stdout.strip().splitlines():
        if not line.strip():continue
        fields=line.split(',')
        if len(fields)!=2 or fields[1].strip()!=UUID or int(fields[0])!=os.getpid():
            raise RuntimeError('GPU0 has another compute owner; no second model is loaded')


def resolve_images(messages,directory,*,image_limit=4,byte_limit=32*1024**2):
    messages=copy.deepcopy(messages);count=0
    for row in messages:
        if not isinstance(row.get('content'),list):continue
        for part in row['content']:
            if part.get('type')!='image_url':continue
            count+=1
            if count>image_limit:raise RequestError('Too many images in one request')
            url=part.get('image_url',{}).get('url')
            if not isinstance(url,str):raise RequestError('Expected an image URL')
            try:
                if url.startswith('data:image/'):
                    header,data=url.split(',',1)
                    if not header.endswith(';base64') or len(data)>byte_limit*4//3+8:raise ValueError('Image exceeds byte budget')
                    raw=base64.b64decode(data,validate=True)
                elif url.startswith(('http://','https://')):
                    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(byte_limit+1)
                else:raise ValueError('Images must be data: or HTTP(S) URLs')
                if len(raw)>byte_limit:raise ValueError('Image exceeds byte budget')
                path=Path(directory)/f'image-{count}.bin';path.write_bytes(raw)
                from PIL import Image
                with Image.open(path) as image:
                    if image.width*image.height>40_000_000:raise ValueError('Image pixel dimensions exceed budget')
                    image.verify()
                part['image_url']['url']=str(path)
            except Exception as error:raise RequestError('Cannot decode the requested image: '+str(error)) from error
    return messages


class NativeBundleRuntime:
    def __init__(self,settings):
        self.settings=settings;self.resources=ExitStack();self.reference=None
        validate_settings(settings)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(_index_uuids()):raise ValueError('Runtime needs the physical GPU0 UUID mask')
        status=json.loads((ROOT/'state/resident-lab-status.json').read_text()) if (ROOT/'state/resident-lab-status.json').exists() else {}
        other=status.get('pid')
        if other and other!=os.getpid() and Path('/proc',str(other)).exists() and status.get('phase') in ('constructing','packed_ram_copy','candidate_packed_ram_copy','ready','running_case'):
            raise RuntimeError('The project resident owner is still active; release it before loading another engine')
        path=Path(settings.model_path).resolve()
        self.manifest=json.loads((path/'model.bundle.json').read_text())
        self.bundle_sha256=hashlib.sha256((path/'model.bundle.json').read_bytes()).hexdigest()
        if not settings.experimental:
            proof=json.loads((ROOT/'state/phase1-final.json').read_text())
            admit_production(path,settings.context,proof,runtime_configuration(settings))
        admit_single_gpu_owner()
        import torch
        from .streamed_reference import NativeReference
        from .context_capacity import experimental_capacity
        self.torch=torch;torch.set_num_threads(settings.cpu_dense_threads);torch.backends.cuda.matmul.allow_tf32=False
        try:
            self.reference=NativeReference(ROOT,'cuda:0',4096,4096,'native-cuda','sm89',row_decode_mode='cuda',
                weight_cache_bytes=settings.weight_cache_gib*1024**3,bundle_path=path)
            self.ram_proof=self.reference.store.preload_packed_ram(budget_bytes=settings.packed_ram_gib*1024**3,reserve_bytes=80*1024**3,
                on_progress=lambda report:print(json.dumps({'loading':report}),flush=True))
            self.ram_proof.update(checkpoint_kind='own_mixed_candidate',model_bundle_sha256=self.bundle_sha256)
            self.config=json.loads((path/'runtime-config.json').read_text())
            self.capacity=self.resources.enter_context(experimental_capacity(self.reference,self.config,settings.context,
                weight_cache_bytes=settings.weight_cache_gib*1024**3,kv_cache_type=settings.kv_cache_type))
        except BaseException:self.close();raise
        print(json.dumps({'phase':'bundle_runtime_ready','configuration':self.configuration()}),flush=True)

    def configuration(self):
        return {**runtime_configuration(self.settings),'model':MODEL_ID,'weight_profile':self.manifest['recipe']['profile'],
            'model_bundle_sha256':self.bundle_sha256,'model_path':str(Path(self.settings.model_path).resolve()),
            'experimental':self.settings.experimental,'api_key_required':False,'packed_ram_bytes':self.ram_proof['resident_bytes']}

    def _sample(self,logits,request,generated):
        torch=self.torch;values=logits.reshape(-1,logits.shape[-1])[-1].float().clone()
        if not bool(torch.isfinite(values).all()):raise RuntimeError('Model generated nonfinite logits')
        if request['repetition_penalty']!=1 and generated:
            ids=torch.tensor(sorted(set(generated)),device=values.device)
            original=values[ids];penalty=request['repetition_penalty']
            values[ids]=torch.where(original<0,original*penalty,original/penalty)
        if request['temperature']==0:return int(values.argmax())
        values/=request['temperature']
        if request['top_k']:
            cutoff=values.topk(min(request['top_k'],len(values))).values[-1];values.masked_fill_(values<cutoff,-torch.inf)
        probabilities=values.softmax(-1)
        if request['top_p']<1:
            sorted_probs,ids=probabilities.sort(descending=True);remove=sorted_probs.cumsum(0)-sorted_probs>request['top_p']
            sorted_probs.masked_fill_(remove,0);probabilities.zero_().scatter_(0,ids,sorted_probs)
        return int(torch.multinomial(probabilities,1))

    def run(self,body,cancel,emit):
        try:request=native_request(body,context_capacity=self.settings.context)
        except (ValueError,TypeError,KeyError) as error:raise RequestError(str(error)) from error
        from .chunked_reference import ChunkedReference
        from .layerwise_reference import LayerwisePrefill
        from .sm89_index import score_index_with_small_fallback
        from .dense_residency import dense_native_cache
        from .cpu_experts import cpu_expert_dispatch
        from .grouped_experts import grouped_prefill
        from .speculative import GreedySpeculator
        from .weights import available_memory_bytes
        torch=self.torch;ref=self.reference;settings=self.settings
        mtp=settings.mtp!='off' and request['temperature']==0 and request['repetition_penalty']==1
        emitted=[];decoder=ByteLevelDecoder(ref.tokenizer);parser=NativeDeltaParser(request['thinking_mode'])
        first_token_at=None;last_token_at=None;started=time.perf_counter()
        def output(tokens):
            nonlocal first_token_at,last_token_at
            for token in tokens:
                if first_token_at is None:first_token_at=time.perf_counter()
                last_token_at=time.perf_counter()
                emitted.append(token)
                for delta in parser.feed(decoder.token(token)):
                    if not emit({'event':'delta','delta':delta}):cancel.set()
        def progress(row):
            if cancel.is_set():raise RequestCancelled('Client cancelled the request')
            if row['layer_complete']:print(json.dumps(row),flush=True)
        # Every exception restores request adapters before the next FIFO job.
        with ExitStack() as lifetime:
            lifetime.callback(ref.stop_chunked_sequence)
            images_dir=lifetime.enter_context(tempfile.TemporaryDirectory(prefix='deepseek41-image-'))
            dispatches=lifetime.enter_context(ExitStack())
            if cancel.is_set():raise RequestCancelled('Client cancelled before prefill')
            request['messages']=resolve_images(request['messages'],images_dir)
            try:prompt,media=encode_messages(request['messages'],thinking_mode=request['thinking_mode'],
                reasoning_effort=request['reasoning_effort'],return_multi_modal_data=True)
            except (ValueError,AssertionError,KeyError,TypeError) as error:raise RequestError(str(error)) from error
            images=types=None
            if media['images']:
                spec=importlib.util.spec_from_file_location('deepseek41_runtime_images',ROOT/'reference/inference/image_processor.py')
                processor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=processor;spec.loader.exec_module(processor)
                ids,type_ids,image_inputs=processor.prepare_vl_inputs(prompt,media['images'],ref.tokenizer,ref.graph.ModelArgs(**self.config))
                images=[image_inputs];types=torch.tensor([type_ids],dtype=torch.long)
            else:ids=ref.tokenizer.encode(prompt,add_special_tokens=False)
            if not ids or len(ids)+request['max_new_tokens']>settings.context:raise RequestError('Prompt plus requested output exceeds the configured context')
            if not emit({'event':'usage','prompt_tokens':len(ids)}):raise RequestCancelled('Client disconnected before prefill')
            ref.stop_chunked_sequence()
            ref.chunked=ChunkedReference(ref.graph,ref.model,head_tile=32,score_function=score_index_with_small_fallback,experimental_long_context=True)
            schedule=LayerwisePrefill(ref.chunked,tile_sizes=(settings.tile_tokens,),host_budget_bytes=settings.host_workspace_gib*1024**3,
                collect_mtp_tail=mtp,on_progress=progress)
            needed=schedule.plan(len(ids),False)['total_workspace_bytes']
            if available_memory_bytes()<needed+80*1024**3:raise RequestError('Insufficient available RAM for this prompt workspace')
            dispatches.enter_context(dense_native_cache(ref))
            if settings.packed_row_staging:
                from .row_staging import pinned_row_staging
                dispatches.enter_context(pinned_row_staging(ref))
            if settings.own_rows_cuda:
                from .own_quant_rows import cuda_own_rows
                dispatches.enter_context(cuda_own_rows(ref))
            if settings.native_launch_cache:
                from .native_factory_cache import native_launch_cache
                dispatches.enter_context(native_launch_cache(ref))
            if settings.hc_sinkhorn_mode=='portable-graph':
                from .hc_graph import experimental_portable_hc_graph
                dispatches.enter_context(experimental_portable_hc_graph(ref))
            elif settings.hc_sinkhorn_mode=='native-tilelang':
                from .hc_native import experimental_native_hc
                dispatches.enter_context(experimental_native_hc(ref))
            if settings.expert_gemv_mode!='native':
                from .expert_gemv import experimental_expert_gemv
                dispatches.enter_context(experimental_expert_gemv(ref))
            if settings.expert_dispatch!='gpu-only':dispatches.enter_context(cpu_expert_dispatch(ref,mode=settings.expert_dispatch,
                threads=settings.cpu_threads,max_rows=8,worker_cpus=physical_cpu_workers(settings),
                persistent_workers=settings.persistent_cpu_workers,own_uniform_mode=settings.own_cpu_mode,
                batch_projections=settings.cpu_batch_projections,cpu_wxa8=settings.cpu_wxa8))
            if settings.grouped_prefill:dispatches.enter_context(grouped_prefill(ref,minimum_tokens=512))
            if settings.selected_expert_batch:
                from .selected_experts import selected_decode
                dispatches.enter_context(selected_decode(ref))
            fork_statistics=head_statistics=None
            if settings.fork_fp8_gemv:
                from .fork_fp8_gemv import fork_fp8_decode
                fork_statistics=dispatches.enter_context(fork_fp8_decode(ref))
            if settings.head_gemv_splits is not None:
                from .head_bf16_gemv import head_decode_gemv
                head_statistics=dispatches.enter_context(head_decode_gemv(ref,splits=settings.head_gemv_splits))
            import tvm_ffi
            dispatches.enter_context(tvm_ffi.use_torch_stream())
            with torch.inference_mode(),torch.device(ref.device):
                torch.cuda.synchronize(ref.device);prefill_started=time.perf_counter()
                _,logits,main=schedule.run(torch.tensor([ids],dtype=torch.long),images=images,token_types=types)
                torch.cuda.synchronize(ref.device);prefill_ended=time.perf_counter()
                if settings.decode_weight_cache_gib is not None:
                    from .decode_budget import decode_weight_budget
                    dispatches.enter_context(decode_weight_budget(ref,settings.decode_weight_cache_gib))
                if settings.critical_decode_residency:
                    from .critical_residency import critical_decode_cache
                    dispatches.enter_context(critical_decode_cache(ref))
                linear_graph_statistics=None
                if settings.native_linear_graph:
                    from .linear_graph import native_linear_graphs
                    linear_graph_statistics=dispatches.enter_context(native_linear_graphs(ref))
                moe_graph_statistics=None
                if settings.cpu_moe_graph_sync is not None:
                    from .cpu_moe_graph import cpu_moe_graph_dispatch
                    moe_graph_statistics=dispatches.enter_context(cpu_moe_graph_dispatch(ref,physical_cpu_workers(settings),
                        a8=settings.cpu_moe_graph_wxa8,capture=settings.cpu_moe_graph_capture,sync_mode=settings.cpu_moe_graph_sync))
                pending=self._sample(logits,request,emitted);mtp_statistics=None
                if mtp and len(ids)>=2:
                    driver=GreedySpeculator(ref.chunked,eos_token_id=ref.tokenizer.eos_token_id,verification=settings.mtp)
                    driver.seed(main,schedule.mtp_tail_start)
                    _,mtp_statistics=driver.generate(pending,request['max_new_tokens'],on_tokens=output,cancel=cancel)
                else:
                    output([pending])
                    while len(emitted)<request['max_new_tokens'] and emitted[-1]!=ref.tokenizer.eos_token_id and not cancel.is_set():
                        _,logits,_=ref.forward(torch.tensor([[emitted[-1]]],dtype=torch.long),ref.chunked.position)
                        output([self._sample(logits,request,emitted)])
            tail=decoder.finish()
            if tail:
                for delta in parser.feed(tail):emit({'event':'delta','delta':delta})
            message,reason=parser.finish(bool(emitted and emitted[-1]==ref.tokenizer.eos_token_id))
            result={'message':message,'finish_reason':reason,'usage':{'prompt_tokens':len(ids),'completion_tokens':len(emitted),
                'total_tokens':len(ids)+len(emitted)},'timings':{'ttft_seconds':first_token_at-started,
                'prefill_seconds':prefill_ended-prefill_started,'new_prompt_tps':len(ids)/(prefill_ended-prefill_started),
                'generation_tps':max(0,len(emitted)-1)/max(last_token_at-first_token_at,1e-30),
                'prefix_cache_reused':False},'mtp':mtp_statistics,'cancelled':cancel.is_set(),
                'runtime_configuration':runtime_configuration(settings),
                'complete_MoE_statistics':moe_graph_statistics()if moe_graph_statistics is not None else None,
                'linear_graph_statistics':linear_graph_statistics,
                'fork_fp8_statistics':fork_statistics,'head_gemv_statistics':head_statistics,
                'checkpoint':{'model_bundle_sha256':self.bundle_sha256,'weight_profile':self.manifest['recipe']['profile']}}
            emit({'event':'result','result':result})
            print(json.dumps({'request_complete':result['usage'],'timings':result['timings'],'mtp':mtp_statistics}),flush=True)

    def close(self):
        self.resources.close()
        if self.reference is not None:self.reference.close();self.reference=None
