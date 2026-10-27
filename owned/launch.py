#!/usr/bin/env python3
"""Select an owned numerical provider without loading CUDA in this parent."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
import importlib.util
_numa_spec=importlib.util.spec_from_file_location('strata_owned_numa_policy',HERE/'common/numa_policy.py')
if _numa_spec is None or _numa_spec.loader is None: raise RuntimeError('NUMA source helper missing')
_numa=importlib.util.module_from_spec(_numa_spec);_numa_spec.loader.exec_module(_numa)

def plan(options):
    p=HERE/'profiles'/f'{options.profile}.json'
    if not p.is_file():raise ValueError('unknown source profile; no alternate backend is selected')
    profile=json.loads(p.read_text())
    if not isinstance(profile,dict) or 'gpus' not in profile:
        raise ValueError('profile metadata is not a launchable native profile')
    if profile.get('launch_enabled') is False:
        raise ValueError('source profile is not yet qualified for native launch; use its source build and qualification plan')
    devices=[int(x) for x in options.gpus.split(',')]
    if len(devices)!=profile['gpus'] or len(set(devices))!=len(devices) or min(devices)<0:
        raise ValueError('GPU selection must match the exact source profile count, without duplicates')
    if options.context not in profile['contexts']:raise ValueError('context is not admitted by this source profile')
    context=profile['contexts'][options.context]
    values=dict(pack=str(options.pack),common=str(options.common),mtp=str(options.mtp),expert_profile=str(options.expert_profile),context=str(context['tokens']),rope_scale=context['rope_scale'])
    argv=[str(options.build_dir/profile['binary'])]+[a.format_map(values)for a in profile['args']]
    env=dict(profile['env'])
    if options.shared_expert_arena:
        argv+=['--shared-expert-arena',str(options.shared_expert_arena)];env['STRATA_SHARED_ARENA_REUSE_V421']='1'
    if options.native_server:argv+=['--serve']
    numa_args=_numa.resolve_interleave_args(['--interleave=all']) if options.memory_policy=='interleave' else []
    return dict(numa_arguments=numa_args,profile=options.profile,family=profile['family'],current_gpu_indices=devices,context=context['tokens'],argv=argv,environment=env,native_p2p_required=True,legacy_fallback=False)

def validate_build(selected):
    binary=Path(selected['argv'][0])
    marker=binary.with_name('native-profile.json')
    if not binary.is_file() or not marker.is_file():raise ValueError('native source target and its build stamp are required')
    stamp=json.loads(marker.read_text())
    profile=json.loads((HERE/'profiles'/f"{selected['profile']}.json").read_text())
    if stamp.get('profile') not in profile['native_profiles']:raise ValueError('built numerical provider does not match the requested profile')
    with binary.open('rb')as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=stamp.get('binary_sha256'):raise ValueError('built numerical provider hash does not match its stamp')

def current_visibility(indices):
    # NVML's system indices need not be CUDA's PCI-ordered ordinals. Resolve afresh.
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
    by_index={int(a.strip()):b.strip()for a,b in (line.split(',',1)for line in rows.splitlines())}
    try:return ','.join(by_index[i]for i in indices)
    except KeyError as e:raise ValueError(f'current GPU index unavailable: {e.args[0]}')from e

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',default='swift-nvfp4-two')
    p.add_argument('--gpus',default='0,1',help='current NVIDIA system indices, resolved at each launch')
    p.add_argument('--context',required=True,choices=['256k','512k','1m'])
    p.add_argument('--pack',type=Path,required=True);p.add_argument('--common',type=Path,required=True)
    p.add_argument('--mtp',type=Path,required=True);p.add_argument('--expert-profile',type=Path,required=True)
    p.add_argument('--build-dir',type=Path,default=HERE.parent/'build-owned')
    p.add_argument('--shared-expert-arena',type=Path)
    p.add_argument('--memory-policy',choices=['interleave','inherit'],default='interleave',help='fresh online/allowed memory-node interleave; inherit only when an explicit parent policy is provided')
    p.add_argument('--native-server',action='store_true',help='native stdin protocol; this is not an HTTP listener')
    p.add_argument('--plan',action='store_true',help='print exact launch config without NVML/CUDA or model reads')
    return p

def main():
    options=parser().parse_args()
    try:
        selected=plan(options)
        if options.plan:print(json.dumps(selected,indent=2));return 0
        for item in [options.pack,options.common,options.mtp,options.expert_profile]:
            if not item.exists():raise ValueError(f'missing model input: {item}')
        validate_build(selected)
        snapshot=current_visibility(selected['current_gpu_indices'])
        # Inherited experimental STRATA knobs must not silently alter a selected profile.
        env={k:v for k,v in os.environ.items()if not k.startswith('STRATA_')}
        env.update(selected['environment']);env.update(CUDA_VISIBLE_DEVICES=snapshot,CUDA_DEVICE_ORDER='PCI_BUS_ID',STRATA_RUN_GPU_UUIDS=snapshot,STRATA_RUN_GPU_INDICES=','.join(map(str,selected['current_gpu_indices'])))
        print(json.dumps(dict(profile=selected['profile'],context=selected['context'],current_gpu_indices=selected['current_gpu_indices'],native_p2p_required=True)),flush=True)
        command=selected['argv']
        if selected['numa_arguments']:
            command=['/usr/bin/numactl',*_numa.resolve_interleave_args(['--interleave=all']),*command]
        os.execvpe(command[0],command,env)
    except (ValueError,OSError,subprocess.SubprocessError)as e:print(f'strata owned: {e}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
