"""Pinned actual334R4 component/build lineage. Stdlib metadata only."""

def _ds_publication_source(relative):
 from pathlib import Path as _PublicationPath
 return str(_PublicationPath(__file__).resolve().parents[9] / relative)

def _ds_publication_model_dir():
 import os as _publication_os
 return _publication_os.environ["STRATA_MODEL_DIR"]

from pathlib import Path
import hashlib,json,os
P=Path(__file__).resolve().parents[1];M=Path(_ds_publication_source('DeepSeek V4.1/work/deepseek41-1017-multigpu'))
PHASE='ROOT344_ONE524_LINK_K14_REGISTER_ARCHIVE_PASS_MODEL_PENDING'
def need(v,m):
 if not v:raise RuntimeError('350 K14 binding: '+m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(row):
 p=Path(row['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and digest(p)==row['sha256'],'actual pin '+str(p));return p
def check(require_ready=True):
 v=json.loads((P/'K14_BINDING.json').read_text());need(v.get('schema')==408 and(not require_ready or v.get('ready')is True),'Root binding pending')
 need(v.get('flag')=={'DS1017_K14_PACKED_REGISTER_V334':'1'}and v.get('context')==524288,'explicit334ON/only512')
 for name in('archive','source_pins','kernel_build','offline_resource','recipe'):pin(v[name])
 need(v['archive']['path']==str(M/'build/k14-packed-register-score-v334-r4/libds41_k14_indexer_prefill.a'),'actual qualified R4 archive only')
 kb=json.loads(Path(v['kernel_build']['path']).read_text());need(kb.get('phase')=='ROOT334_R4_ALL8_BUILD_STEPS_PASS_RESOURCES_AND_NATIVE_PENDING'and len(kb.get('steps',[]))==8 and all(x['returncode']==0 for x in kb['steps'])and kb.get('source_pins_sha256')==v['source_pins']['sha256'],'actual R4 eight steps')
 need(kb['artifacts'][v['archive']['path']]['sha256']==v['archive']['sha256'],'actual built qualified archive')
 resource=json.loads(Path(v['offline_resource']['path']).read_text());need((resource.get('registers'),resource.get('stack'),resource.get('local'),resource.get('LDL_STL_count'))==(63,0,0,0),'actual R4 offline resource/no-spill proof')
 need(len(v['component_proofs'])==2,'both actual device proofs')
 for device,row in enumerate(v['component_proofs']):
  a=json.loads(pin(row).read_text());need(a.get('phase')=='BOUNDED_ACTUAL_K14_PACKED_REGISTER334_SCORE_TOPK_PASS_NOT_MODEL'and a.get('native_rc')==0 and a.get('device')==device,'actual requestedGPU native PASS')
  need((a.get('registers'),a.get('local_bytes'),a.get('active_CTAs'),a.get('shared_bytes'))==(63,0,2,49280),'actual bothGPU resources')
  need((a.get('compared_live_BF16_scores'),a.get('compared_actual_top512_words'),a.get('actual_public_cases'),a.get('changed_graph_replays'))==(109006180,1984512,14,33),'full live score/top512/changed graph proof')
  need(a.get('candidate')==str(M/'build/k14-packed-register-score-v334-r4/candidate.so')and kb['artifacts'][a['candidate']]['sha256']==digest(a['candidate']),'same actual R4 candidate code')
  need(a.get('score_holes_checked')is True and a.get('input_readonly_and_canaries')is True and a.get('model_executed')is False,'proof scope and holes/canaries')
 if require_ready:
  from composition_binding_v506 import check as check_composed
  c=check_composed();pin(v['DSO']);pin(v['build'])
  need(v['DSO']==c['library']and v['build']==c['build'],'actual349 composition selects qualifiedarchive unchanged')
 return v
def record():
 v=check();return dict(source_variant='334R4_K14_unchanged_in375_over349',flag=dict(v['flag']),actual_archive=v['archive'],actual_kernel_build=v['kernel_build'],actual_link_build=v['build'],actual_both_GPU_component_proofs=v['component_proofs'],counter_scope=v['counter_scope'],source_guard='ordinary ratio2, no cand/cand_out/block_ids; ratio1S8/compact and all unsupported shapes literal original',model_parity_pending=True,quality_admitted=False,performance_admitted=False)
