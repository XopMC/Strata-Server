"""Finite source/AST checks only. Never imports runtime/transport/owner/device code."""
from pathlib import Path
import ast,hashlib,json,re
P=Path(__file__).resolve().parent;M=P.parents[1];A=M/'research/paired-server-v252'
def check():
 literal=('strict_owner.py','api_transport_v252.py','lifecycle.py','maintenance_v146.py','test_transport_v252.py')
 for name in literal:assert(P/name).read_bytes()==(A/name).read_bytes(),name
 assert(P/'index_devices_v536.py').read_bytes()==(M/'research/pair-runtime-adapter-v506/source/index_devices_v506.py').read_bytes()
 files=sorted(P.glob('*.py'))
 for p in files:ast.parse(p.read_text(),filename=str(p))
 server=(P/'serve.py').read_text();binding=(P/'bindings.py').read_text();launch=(P/'launch.py').read_text();transport=(P/'serving_transport.py').read_text();root=(P/'root_metadata_v565.py').read_text()
 for text in(server,binding,launch):assert re.search(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}',text)is None
 assert "require_prior_owner_closed();pair=resolve_current_pair()"in server
 assert 'acquire_leases(lease_paths(pair))'in server and 'require_prior_owner_closed();health();parent_guard();save()'in server
 assert 'assert_pair_current(pair)'in server and "CUDA_VISIBLE_DEVICES=','.join(row['uuid']for row in pair)"in server
 assert 'def parent_cpu_guard():'in server and 'def maintenance_exclusion(child):'in server
 assert '     def exclusion():\n      maintenance_exclusion(child)'in server
 assert 'parent_cpu_guard()\n need({int(p.name)'in server
 assert server.count('owner.OneBankStore(')==1 and 'preload_selected('in server
 assert "maintenance=Manager(store,owner,v['maintenance_config']"in server
 assert "Path(v['adapter_root']),v['adapter_seal_sha256']"in server
 assert "selected=importlib.import_module('serving_runtime_v529')"in server and 'runtime=selected.create(store,context=v[\'context\'])'in server
 tree=ast.parse(server);child=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='serve_child')
 calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='import_module'and n.args and isinstance(n.args[0],ast.Constant)and n.args[0].value=='serving_runtime_v529']
 assert len(calls)==1 and calls[0]in list(ast.walk(child))
 assert "need(exit_code==0,'child failed/abnormal; no replacement or automatic replay')"in server
 assert server.index("close=read(close_path);validate_close_proof")<server.index('if next_plan is not None and not stop:')
 assert server.index("need(child is None,'bank must remain pinned")<server.index('store.close()')
 assert 'sequence_base=generation<<32'in server and 'request_sequence_limit'in transport
 for name in('output_token_ids_sha256','prefill_logits_sha256','encoded_prompt_sha256','encoded_prompt_token_ids_sha256','checkpoint','actual_prefill_shape','output_unique_ids','mtp_summary','serving528','request_sequence_id'):assert name in transport
 assert "self.log('REQUEST_RESULT',request_sequence_id=sequence,**compact_result_proof(v))"in transport
 assert "summary={k:stats[k]for k in keys"in transport and "'windows'"not in transport.split('keys=')[1].split('summary=')[0]
 assert "BASE_CONFIG=M/'requests/root-paired-maintenance-v155/maintenance-config.json'"in root and "cfg['format_contracts']=[]"in root
 assert "if k!='format_contracts'"in root and 'source_guard();d=delivery(execution=True)'in root
 assert 'parent_or_model_started=False'in root
 t=json.loads((P/'PROFILE_TEMPLATES.json').read_text());assert set(t)=={'262144','524288'}
 for c,v in t.items():assert v['context']==int(c)and v['ready']is False and v['selected_system_indices']==[0,1]and v['persistent_UUID_allowlist']is False and v['host']=='0.0.0.0'and v['port']==49287 and v['TTL_seconds']==7200 and v['api_key_required']is False and v['actual_source_revision']==565 and v['initial_factory_source_revision']==563 and v['initial_native_source_revision']==558
 assert "'-p','test_*.py'"in(P/'root_transport_tests_v565.py').read_text()
 methods=0
 for name in('test_transport_v252.py','test_metadata_v536.py','test_admission_v536.py','test_dependency_pins_v536.py','test_delivery_contract_v536.py'):
  methods+=sum(isinstance(n,ast.FunctionDef)and n.name.startswith('test_')for n in ast.walk(ast.parse((P/name).read_text())))
 assert methods==23
 for name in('test_closure_hash_cache.py','test_cache_integration_v565.py'):
  methods+=sum(isinstance(n,ast.FunctionDef)and n.name.startswith('test_')for n in ast.walk(ast.parse((P/name).read_text())))
 assert methods==36
 assert "FACTORY_ROOT=M/'research/paired-console-delivery-v563'"in binding
 assert 'allow_profile_admission_v163=True,profile_validator_v163=validate_manager_profile'in server
 assert "state.update(phase='BANK_IDLE'"in server and "if unload and not stop:"in server
 assert server.index('close=read(close_path);validate_close_proof')<server.index('if unload and not stop:')
 assert 'registry.admit(pin)'in server and 'registry.select(parts[1]'in server
 assert "factory_root=Path(v['factory_root'])"in server and 'str(factory_root)'in server
 assert 'line.strip().lower()'not in server
 assert 'SO_PEERCRED'in server and 'len(raw)<=4096'in server
 admission=(P/'admission_v536.py').read_text()
 assert 'self.manager.admit_profile_v163('in admission and 'self.manager._read_profile_v163('in admission
 assert admission.index('self.manager._read_profile_v163(')<admission.index('self.manager.admit_profile_v163(')<admission.index('self.entries[ident]=')
 assert 'ALREADY_ADMITTED'in admission and 'partial admission pin differs'in admission
 assert "self.manager.child_drain_proved"in admission and 'from cpu_expected_preflight_v512_r2 import check_expected'in admission
 assert 'importlib'not in admission and 'ctypes'not in admission and 'subprocess'not in admission
 assert 'check_delivery_header(v)'in binding and "check_native_receipts(build,idle,v['contexts'])"in binding
 assert 'native_evidence=native_provenance(v,root,build,idle)'in binding and "idle.get('build')==v['native528_build']"in binding
 assert 'native_source_evidence=descriptor' in admission
 assert "native=result.get('serving528'"in transport and "'carry_rows'"in transport
 assert (P/'cpu_expected_preflight_v512_r2.py').read_bytes()==(M/'research/pair-runtime-adapter-v512/cpu_expected_preflight_v512_r2.py').read_bytes()
 old_binding=(M/'research/paired-console-server-v536/bindings.py').read_text()
 sha_ast=lambda s:ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='sha'))
 assert sha_ast(binding)==sha_ast(old_binding)
 assert 'CONTENT_HASH_CACHE.verify_project' in binding
 assert 'CONTENT_HASH_CACHE.verify_project' in (P/'dependency_pins_v536.py').read_text()
 assert 'from content_cache_v565 import CONTENT_HASH_CACHE' in binding
 assert "actual_source_revision=565" in server and "'actual_source_revision'" in binding
 assert 'pair-serving-v565' in binding and 'pair-serving-v536/pair-544-recovery-20261026/state.json' in binding
 assert "ACTUAL_SOURCE_REVISION=565" in binding
 assert "DS1017_V565_ROOT_METADATA" in root
 return dict(phase='SOURCE536_LITERAL_LIFECYCLE_AND_CURRENT_INDEX_AST_PASS',actual_source_revision=565,literal252_files=list(literal),index_resolver_literal506=True,source_modules=len(files),Root_fixture_cases=methods,one_cold_bank=True,no_fixed_UUID_profiles=True,only_fresh_child_imports_factory=True,new_log_proof_is_compact=True,API_model_proof_pending=True,author_transport_native_GPU_bank_load_or_server_execution=False)
if __name__=='__main__':print(json.dumps(check(),indent=2))
