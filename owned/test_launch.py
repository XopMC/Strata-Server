import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('owned_launch',Path(__file__).with_name('launch.py'));launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
class LaunchPlanTests(unittest.TestCase):
 def args(self,*extra):return launch.parser().parse_args(['--context','512k','--pack','/models/pack','--common','/models/common.gguf','--mtp','/models/mtp','--expert-profile','/models/expert-profile.bin',*extra])
 def test_exact_profile(self):
  p=launch.plan(self.args());self.assertEqual(p['context'],524288);self.assertTrue(p['native_p2p_required']);self.assertFalse(p['legacy_fallback']);self.assertEqual(p['environment']['STRATA_ROUTE_TAIL_SKIP'],'0');self.assertNotIn('CUDA_VISIBLE_DEVICES',p['environment'])
 def test_context_1m(self):
  p=launch.plan(self.args('--context','1m'));self.assertEqual(p['context'],1048576);self.assertEqual(p['argv'][p['argv'].index('--rope-scale')+1],'4.0')
 def test_no_topology_substitution(self):
  for g in ['0','0,1,2','0,0','-1,1']:
   with self.assertRaises(ValueError):launch.plan(self.args('--gpus='+g))
 def test_unqualified_context_refused(self):
  with self.assertRaises(ValueError):launch.plan(self.args('--context','256k'))
 def test_fresh_nonzero_indices(self):self.assertEqual(launch.plan(self.args('--gpus','3,5'))['current_gpu_indices'],[3,5])
 def test_unqualified_import_refused(self):
  with self.assertRaisesRegex(ValueError,'not yet qualified'):
   launch.plan(self.args('--profile','swift-nvfp4-one-512k','--gpus','0'))
 def test_no_shared_bank_default(self):self.assertNotIn('--shared-expert-arena',launch.plan(self.args())['argv'])
if __name__=='__main__':unittest.main()
