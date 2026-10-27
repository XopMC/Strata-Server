import tempfile,unittest
from pathlib import Path
from numa_policy import NumaPolicyError,resolve_interleave_args

class NumaPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.nodes=self.root/'nodes';self.nodes.mkdir();self.status=self.root/'status'
    def fixture(self,online,allowed,totals):
        (self.nodes/'online').write_text(online+'\n');self.status.write_text('Name:\ttest\nMems_allowed_list:\t'+allowed+'\n')
        for n,total in totals.items():
            p=self.nodes/f'node{n}';p.mkdir(exist_ok=True);(p/'meminfo').write_text(f'Node {n} MemTotal: {total} kB\n')
    def resolve(self,args):return resolve_interleave_args(args,node_root=self.nodes,status_path=self.status)
    def test_two_nodes_identical(self):
        self.fixture('0-1','0-1',{0:402653184,1:402653184})
        args=['numactl','--all','--physcpubind=0-127','--interleave=0,1','native','--serve']
        self.assertEqual(args,self.resolve(args))
    def test_eight_nodes(self):
        self.fixture('0-7','0-7',{n:100663296 for n in range(8)})
        self.assertEqual(self.resolve(['--interleave=0,1']),['--interleave=0,1,2,3,4,5,6,7'])
        self.assertEqual(self.resolve(['--interleave=all']),['--interleave=0,1,2,3,4,5,6,7'])
    def test_scope_intersection(self):
        self.fixture('0-7','2-5',{n:100 for n in range(2,6)})
        self.assertEqual(self.resolve(['--interleave=all']),['--interleave=2,3,4,5'])
    def test_memoryless_filtered(self):
        self.fixture('0-2','0-2',{0:100,1:0,2:100})
        self.assertEqual(self.resolve(['--interleave=all']),['--interleave=0,2'])
    def test_non_target_untouched_without_discovery(self):
        args=['--interleave=2,3','--membind=1','unchanged']
        self.assertEqual(args,self.resolve(args))
    def test_empty_or_disallowed(self):
        for online,allowed,totals in [('','0',{}),('0-1','',{}),('0-1','2-3',{}),('0','0',{0:0})]:
            with self.subTest(online=online,allowed=allowed):
                self.fixture(online,allowed,totals)
                with self.assertRaises(NumaPolicyError):self.resolve(['--interleave=all'])
    def test_missing_or_ambiguous_inventory(self):
        self.fixture('0-1','0-1',{0:100})
        with self.assertRaises(NumaPolicyError):self.resolve(['--interleave=all'])
        p=self.nodes/'node0'/'meminfo';p.write_text('Node 1 MemTotal: 100 kB\n')
        with self.assertRaises(NumaPolicyError):self.resolve(['--interleave=all'])
    def test_conflicting_policies_refused(self):
        for args in [['--interleave=all','--interleave=0,1'],['--interleave=0,1','--membind=0'],['--interleave=all','--preferred=0'],['--interleave=all','--localalloc'],['--interleave=all','--interleave','2']]:
            with self.subTest(args=args):
                with self.assertRaises(NumaPolicyError):self.resolve(args)
    def test_ambiguous_scope_refused(self):
        self.fixture('0-1','0-1',{0:100,1:100})
        self.status.write_text('Mems_allowed_list: 0-1\nMems_allowed_list: 0\n')
        with self.assertRaises(NumaPolicyError):self.resolve(['--interleave=all'])
    def test_bad_lists(self):
        for bad in ['0-1,1','x','1-0','0,','0 1']:
            self.fixture(bad,'0-1',{0:100,1:100})
            with self.assertRaises(NumaPolicyError):self.resolve(['--interleave=all'])
    def test_input_validation(self):
        for value in ['--interleave=all',[123]]:
            with self.assertRaises(NumaPolicyError):self.resolve(value)

if __name__=='__main__':unittest.main(verbosity=2)
