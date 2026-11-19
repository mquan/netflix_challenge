"""Actual86-input family fits/replay and independent budget-origin integrity."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_cf12_matching_production as m

CHILD='''
import sys
from pathlib import Path
import upstream_cf12_matching_production as a
a.configure();root=Path(sys.argv[1]);family=sys.argv[2];c=a.core
a.RUN=c.RUN=root/'models';a.REPORT=c.REPORT=root/'report';a.bank.RUN=root/'bank'
plan=a.u.read(c.REPORT/'plan.json');a.n.CONFIG=plan['config']
a.development.parent.CONFIG=plan['config']
c.verify=lambda:plan
a.ORIGINAL_WORKER(family,sys.argv[3]=='replay')
'''


class MatchingProductionTest(unittest.TestCase):
    def setUp(self):m.development.configure()

    def test_real_both86_families_roundtrip_and_retagged_prediction_rejected(self):
        repo=m.u.ROOT
        env=dict(os.environ,PYTHONPATH=str(repo/'scripts'),OPENBLAS_NUM_THREADS='2',
                 VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
        with tempfile.TemporaryDirectory(dir=repo/'artifacts') as temp:
            root=Path(temp)
            for name in ['bank','models','report']:(root/name).mkdir()
            rng=np.random.default_rng(42)
            for split,rows in [('probex',64),('qual',7)]:
                z=rng.normal(size=(rows,162));z[:,-1]=1
                f=rng.normal(size=(rows,222)).astype(np.float32)
                np.save(root/'bank'/(split+'-design.npy'),z)
                np.save(root/'bank'/(split+'-features.npy'),f)
                if split=='probex':np.save(root/'bank/labels.npy',3.6+.4*f[:,0])
            weights=np.zeros(162);weights[-1]=3.6;np.save(root/'models/ridge.npy',weights)
            cfg=dict(m.development.CONFIG,tree_min_leaf=2,nn_hidden=8,nn_batch=16)
            plan=dict(config=cfg,seeds=cfg['seeds'],iterations=dict(tree=[2,0,2],nn=[0,1,1]),
                      qualifying_rows=7,ridge_sha256=m.u.sha(root/'models/ridge.npy'))
            m.u.write(root/'report/plan.json',plan)
            def child(family,mode):
                return subprocess.run([sys.executable,'-c',CHILD,str(root),family,mode],cwd=repo,
                    env=env,capture_output=True,text=True,timeout=45)
            for family in ['tree','nn']:
                for mode in ['fit','replay']:
                    result=child(family,mode);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertTrue(m.u.read(root/'report'/(family+'-replay.json'))['bitwise_equal'])
                values=np.fromfile(root/'models'/family/'qualifying.f32',dtype='<f4')
                self.assertEqual(values.shape,(7,));self.assertTrue(np.isfinite(values).all())
            import lightgbm as lgb
            model=lgb.Booster(model_file=str(root/'models/tree/tree-2026.txt'))
            self.assertEqual(model.num_trees(),2)
            self.assertFalse((root/'models/tree/tree-7.txt').exists())
            self.assertFalse((root/'models/nn/nn-2026.pt').exists())
            path=root/'models/nn/qualifying.f32';values=np.fromfile(path,dtype='<f4');values[0]+=.1;values.tofile(path)
            receipt=m.u.read(root/'report/nn.json');receipt['outputs']['qualifying.f32']=m.u.sha(path)
            m.u.write(root/'report/nn.json',receipt)
            failed=child('nn','replay');self.assertNotEqual(failed.returncode,0)
            self.assertIn('Final model replay differs',failed.stderr)

    def test_actual_distinct_budget_origins_and_source_mutation_rejection(self):
        with tempfile.TemporaryDirectory(dir=m.u.ROOT/'artifacts') as temp:
            root=Path(temp);origins=[]
            for family in ['tree','nn']:
                report=root/(family+'-report');cache=root/(family+'-cache');report.mkdir();cache.mkdir()
                m.u.write(report/'manifest.json',dict(config=m.n.CONFIG,origin=family))
                for fold in range(5):
                    directory=cache/family/str(fold);directory.mkdir(parents=True)
                    model=directory/'model.bin';model.write_bytes(bytes([fold]))
                    m.u.write(report/family/(str(fold)+'.json'),dict(family=family,fold=fold,
                        manifest_sha256=m.u.sha(report/'manifest.json'),chosen_iterations=[fold+1,fold+2,fold+3],
                        outputs={'model.bin':m.u.sha(model)}))
                origins.append((family,report,cache))
            budgets,sources=m.budget_sources(origins)
            self.assertEqual(budgets,dict(tree=[3,4,5],nn=[3,4,5]))
            self.assertFalse((origins[0][1]/'nn').exists())
            nn_receipt=origins[1][1]/'nn/0.json'
            self.assertIn(str(nn_receipt.relative_to(m.u.ROOT)),sources)
            tree_receipt=origins[0][1]/'tree/0.json';original=m.u.read(tree_receipt)
            changed=dict(original,manifest_sha256=m.u.sha(origins[1][1]/'manifest.json'))
            m.u.write(tree_receipt,changed)
            with self.assertRaisesRegex(ValueError,'originating'):m.budget_sources(origins)
            m.u.write(tree_receipt,original);(origins[1][2]/'nn/0/model.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):m.budget_sources(origins)


if __name__=='__main__':unittest.main()
