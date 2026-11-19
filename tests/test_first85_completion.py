"""Reject altered family averages and wrong recovered group populations."""
import unittest
from unittest.mock import patch
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import verify_upstream_first85_completion as v


class CompletionAuditTest(unittest.TestCase):
    def test_real_family_replays_use_isolated_runtimes_and_reject_substitution(self):
        repo=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(repo/'scripts'),OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
        child='''
import sys
from pathlib import Path
import verify_upstream_first85_completion as v
root=Path(sys.argv[1]);family=sys.argv[2];mode=sys.argv[3]
v.adapter.RUN=root/'models';v.adapter.REPORT=root/'report';v.adapter.bank.RUN=root/'bank'
v.REPORT=root/'audit';v.ROWS=7;v.adapter.configure()
plan=v.u.read(root/'report/plan.json');v.adapter.core.verify=lambda:plan
if mode=='fit':v.adapter.core.worker(family)
else:v.replay_family(family)
'''
        with tempfile.TemporaryDirectory(dir=repo/'artifacts') as temporary:
            root=Path(temporary)
            for name in ['bank','models','report','audit']:(root/name).mkdir()
            rng=np.random.default_rng(20261119)
            for split,rows in [('probex',64),('qual',7)]:
                z=rng.normal(size=(rows,161));z[:,-1]=1;f=rng.normal(size=(rows,221)).astype(np.float32)
                np.save(root/'bank'/(split+'-design.npy'),z);np.save(root/'bank'/(split+'-features.npy'),f)
                if split=='probex':np.save(root/'bank/labels.npy',3.6+.2*f[:,0])
            weights=np.zeros(161);weights[-1]=3.6;np.save(root/'models/ridge.npy',weights)
            plan=dict(config=v.adapter.core.n.CONFIG,seeds=[2026,7,13],iterations=dict(tree=[0,1,1],nn=[0,1,1]),
                ridge_sha256=v.u.sha(root/'models/ridge.npy'))
            v.u.write(root/'report/plan.json',plan)
            for family in ['tree','nn']:
                for mode in ['fit','replay']:
                    result=subprocess.run([sys.executable,'-c',child,str(root),family,mode],cwd=repo,env=env,
                        capture_output=True,text=True,timeout=60)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                proof=v.u.read(root/'audit'/(family+'-replay.json'))
                self.assertTrue(proof['bitwise_equal']);self.assertEqual(proof['family'],family)
                path=root/'models'/family/'qualifying.f32';values=np.fromfile(path,dtype='<f4')
                (values+.125).astype('<f4').tofile(path)
                receipt_path=root/'report'/(family+'.json');receipt=v.u.read(receipt_path)
                receipt['outputs']['qualifying.f32']=v.u.sha(path);v.u.write(receipt_path,receipt)
                result=subprocess.run([sys.executable,'-c',child,str(root),family,'replay'],cwd=repo,env=env,
                    capture_output=True,text=True,timeout=60)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Independent saved-model inference differs',result.stderr)

    def test_fixed_average_rejects_valid_but_substituted_final(self):
        tree=np.array([1.05,3.25,4.95],dtype='<f4');nn=np.array([4.95,2.75,1.05],dtype='<f4')
        final=((tree.astype(np.float64)+nn.astype(np.float64))/2).astype('<f4')
        self.assertTrue(v.average_contract(tree,nn,final))
        with self.assertRaisesRegex(ValueError,'exact fixed family average'):
            v.average_contract(tree,nn,final+.01)
        with self.assertRaisesRegex(ValueError,'Invalid clipped'):
            v.average_contract(tree.astype(np.float64),nn,final)

    def test_both_original_source_groups_must_pass_strict_target(self):
        labels=np.array([[1,0],[2,0],[3,1],[4,1]],dtype=np.int16)
        # Use an exactly representable boundary to distinguish '<' from '<='.
        with patch.object(v,'TARGET',1.0):
            predictions=labels[:,0].astype(np.float64)+1.0
            groups=v.score_groups(predictions,labels,expected_counts=(2,2))
            self.assertTrue(all(g['rmse']==1.0 for g in groups.values()))
            self.assertFalse(any(g['below_target'] for g in groups.values()))
        predictions=np.array([1.1,2.1,3.1,4.1])
        groups=v.score_groups(predictions,labels,expected_counts=(2,2))
        self.assertTrue(all(g['below_target'] for g in groups.values()))
        for group in groups.values():self.assertAlmostEqual(group['rmse'],.1)
        with self.assertRaisesRegex(ValueError,'required populations'):
            v.score_groups(predictions,labels,expected_counts=(1,3))
        labels[0,1]=2
        with self.assertRaisesRegex(ValueError,'label shape/domain'):
            v.score_groups(predictions,labels,expected_counts=(2,2))


if __name__=='__main__':unittest.main()
