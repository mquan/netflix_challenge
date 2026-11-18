"""Real full-fit/inference execution for the context production adapter."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_context_blend as c

CHILD='''
import sys
from pathlib import Path
import upstream_context_production as adapter
adapter.configure()
root=Path(sys.argv[1]);core=adapter.core
core.RUN=root/'models';core.REPORT=root/'report';adapter.bank.RUN=root/'bank'
plan=adapter.u.read(core.REPORT/'plan.json')
core.verify=lambda:plan
core.worker('nn',replay_only=sys.argv[2]=='replay')
'''


class ContextProductionTest(unittest.TestCase):
    def test_prepare_uses_distinct_budget_receipts_and_rejects_wrong_manifest(self):
        import upstream_context_production as adapter
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo/'artifacts') as temporary, ExitStack() as stack:
            root=Path(temporary)
            for name in ['bank','bank-report','dev','dev-report','parent','parent-report','report']:
                (root/name).mkdir()
            paths=[(adapter,'RUN',root/'models'),(adapter,'REPORT',root/'report'),
                (adapter.bank,'RUN',root/'bank'),(adapter.bank,'REPORT',root/'bank-report'),
                (c,'RUN',root/'dev'),(c,'REPORT',root/'dev-report'),
                (c.parent,'RUN',root/'parent'),(c.parent,'REPORT',root/'parent-report'),
                (adapter.n,'CONFIG',c.CONFIG)]
            for module,name,value in paths:stack.enter_context(patch.object(module,name,value))
            stack.enter_context(patch.object(adapter,'development_gate',return_value=.8))
            stack.enter_context(patch.object(adapter,'code_hashes',return_value={}))
            for name in ['manifest','evaluation','independent-verification','nn-replay']:
                c.u.write(root/'dev-report'/(name+'.json'),dict(origin='context'))
            c.u.write(root/'parent-report/manifest.json',dict(origin='original tree'))
            c.u.write(root/'bank-report/manifest.json',{})
            for family,origin in [('tree',root/'parent-report'),('nn',root/'dev-report')]:
                for fold in range(5):
                    budget=[10+fold,20+fold,30+fold] if family=='tree' else [1+fold,2+fold,3+fold]
                    c.u.write(origin/family/(str(fold)+'.json'),dict(outputs={},chosen_iterations=budget,
                        manifest_sha256=c.u.sha(origin/'manifest.json')))
            z=np.zeros((8,161));z[:,-1]=1
            np.save(root/'bank/probex-design.npy',z);np.save(root/'bank/labels.npy',np.full(8,3.6))
            data=dict(source_sha256={},files={name:c.u.sha(root/'bank'/name) for name in ['probex-design.npy','labels.npy']},qualifying_rows=7)
            stack.enter_context(patch.object(adapter.bank,'build',return_value=data))
            path=root/'parent-report/tree/0.json';receipt=c.u.read(path)
            wrong=dict(receipt,manifest_sha256=c.u.sha(root/'dev-report/manifest.json'))
            c.u.write(path,wrong)
            with self.assertRaisesRegex(ValueError,'budget source changed'):adapter.prepare()
            self.assertFalse((root/'models').exists())
            c.u.write(path,receipt);plan=adapter.prepare()
            self.assertEqual(plan['iterations'],dict(tree=[12,22,32],nn=[3,4,5]))
            self.assertNotEqual(plan['budget_provenance']['tree'],plan['budget_provenance']['nn'])
            self.assertEqual(len([name for name in plan['source_sha256'] if '/tree/' in name or '/nn/' in name]),10)
            self.assertFalse(plan['qualifying_labels_read'])

    def test_full_fit_saved_model_replay_rejects_prediction_substitution(self):
        repo=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(repo/'scripts'),OPENBLAS_NUM_THREADS='2',
                 VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ['bank','models','report']:(root/name).mkdir()
            rng=np.random.default_rng(20261118)
            for split,rows in [('probex',64),('qual',7)]:
                z=rng.normal(size=(rows,161));z[:,-1]=1
                features=rng.normal(size=(rows,221)).astype(np.float32)
                np.save(root/'bank'/(split+'-design.npy'),z)
                np.save(root/'bank'/(split+'-features.npy'),features)
                if split=='probex':np.save(root/'bank/labels.npy',3.6+.25*features[:,0])
            weights=np.zeros(161);weights[-1]=3.6;np.save(root/'models/ridge.npy',weights)
            plan=dict(config=c.CONFIG,seeds=c.CONFIG['seeds'],iterations=dict(nn=[0,1,1]),
                ridge_sha256=c.u.sha(root/'models/ridge.npy'))
            c.u.write(root/'report/plan.json',plan)
            for mode in ['fit','replay']:
                result=subprocess.run([sys.executable,'-c',CHILD,str(root),mode],cwd=repo,
                    env=env,capture_output=True,text=True,timeout=45)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            proof=c.u.read(root/'report/nn-replay.json');self.assertTrue(proof['bitwise_equal'])
            self.assertFalse((root/'models/nn/nn-2026.pt').exists())
            path=root/'models/nn/qualifying.f32';prediction=np.fromfile(path,dtype='<f4')
            self.assertEqual(prediction.shape,(7,));self.assertTrue(np.isfinite(prediction).all())
            (prediction+.125).astype('<f4').tofile(path)
            receipt=c.u.read(root/'report/nn.json');receipt['outputs']['qualifying.f32']=c.u.sha(path)
            c.u.write(root/'report/nn.json',receipt)
            result=subprocess.run([sys.executable,'-c',CHILD,str(root),'replay'],cwd=repo,
                env=env,capture_output=True,text=True,timeout=45)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Final model replay differs',result.stderr)


if __name__=='__main__':unittest.main()
