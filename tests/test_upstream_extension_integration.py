"""Exercise the new-width production adapter with real isolated model runtimes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import upstream_extension_bank as bank
import upstream_nonlinear as nonlinear
import upstream_repro as u


CHILD = '''
import sys
from pathlib import Path
import upstream_extension_production as adapter
adapter.configure()
root=Path(sys.argv[1]);family=sys.argv[2]
core=adapter.core
core.RUN=root/'models';core.REPORT=root/'report';adapter.bank.RUN=root/'bank'
plan=adapter.u.read(core.REPORT/'plan.json')
core.n.CONFIG=plan['config']
core.verify=lambda:plan
core.worker(family,replay_only=sys.argv[3]=='replay')
'''


class ExtensionRuntimeChecks(unittest.TestCase):
    def test_real_new_width_models_roundtrip_and_reject_semantic_corruption(self):
        repo=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(repo/'scripts'),OPENBLAS_NUM_THREADS='2',
                 VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'bank').mkdir();(root/'models').mkdir();(root/'report').mkdir()
            rng=np.random.default_rng(20261118)
            for split,rows in [('probex',64),('qual',7)]:
                old=rng.normal(size=(rows,161));old[:,-1]=1
                features=rng.normal(size=(rows,221)).astype(np.float32)
                added=rng.normal(3.6,.2,size=(rows,6)).astype(np.float32)
                z,f=bank.blocks(old,features,added)
                self.assertEqual((z.shape[1],f.shape[1]),(167,227))
                np.save(root/'bank'/(split+'-design.npy'),z)
                np.save(root/'bank'/(split+'-features.npy'),f)
                if split=='probex':np.save(root/'bank/labels.npy',3.6+.25*features[:,0])
            weights=np.zeros(167);weights[-1]=3.6;np.save(root/'models/ridge.npy',weights)
            cfg=dict(nonlinear.CONFIG,nn_hidden=8,nn_batch=16,tree_min_leaf=2)
            plan=dict(config=cfg,seeds=cfg['seeds'],iterations=dict(tree=[0,2,2],nn=[0,1,1]),
                      ridge_sha256=u.sha(root/'models/ridge.npy'))
            u.write(root/'report/plan.json',plan)
            for family in ['tree','nn']:
                for mode in ['fit','replay']:
                    result=subprocess.run([sys.executable,'-c',CHILD,str(root),family,mode],cwd=repo,
                        env=env,capture_output=True,text=True,timeout=45)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                record=u.read(root/'report'/(family+'-replay.json'))
                self.assertTrue(record['bitwise_equal'])
                prediction=np.fromfile(root/'models'/family/'qualifying.f32',dtype='<f4')
                self.assertEqual(prediction.shape,(7,));self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(np.all((prediction>=1.05-1e-6)&(prediction<=4.95+1e-6)))
                suffix='txt' if family=='tree' else 'pt'
                self.assertFalse((root/'models'/family/(family+'-2026.'+suffix)).exists())
            # Updating the receipt's byte hash must not substitute for inference
            # from the actual new-width saved network and paired feature matrix.
            path=root/'models/nn/qualifying.f32'
            prediction=np.fromfile(path,dtype='<f4');prediction[0]+=.01;prediction.tofile(path)
            receipt=u.read(root/'report/nn.json');receipt['outputs']['qualifying.f32']=u.sha(path)
            u.write(root/'report/nn.json',receipt)
            failed=subprocess.run([sys.executable,'-c',CHILD,str(root),'nn','replay'],cwd=repo,
                env=env,capture_output=True,text=True,timeout=45)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('Final model replay differs',failed.stdout+failed.stderr)


if __name__=='__main__':unittest.main()
