"""Actual neural refits and semantic replay of the explicitly reused tree."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_cf12_production as m

CHILD='''
import sys
from pathlib import Path
from types import SimpleNamespace
import upstream_cf12_production as a
a.configure();root=Path(sys.argv[1]);mode=sys.argv[2];c=a.core
a.RUN=c.RUN=root/'models';a.REPORT=c.REPORT=root/'report';a.bank.RUN=root/'bank'
a.FIRST_RUN=root/'first-models';a.FIRST_REPORT=root/'first-report'
plan=a.u.read(a.REPORT/'plan.json') if mode!='make-origin' else None
c.verify=lambda:plan
def origin(replay):
    saved=(c.RUN,c.REPORT,c.bank,a.n.CONFIG,c.verify)
    try:
        old=a.u.read(a.FIRST_REPORT/'plan.json')
        c.RUN=a.FIRST_RUN;c.REPORT=a.FIRST_REPORT;c.bank=SimpleNamespace(RUN=root/'first-bank')
        a.n.CONFIG=old['config'];c.verify=lambda:old
        a.ORIGINAL_WORKER('tree',replay)
    finally:c.RUN,c.REPORT,c.bank,a.n.CONFIG,c.verify=saved
if mode=='make-origin':
    origin(False);origin(True)
elif mode.startswith('nn-'):a.worker('nn',mode=='nn-replay')
else:
    a.replay_origin_tree=lambda:origin(True)
    a.worker('tree',mode=='tree-replay')
'''


class Cf12ProductionTest(unittest.TestCase):
    def run_child(self,root,mode,success=True):
        repo=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(repo/'scripts'),OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
        result=subprocess.run([sys.executable,'-c',CHILD,str(root),mode],cwd=repo,env=env,capture_output=True,text=True,timeout=45)
        if success:self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        else:self.assertNotEqual(result.returncode,0)
        return result

    def fixture(self,root):
        for name in ['bank','models','report','first-bank','first-models','first-report']:(root/name).mkdir()
        rng=np.random.default_rng(42)
        for split,rows in [('probex',64),('qual',7)]:
            old=rng.normal(size=(rows,161));old[:,-1]=1
            features=rng.normal(size=(rows,221)).astype(np.float32)
            added=np.clip(rng.normal(size=(rows,1))+3.6,0,6).astype(np.float32)
            new,new_features=m.bank.blocks(old,features,added)
            for folder,z,f in [('first-bank',old,features),('bank',new,new_features)]:
                np.save(root/folder/(split+'-design.npy'),z);np.save(root/folder/(split+'-features.npy'),f)
                if split=='probex':np.save(root/folder/'labels.npy',3.6+.25*features[:,0])
        m.u.write(root/'first-report/bank-manifest.json',dict(prediction_columns=85))
        old_weights=np.zeros(161);old_weights[-1]=3.6;np.save(root/'first-models/ridge.npy',old_weights)
        weights=np.zeros(162);weights[-1]=3.6;np.save(root/'models/ridge.npy',weights)
        sources={str(p.relative_to(m.u.ROOT)):m.u.sha(p) for p in (root/'first-bank').glob('*.npy')}
        sources[str((root/'first-report/bank-manifest.json').relative_to(m.u.ROOT))]=m.u.sha(root/'first-report/bank-manifest.json')
        old=dict(config=m.development.context.BASE_CONFIG,versions=m.n.versions(),seeds=m.development.CONFIG['seeds'],
            iterations=dict(tree=[1,0,0]),ridge_sha256=m.u.sha(root/'first-models/ridge.npy'),
            qualifying_rows=7,qualifying_labels_read=False,source_sha256=sources,
            code_sha256={'scripts/upstream_production.py':m.u.sha(m.CORE_FILE)})
        m.u.write(root/'first-report/plan.json',old)
        self.run_child(root,'make-origin')
        receipt=m.u.read(root/'first-report/tree.json')
        reuse=dict(plan_sha256=m.u.sha(root/'first-report/plan.json'),family_receipt_sha256=m.u.sha(root/'first-report/tree.json'),
            prediction_sha256=receipt['outputs']['qualifying.f32'],source_directory=str(root/'first-models/tree'),
            source_report=str(root/'first-report'),prediction_columns=85,fit_performed_in_this_run=False)
        plan=dict(config=m.development.CONFIG,seeds=m.development.CONFIG['seeds'],iterations=dict(tree=[1,0,0],nn=[0,1,1]),
            qualifying_rows=7,ridge_sha256=m.u.sha(root/'models/ridge.npy'),tree_reuse=reuse)
        m.u.write(root/'report/plan.json',plan)
        return plan

    def test_actual86_neural_fit_replay_and_prediction_substitution(self):
        with tempfile.TemporaryDirectory(dir=m.u.ROOT/'artifacts') as temporary:
            root=Path(temporary);self.fixture(root)
            self.run_child(root,'nn-fit');self.run_child(root,'nn-replay')
            self.assertTrue(m.u.read(root/'report/nn-replay.json')['bitwise_equal'])
            self.assertFalse((root/'models/nn/nn-2026.pt').exists())
            path=root/'models/nn/qualifying.f32';values=np.fromfile(path,dtype='<f4')
            self.assertEqual(values.shape,(7,));self.assertTrue(np.isfinite(values).all())
            (values+.125).astype('<f4').tofile(path)
            receipt=m.u.read(root/'report/nn.json');receipt['outputs']['qualifying.f32']=m.u.sha(path);m.u.write(root/'report/nn.json',receipt)
            result=self.run_child(root,'nn-replay',False)
            self.assertIn('Final model replay differs',result.stderr)

    def test_actual_reused_tree_semantic_replay_and_retagged_corruption(self):
        with tempfile.TemporaryDirectory(dir=m.u.ROOT/'artifacts') as temporary:
            root=Path(temporary);plan=self.fixture(root)
            self.run_child(root,'tree-fit');self.run_child(root,'tree-replay')
            receipt=m.u.read(root/'report/tree.json');self.assertIs(receipt['fit_performed'],False)
            self.assertEqual(list(receipt['outputs']),['qualifying.f32'])
            self.assertTrue(m.u.read(root/'report/tree-replay.json')['bitwise_equal'])
            path=root/'first-models/tree/qualifying.f32'
            (np.fromfile(path,dtype='<f4')+.125).astype('<f4').tofile(path)
            # Retag both publication layers. Only reconstructing the original
            # saved tree with its original161-column ridge exposes the change.
            old=m.u.read(root/'first-report/tree.json');old['outputs']['qualifying.f32']=m.u.sha(path);m.u.write(root/'first-report/tree.json',old)
            plan['tree_reuse']['prediction_sha256']=m.u.sha(path)
            plan['tree_reuse']['family_receipt_sha256']=m.u.sha(root/'first-report/tree.json')
            m.u.write(root/'report/plan.json',plan)
            receipt['plan_sha256']=m.u.sha(root/'report/plan.json');receipt['reused_from']=plan['tree_reuse']
            receipt['outputs']['qualifying.f32']=m.u.sha(path);m.u.write(root/'report/tree.json',receipt)
            result=self.run_child(root,'tree-replay',False)
            self.assertIn('Final model replay differs',result.stderr)

    def test_tree_origin_requires_exact_parent_bank_and_nested_budgets(self):
        with tempfile.TemporaryDirectory(dir=m.u.ROOT/'artifacts') as temporary:
            root=Path(temporary);self.fixture(root)
            data=dict(qualifying_rows=7,parent_bank_manifest_sha256=m.u.sha(root/'first-report/bank-manifest.json'))
            with patch.object(m,'FIRST_RUN',root/'first-models'),patch.object(m,'FIRST_REPORT',root/'first-report'),patch.object(m.bank.parent,'REPORT',root/'first-report'),patch.object(m.n,'CONFIG',m.development.CONFIG):
                # Match the canonical parent filename used by the real adapter.
                (root/'first-report/manifest.json').write_bytes((root/'first-report/bank-manifest.json').read_bytes())
                old=m.u.read(root/'first-report/plan.json')
                old['source_sha256'][str((root/'first-report/manifest.json').relative_to(m.u.ROOT))]=data['parent_bank_manifest_sha256']
                m.u.write(root/'first-report/plan.json',old)
                # Reissue actual origin receipts under that amended source plan.
                for name in ['tree.json','tree-replay.json']:(root/'first-report'/name).unlink()
                import shutil
                shutil.rmtree(root/'first-models/tree')
                self.run_child(root,'make-origin')
                reuse,_=m.tree_origin(data,[1,0,0]);self.assertIs(reuse['fit_performed_in_this_run'],False)
                with self.assertRaisesRegex(ValueError,'exact85'):m.tree_origin(dict(data,parent_bank_manifest_sha256='bad'),[1,0,0])
                with self.assertRaisesRegex(ValueError,'budgets'):m.tree_origin(data,[2,0,0])


if __name__=='__main__':unittest.main()
