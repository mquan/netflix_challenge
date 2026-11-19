"""Actual native-byte integrity and saved-family inference for remote intake."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_remote_temporal_pair as pair

CHILD='''
import sys
from pathlib import Path
import upstream_recovered_temporal_production as adapter
adapter.configure();root=Path(sys.argv[1]);core=adapter.core
core.RUN=root/'models';core.REPORT=root/'report';adapter.bank.RUN=root/'bank'
plan=adapter.u.read(core.REPORT/'plan.json');core.verify=lambda:plan
core.worker(sys.argv[2],replay_only=sys.argv[3]=='replay')
'''


class RecoveredTemporalTest(unittest.TestCase):
    def test_lineage_is_bound_before_freeze_without_changing_prediction_hash(self):
        import upstream_recovered_temporal_production as adapter
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);report=root/'report';bank_report=root/'bank'
            report.mkdir();bank_report.mkdir();(bank_report/'manifest.json').write_text('{"own_bank":true}')
            with patch.object(adapter,'REPORT',report),patch.object(adapter.bank,'REPORT',bank_report):
                record=dict(state='complete',frozen_prediction_sha256='original-frozen-hash',qualifying_labels_read=False)
                adapter.publish(report/'report.json',record)
                saved=pair.u.read(report/'report.json')
                self.assertEqual(saved['frozen_prediction_sha256'],record['frozen_prediction_sha256'])
                self.assertEqual(saved['paired_bank_manifest_sha256'],pair.u.sha(bank_report/'manifest.json'))
                self.assertIn('completed original remote',saved['scope'])
                self.assertFalse(saved['qualifying_labels_read'])
                adapter.publish(report/'unrelated.json',record)
                self.assertEqual(pair.u.read(report/'unrelated.json'),record)

    def test_native_byte_reconstruction_rejects_changed_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);original=root/'original.npy';values=np.array([1.2,3.4,4.5],dtype=np.float32)
            np.save(original,values);header=pair.native_header(original,3)
            digest=hashlib.sha256(original.read_bytes()).hexdigest()
            pair.reconstruct(values,header,digest,root/'reconstructed.npy')
            self.assertEqual(original.read_bytes(),(root/'reconstructed.npy').read_bytes())
            with self.assertRaisesRegex(ValueError,'original native NPY hash'):
                pair.reconstruct(values+.001,header,digest,root/'invalid.npy')
            self.assertFalse((root/'invalid.npy').exists())

    def test_actual_both_families_replay_and_reject_retagged_vector(self):
        repo=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(repo/'scripts'),OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
        import upstream_nonlinear as n
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ['models','report','bank']:(root/name).mkdir()
            rng=np.random.default_rng(20261119)
            for split,rows in [('probex',64),('qual',7)]:
                z=rng.normal(size=(rows,161));z[:,-1]=1
                f=rng.normal(size=(rows,221)).astype(np.float32)
                np.save(root/'bank'/(split+'-design.npy'),z);np.save(root/'bank'/(split+'-features.npy'),f)
                if split=='probex':np.save(root/'bank/labels.npy',3.6+.2*f[:,0])
            weights=np.zeros(161);weights[-1]=3.6;np.save(root/'models/ridge.npy',weights)
            plan=dict(config=n.CONFIG,seeds=n.CONFIG['seeds'],iterations=dict(tree=[0,1,1],nn=[0,1,1]),ridge_sha256=pair.u.sha(root/'models/ridge.npy'))
            pair.u.write(root/'report/plan.json',plan)
            for family in ['tree','nn']:
                for mode in ['fit','replay']:
                    result=subprocess.run([sys.executable,'-c',CHILD,str(root),family,mode],cwd=repo,env=env,capture_output=True,text=True,timeout=60)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertTrue(pair.u.read(root/'report'/(family+'-replay.json'))['bitwise_equal'])
                prediction=root/'models'/family/'qualifying.f32';values=np.fromfile(prediction,dtype='<f4')
                (values+.125).astype('<f4').tofile(prediction)
                receipt_path=root/'report'/(family+'.json');receipt=pair.u.read(receipt_path)
                receipt['outputs']['qualifying.f32']=pair.u.sha(prediction);pair.u.write(receipt_path,receipt)
                result=subprocess.run([sys.executable,'-c',CHILD,str(root),family,'replay'],cwd=repo,env=env,capture_output=True,text=True,timeout=60)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Final model replay differs',result.stderr)


if __name__=='__main__':unittest.main()
