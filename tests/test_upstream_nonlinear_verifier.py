from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_nonlinear as n
import verify_upstream_nonlinear as v


class NonlinearVerification(unittest.TestCase):
    def test_model_replay_rejects_self_consistent_but_wrong_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; report = root/'report'
            run.mkdir(); report.mkdir()
            n.u.write(report/'manifest.json', {})
            groups = np.tile(np.arange(10), 2)
            np.save(run/'features.npy', np.zeros((20, 3), dtype=np.float32))
            np.save(run/'design.npy', np.ones((20, 1)))
            np.save(run/'groups.npy', groups)
            for fold in range(5):
                dest = run/'tree'/str(fold); dest.mkdir(parents=True)
                np.savez(dest/'ridge.npz', full=np.array([3.6]), inner=np.array([3.6]))
                np.full(4, 3.6, dtype='<f4').tofile(dest/'prediction.f32')
                n.u.write(report/'tree'/(str(fold)+'.json'),
                    dict(manifest_sha256=n.u.sha(report/'manifest.json'), chosen_iterations=[0,0,0],
                         outputs={p.name:n.u.sha(p) for p in dest.iterdir()}))
            with patch.multiple(n, RUN=run, REPORT=report), patch.object(n, 'verify_prepared', return_value={}):
                v.replay('tree')
                self.assertTrue(n.u.read(report/'tree-replay.json')['records'][0]['bitwise_equal'])
                changed = run/'tree/2/prediction.f32'
                np.full(4, 3.7, dtype='<f4').tofile(changed)
                receipt = n.u.read(report/'tree/2.json')
                receipt['outputs']['prediction.f32'] = n.u.sha(changed)
                n.u.write(report/'tree/2.json', receipt)
                with self.assertRaisesRegex(ValueError, 'Saved model replay differs'): v.replay('tree')

    def test_independent_score_and_fixed_average_handle_float32_clip_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; report = root/'report'
            run.mkdir(); report.mkdir()
            y = np.tile(np.arange(1, 6), 4).astype(np.float64)
            users = np.repeat(np.arange(5), 4)
            baseline = np.full(20, 3.6)
            for name, value in [('labels',y), ('users',users), ('baseline',baseline)]:
                np.save(run/(name+'.npy'), value)
            tree = np.tile([1.05, 4.95], 10).astype('<f4')
            neural = np.full(20, 3.5, dtype='<f4')
            predictions = dict(control=baseline, tree=tree, nn=neural,
                               **{'fixed-half-tree-nn':(tree.astype(np.float64)+neural)/2})
            records = []
            for arm, value in predictions.items():
                path = run/(arm+'.f32'); value.astype('<f4').tofile(path)
                records.append(dict(arm=arm, rmse=n.x.metric(np.fromfile(path,dtype='<f4'),y),
                                    prediction_sha256=n.u.sha(path)))
            n.u.write(report/'evaluation.json', dict(records=records))
            with patch.multiple(n, RUN=run, REPORT=report), patch.object(n, 'verify_prepared', return_value={}):
                v.score()
                verified = n.u.read(report/'independent-verification.json')
                self.assertTrue(verified['fixed_average_bitwise_equal'])
                self.assertEqual(verified['scored_users'], 5)
                np.zeros(20, dtype='<f4').tofile(run/'tree.f32')
                with self.assertRaisesRegex(ValueError, 'Final prediction hash changed'): v.score()


if __name__ == '__main__': unittest.main()
