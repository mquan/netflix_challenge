import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_nonlinear as n


class NonlinearChecks(unittest.TestCase):
    def test_grouped_ridge_matches_direct_and_excludes_held_labels(self):
        rng = np.random.default_rng(29)
        z = np.column_stack([rng.normal(size=(300, 5)), np.ones(300)])
        y = rng.uniform(1, 5, 300); groups = np.tile(np.arange(10), 30)
        grams, rhs = n.grouped_stats(z, y, groups)
        for inner in [False, True]:
            eligible = (np.arange(10)//2 != 2) & ((np.arange(10)%2 == 0) if inner else True)
            selected = np.isin(groups, np.flatnonzero(eligible))
            actual = n.ridge_weights(grams, rhs, eligible)
            reg = np.eye(6)*1000; reg[-1, -1] = 0
            expected = np.linalg.solve(z[selected].T@z[selected]+reg, z[selected].T@y[selected])
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            poison = y.copy(); poison[groups//2 == 2] += 100
            pg, pr = n.grouped_stats(z, poison, groups)
            np.testing.assert_array_equal(actual, n.ridge_weights(pg, pr, eligible))

    def test_normalization_uses_only_its_training_rows(self):
        features = np.arange(100, dtype=np.float32).reshape(20, 5)
        fit = np.arange(20) < 10
        before, bs = n.scaled(features, fit)
        poison = features.copy(); poison[~fit] += 10000
        after, ps = n.scaled(poison, fit)
        for key in bs: np.testing.assert_array_equal(bs[key], ps[key])
        np.testing.assert_array_equal(before[fit], after[fit])

    def test_preparation_refuses_changed_code_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'data.npy').write_bytes(b'fixed')
            manifest = dict(config=n.CONFIG, versions=n.versions(), code_sha256=n.code_hashes(),
                prepared_sha256={'data.npy':n.u.sha(root/'data.npy')}, source_manifests_sha256={})
            n.u.write(root/'manifest.json', manifest)
            with patch.multiple(n, REPORT=root, RUN=root):
                n.verify_prepared()
                (root/'data.npy').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Prepared input changed'): n.verify_prepared()

    def test_actual_backends_ignore_outer_labels_through_stopping_and_refit(self):
        # Separate interpreters avoid loading two OpenMP runtimes into this process.
        program = r'''
import tempfile
from pathlib import Path
import numpy as np
import upstream_nonlinear as n
family = FAMILY
rng = np.random.default_rng(51)
features = rng.normal(size=(300, 6)).astype(np.float32)
z = np.column_stack([features[:, :3], np.ones(300)])
y = np.clip(3+features[:,0]*.3+features[:,1]**2*.2+rng.normal(0,.2,300),1,5)
groups = np.tile(np.arange(10), 30)
cfg = dict(n.CONFIG, seeds=[7], tree_rounds=8, tree_patience=3, tree_min_leaf=5,
           tree_leaves=7, nn_epochs=4, nn_patience=2, nn_hidden=8, nn_batch=64,
           nn_lr=.01, nn_dropout=0., ridge_lambda=10)
with tempfile.TemporaryDirectory() as tmp:
    outputs = []
    records = []
    for label, target in [('before', y), ('after', y+np.where(groups//2==2, 100., 0.))]:
        grams, rhs = n.grouped_stats(z, target, groups)
        dest = Path(tmp)/label
        records.append(n.fit_fold(features,z,target,groups,grams,rhs,2,cfg,family,dest))
        outputs.append(np.fromfile(dest/'prediction.f32',dtype='<f4'))
    assert records[0]['chosen_iterations'] == records[1]['chosen_iterations']
    assert records[0]['chosen_iterations'][0] > 0, 'fixture must exercise nonlinear refitting'
    np.testing.assert_array_equal(*outputs)
    for filename in ['ridge.npz']+(['scalers.npz'] if family=='nn' else []):
        with np.load(Path(tmp)/'before'/filename) as before, np.load(Path(tmp)/'after'/filename) as after:
            for key in before: np.testing.assert_array_equal(before[key],after[key])
'''
        for family in ['tree', 'nn']:
            with self.subTest(family=family):
                result = subprocess.run([sys.executable, '-c', program.replace('FAMILY', repr(family))],
                    text=True, capture_output=True, env={**n.os.environ, 'PYTHONPATH':str(n.u.ROOT/'scripts')},
                    timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
