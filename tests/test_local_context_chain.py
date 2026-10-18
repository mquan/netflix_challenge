"""Paired uncertainty and strict full-probe feature-manifest contracts."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_context_chain import paired, validate_features
from train_holdout import digest


class ComparisonTests(unittest.TestCase):
    def test_cluster_interval_uses_users_and_clipped_predictions(self):
        y = np.array([1., 2., 4., 5., 3., 2.])
        users = np.array([11, 11, 13, 13, 13, 17])
        control = np.full(6, 3.)
        treatment = np.array([0., 2., 4., 6., 3., 2.])
        report = paired(y, control, treatment, users)
        self.assertEqual(report['treatment_rmse'], 0.)
        delta = -(y - control) ** 2
        cluster_sums = np.array([sum(delta[:2] - delta.mean()),
                                 sum(delta[2:5] - delta.mean()), delta[5] - delta.mean()])
        se = np.sqrt(3 / 2 * np.sum(cluster_sums ** 2)) / 6
        np.testing.assert_allclose(report['cluster_normal_95_mse_interval'],
                                   [delta.mean() - 1.96 * se, delta.mean() + 1.96 * se])
        self.assertEqual(report['users'], 3)
        self.assertFalse(report['all_folds_improve'])  # missing folds cannot pass promotion
        with self.assertRaises(ValueError):
            paired(y, control[:-1], treatment, users)
        with self.assertRaises(ValueError):
            paired(y, control, treatment * np.nan, users)

    def test_feature_manifest_rejects_partial_wrong_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = []
            for k in range(160):
                path = Path(tmp) / f'{k}.f32'
                np.array([1, 2, 3], dtype='<f4').tofile(path)
                names.append(str(path))
            manifest = dict(data_sha256='dataset', query_rows=3, full_query_rows=3,
                            settings=dict(history=32, neighbors=32, candidates=256, seed=20261013),
                            features=dict(pooled=names[:27], paths=names[27:]),
                            outputs={p: digest(p) for p in names})
            validate_features(manifest, 'dataset', 3)
            with self.assertRaisesRegex(ValueError, 'different dataset'):
                validate_features(manifest, 'other', 3)
            partial = copy.deepcopy(manifest)
            partial['query_rows'] = 2
            with self.assertRaisesRegex(ValueError, 'full probe'):
                validate_features(partial, 'dataset', 3)
            duplicate = copy.deepcopy(manifest)
            duplicate['features']['paths'][0] = names[0]
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                validate_features(duplicate, 'dataset', 3)
            np.array([1, 3, 3], dtype='<f4').tofile(names[0])
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_features(manifest, 'dataset', 3)


if __name__ == '__main__':
    unittest.main()
