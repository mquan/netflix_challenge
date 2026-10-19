"""LC5 feature contracts: relevance correspondence, padding and equivariance."""
from pathlib import Path
import sys
import subprocess
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_relevance_pilot import local_features, shuffle_relevance


def context():
    return dict(history_rating=np.array([[1, 5, 3]]),
                neighbor_rating=np.array([[5, 1]]),
                cross_rating=np.array([[[1, 1, 3], [5, 5, 3]]]),
                cross_gap=np.array([[[0, 20, 30], [10, 40, 50]]]),
                neighbor_gap=np.array([[20, 30]]), pooled=np.array([[3.5]]))


class RelevanceTests(unittest.TestCase):
    def test_tree_process_loads_without_neural_runtime(self):
        code = '''import sys
sys.path.insert(0, 'scripts')
import local_relevance_pilot
import numpy as np
import lightgbm as lgb
assert 'torch' not in sys.modules
x = np.random.default_rng(3).normal(size=(128, 4))
model = lgb.train({'objective': 'regression', 'verbosity': -1, 'num_threads': 1},
                  lgb.Dataset(x, label=x[:, 0]), num_boost_round=3)
assert np.isfinite(model.predict(x, num_threads=1)).all()
'''
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_geometry_correspondence_changes_prediction_and_uniform_is_nested(self):
        c = context()
        left = local_features(c, np.array([[1., -1., 0.]]))
        right = local_features(c, np.array([[-1., 1., 0.]]))
        self.assertGreater(abs(left[0, 0] - right[0, 0]), 1)
        np.testing.assert_allclose(local_features(c, np.ones((1, 3))),
                                   local_features(c, np.zeros((1, 3)), uniform=True), atol=1e-6)
        # No query-label argument is consumed, even if an input container has one.
        c['query_rating'] = np.array([1])
        first = local_features(c, np.array([[1., -1., 0.]]))
        c['query_rating'] = np.array([5])
        np.testing.assert_array_equal(first, local_features(c, np.array([[1., -1., 0.]])))

    def test_consistent_movie_and_neighbor_permutations_preserve_features(self):
        c = context()
        sim = np.array([[.7, -.3, .2]])
        expected = local_features(c, sim)
        perm = [2, 0, 1]
        changed = dict(c, history_rating=c['history_rating'][:, perm],
                       cross_rating=c['cross_rating'][:, :, perm], cross_gap=c['cross_gap'][:, :, perm])
        np.testing.assert_allclose(expected, local_features(changed, sim[:, perm]), atol=1e-6)
        changed = dict(c, neighbor_rating=c['neighbor_rating'][:, ::-1], neighbor_gap=c['neighbor_gap'][:, ::-1],
                       cross_rating=c['cross_rating'][:, ::-1], cross_gap=c['cross_gap'][:, ::-1])
        np.testing.assert_allclose(expected, local_features(changed, sim), atol=1e-6)

    def test_shuffle_preserves_nonpadding_multiset_and_empty_context_is_finite(self):
        sim = np.array([[.1, 0, .3, .7], [.2, .3, 0, 0], [0, 0, 0, 0]])
        valid = np.array([[True, False, True, True], [True, True, False, False], [False] * 4])
        shuffled = shuffle_relevance(sim, valid)
        np.testing.assert_array_equal(shuffled[~valid], sim[~valid])
        for i in range(len(sim)):
            np.testing.assert_array_equal(np.sort(sim[i, valid[i]]), np.sort(shuffled[i, valid[i]]))
        np.testing.assert_array_equal(shuffled, shuffle_relevance(sim, valid))
        c = context()
        c['cross_rating'][:] = 0
        values = local_features(c, np.array([[1., -1., 0.]]))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[0, 0], c['pooled'][0, 0])


if __name__ == '__main__':
    unittest.main()
