"""Numeric, bounded-retrieval and label-isolation contracts for the P2 pilot."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import PROBE, blend_fold
from local_paths_pilot import TrainingIndex, signal_test, summarize


def fixture():
    return dict(users=4, items=4, mean=3., train_offsets=np.array([0, 2, 5, 7, 8], dtype=np.uint64),
                train_items=np.array([1, 2, 1, 2, 3, 1, 3, 4], dtype=np.uint16),
                train_days=np.array([2, 4, 1, 3, 5, 2, 4, 1], dtype=np.uint16),
                train_ratings=np.array([5, 1, 5, 1, 4, 1, 2, 3], dtype=np.uint8),
                probe=np.array([(0, 3, 6, 4, b'\0\0\0')], dtype=PROBE))


class LocalPathsTests(unittest.TestCase):
    def test_exact_sparse_paths_and_histogram(self):
        d = fixture()
        graph = TrainingIndex(d).retrieve(0, 3, 6)
        np.testing.assert_array_equal(graph['history_rows'], [1, 0])
        np.testing.assert_array_equal(graph['target_rows'], [4, 6])
        np.testing.assert_array_equal(graph['cross_rows'], [[3, 2], [-1, 5]])
        control, paths, knn = summarize(d, graph, 6)
        expected = np.zeros(125)
        expected[[3, 123, 101]] = 1 / 3
        np.testing.assert_allclose(paths[:125], expected)
        w1, w2 = 2 / 7, np.exp(-8) / 6
        self.assertAlmostEqual(knn, (w1 * 4 + w2 * 6) / (w1 + w2))
        self.assertTrue(np.isfinite(control).all())

    def test_probe_label_invariance_and_rating_independent_retrieval(self):
        d = fixture()
        first = TrainingIndex(d).retrieve(0, 3, 6, history=1, neighbors=1, candidates=1)
        features = summarize(d, first, 6)
        d['probe']['rating'] = 1
        second = TrainingIndex(d).retrieve(0, 3, 6, history=1, neighbors=1, candidates=1)
        for a, b in zip(features, summarize(d, second, 6)):
            np.testing.assert_array_equal(a, b)
        d['train_ratings'][:] = 6 - d['train_ratings']
        third = TrainingIndex(d).retrieve(0, 3, 6, history=1, neighbors=1, candidates=1)
        for key in ('history_rows', 'target_rows', 'cross_rows'):
            np.testing.assert_array_equal(first[key], third[key])
        self.assertLessEqual(third['candidate_count'], 1)

    def test_target_leakage_rejected_and_no_paths_finite(self):
        d = fixture()
        index = TrainingIndex(d)
        with self.assertRaisesRegex(ValueError, 'target'):
            index.retrieve(0, 1, 6)
        graph = index.retrieve(0, 4, 6)
        self.assertEqual(graph['cross_rows'].shape, (0, 2))
        control, paths, knn = summarize(d, graph, 6)
        self.assertTrue(np.isfinite(control).all())
        self.assertTrue(np.isfinite(knn))
        np.testing.assert_array_equal(paths, np.zeros(133))

    def test_evaluation_labels_cannot_change_selection_or_predictions(self):
        rng = np.random.default_rng(1)
        users = np.repeat(np.arange(100, dtype=np.uint32), 2)
        control, paths = rng.normal(size=(200, 3)), rng.normal(size=(200, 5))
        y = np.clip(3 + control[:, 0], 1, 5)
        report, predictions = signal_test(control, paths, y, users, np.full(200, 3.))
        y[blend_fold(users) == 0] = 6 - y[blend_fold(users) == 0]
        changed, changed_predictions = signal_test(control, paths, y, users, np.full(200, 3.))
        for name in predictions:
            np.testing.assert_array_equal(predictions[name], changed_predictions[name])
            self.assertEqual(report['models'][name]['selection_rmse'], changed['models'][name]['selection_rmse'])


if __name__ == '__main__':
    unittest.main()
