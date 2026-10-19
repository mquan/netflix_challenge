"""Contracts for matched structural training, shuffling, and held-label isolation."""
import argparse
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_paths_pilot import TrainingIndex
from local_structure_data import FIELDS, encode_query, shuffle_correspondence
from local_structure_train import NeighborEncoder, fit_preprocessing, predict, train
from test_local_paths_pilot import fixture


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shuffle_preserves_mask_degrees_and_paired_movie_payloads(self):
        ratings = np.array([[1, 0, 2, 5], [2, 3, 0, 1], [5, 2, 3, 4], [0, 5, 1, 3]], dtype='u1')
        gaps = (np.arange(16).reshape(4, 4) * 7 * (ratings > 0)).astype('<i2')
        r, g = shuffle_correspondence(ratings, gaps, 1)
        np.testing.assert_array_equal(r > 0, ratings > 0)
        self.assertTrue(np.any(r != ratings))
        for col in range(4):
            self.assertEqual(sorted(zip(ratings[:, col], gaps[:, col])), sorted(zip(r[:, col], g[:, col])))
        r2, g2 = shuffle_correspondence(ratings, gaps, 1)
        np.testing.assert_array_equal(r, r2)
        np.testing.assert_array_equal(g, g2)

    def test_export_numeric_correspondence_and_query_label_invariance(self):
        d = fixture()
        result, diagnostics = encode_query(d, TrainingIndex(d), 0, 20261013)
        np.testing.assert_array_equal(result['history_rating'][:2], [1, 5])
        np.testing.assert_array_equal(result['neighbor_rating'][:2], [4, 2])
        np.testing.assert_array_equal(result['cross_rating'][:2, :2], [[1, 5], [0, 1]])
        np.testing.assert_array_equal(diagnostics[1:4], [0, 0, 0])
        d['probe']['rating'] = 1
        changed, other = encode_query(d, TrainingIndex(d), 0, 20261013)
        for key in result:
            np.testing.assert_array_equal(result[key], changed[key])
        np.testing.assert_array_equal(diagnostics, other)

    def test_pooling_erases_grouping_while_structure_can_use_it(self):
        torch.manual_seed(8)
        tokens, mask, meta = torch.randn(2, 3, 4, 9), torch.ones(2, 3, 4), torch.randn(2, 3, 3)
        regrouped = tokens.reshape(2, 12, 9).roll(2, dims=1).reshape_as(tokens)
        model = NeighborEncoder(5, width=8, pooled=True)
        original = model.local_summary(tokens, mask, meta)
        torch.testing.assert_close(original, model.local_summary(regrouped, mask, meta))
        pooled_parameters = sum(p.numel() for p in model.parameters())
        model.pooled = False
        self.assertEqual(pooled_parameters, sum(p.numel() for p in model.parameters()))
        original = model.local_summary(tokens, mask, meta)
        self.assertGreater(float((original - model.local_summary(regrouped, mask, meta)).abs().max().detach()), 1e-6)
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(original, model.local_summary(tokens[:, permutation], mask[:, permutation], meta[:, permutation]))
        movies = torch.tensor([3, 0, 2, 1])
        torch.testing.assert_close(original, model.local_summary(tokens[:, :, movies], mask[:, :, movies], meta))
        self.assertTrue(torch.isfinite(model.local_summary(tokens, mask * 0, meta)).all())

    def test_held_labels_cannot_change_preprocessing_epoch_choice_or_predictions(self):
        rng = np.random.default_rng(15)
        n = 30
        predictions, global_features = rng.normal(3, .3, (n, 2)), rng.normal(size=(n, 3)).astype('f4')
        y = np.clip(predictions[:, 0] + .2 * global_features[:, 0], 1, 5)
        arrays = {key: np.zeros((n, *shape), dtype=dtype) for key, (dtype, shape) in FIELDS.items()}
        for key in ('history_rating', 'neighbor_rating', 'cross_rating', 'shuffled_rating'):
            arrays[key][:] = rng.integers(1, 6, size=arrays[key].shape)
        fit, selection, held = np.arange(15), np.arange(15, 23), np.arange(23, 30)
        a = argparse.Namespace(width=4, batch=8, lr=.001, device='cpu', patience=2)
        def run(labels):
            x, base, prep = fit_preprocessing(predictions, global_features, labels, fit)
            _, chosen, history = train(arrays, x, base, labels, fit, a=a, arm='structured', seed=3,
                                        epochs=2, validation=selection)
            refit = np.r_[fit, selection]
            rx, rb, _ = fit_preprocessing(predictions, global_features, labels, refit)
            model, _, _ = train(arrays, rx, rb, labels, refit, a=a, arm='structured', seed=3, epochs=chosen)
            return x, base, chosen, [r['selection_rmse'] for r in history], predict(model, arrays, rx, rb, held, a, False)
        first = run(y)
        changed = y.copy()
        changed[held] = 6 - changed[held]
        second = run(changed)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)


if __name__ == '__main__':
    unittest.main()
