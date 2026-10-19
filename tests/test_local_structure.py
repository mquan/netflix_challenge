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
        changed = y.copy()
        changed[held] = 6 - changed[held]
        for normalization in ('none', 'batch'):
            a.local_normalization = normalization
            first, second = run(y), run(changed)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)

    def test_local_normalization_uses_fit_statistics_without_extra_parameters(self):
        torch.manual_seed(19)
        plain = NeighborEncoder(5, width=8)
        torch.manual_seed(19)
        normalized = NeighborEncoder(5, width=8, local_normalization='batch')
        self.assertEqual(sum(p.numel() for p in plain.parameters()),
                         sum(p.numel() for p in normalized.parameters()))
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, normalized.state_dict()[key], rtol=0, atol=0)
        # Fit-only calibration of small, offset coordinates must increase
        # across-query variation. Evaluation batch composition cannot alter
        # either an individual result or stored calibration statistics.
        fit = torch.randn(64, 8) * .02 + torch.arange(8).float()
        output = normalized.local_norm(fit)
        self.assertGreater(float(output.std(0).median()), .7)
        normalized.eval()
        state = {k: v.clone() for k, v in normalized.local_norm.state_dict().items()}
        single = normalized.local_norm(fit[:1])
        mixed = normalized.local_norm(torch.cat([fit[:1], fit[1:] + 100]))
        torch.testing.assert_close(single, mixed[:1])
        for key, value in state.items():
            torch.testing.assert_close(value, normalized.local_norm.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(normalized.local_norm(torch.zeros_like(fit))).all())

    def test_common_control_retains_head_initialization_and_ignores_context(self):
        torch.manual_seed(31)
        local = NeighborEncoder(5, width=8)
        torch.manual_seed(31)
        common = NeighborEncoder(5, width=8, disable_local=True)
        for key, value in local.state_dict().items():
            torch.testing.assert_close(value, common.state_dict()[key], rtol=0, atol=0)
        torch.nn.init.normal_(common.head[-1].weight)
        common.eval()
        features, baseline = torch.randn(4, 5), torch.full((4,), 3.6)
        expected = baseline + common.head(torch.cat([features, torch.zeros(4, 8)], 1)).squeeze(1)
        torch.testing.assert_close(common({}, features, baseline), expected)
        torch.testing.assert_close(common({'unused_context': torch.randn(20)}, features, baseline), expected)


if __name__ == '__main__':
    unittest.main()
