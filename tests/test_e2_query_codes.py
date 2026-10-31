import dataclasses
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_query_codes import masked_query_batch, parameter_hash, query_code
import masked_sequence as ms
from test_selection_objective import fixture


class QueryCodeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.d = fixture()
        self.c = ms.Config(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=1,
            context='session', time_bins=4, dropout=.1, seq_len=24, max_targets=8,
            pad_width=4, pad_rows=1, device='cpu')
        self.model = ms.SequenceModel(self.d['items'], self.c, ms.weeks(self.d)).eval()

    def test_held_rating_is_absent_from_every_tensor_and_code(self):
        first = masked_query_batch(self.d, 0, 2, 1, self.c)
        changed = dict(self.d, train_ratings=self.d['train_ratings'].copy(), probe=self.d['probe'].copy())
        changed['train_ratings'][1] = 5
        changed['probe']['rating'] = 1
        second = masked_query_batch(changed, 0, 2, 1, self.c)
        for key in first:
            if torch.is_tensor(first[key]):
                self.assertTrue(torch.equal(first[key], second[key]), key)
            else:
                self.assertEqual(first[key], second[key])
        a = query_code(self.model, first)
        b = query_code(self.model, second)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        # A visible rating remains informative, unlike the held-out one.
        changed['train_ratings'][0] = 5
        visible = query_code(self.model, masked_query_batch(changed, 0, 2, 1, self.c))
        self.assertFalse(np.array_equal(a[0], visible[0]))

    def test_padding_compatibility_decoder_reconstruction_and_frozen_weights(self):
        before = parameter_hash(self.model)
        batch = masked_query_batch(self.d, 0, 2, 1, self.c)
        first = query_code(self.model, batch)
        self.model.config = dataclasses.replace(self.c, pad_rows=64)
        second = query_code(self.model, batch)
        np.testing.assert_allclose(first[0], second[0], atol=1e-6, rtol=0)
        np.testing.assert_allclose(first[1], second[1], atol=1e-6, rtol=0)
        self.assertLessEqual(first[2], 1e-6)
        self.assertEqual(before, parameter_hash(self.model))

    def test_unobserved_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            masked_query_batch(self.d, 0, 20, 1, self.c)

    def test_sole_rating_can_be_held_out_with_empty_explicit_context(self):
        only = dict(self.d)
        keep = np.ones(len(self.d['train_items']), dtype=bool)
        keep[1:5] = False
        for name in ('train_items', 'train_days', 'train_ratings'):
            only[name] = self.d[name][keep]
        only['train_offsets'] = self.d['train_offsets']-np.r_[0, np.full(self.d['users'], 4)]
        batch = masked_query_batch(only, 0, 1, 0, self.c)
        self.assertEqual(batch['explicit_index'].numel(), 0)
        self.assertTrue(torch.all(batch['seq_stars'] == ms.MASK))
        raw, vector, error = query_code(self.model, batch)
        self.assertTrue(np.isfinite(raw).all() and np.isfinite(vector).all())
        self.assertLessEqual(error, 1e-6)


if __name__ == '__main__':
    unittest.main()
