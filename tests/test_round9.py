"""Leakage and selection contracts for round 9 feature and history experiments."""
import dataclasses
import pathlib
import sys
import unittest

import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from masked_sequence import Config, window
from round9_latent import features
from test_masked_sequence import MaskedSequenceTests, long_dataset


class RetrievalTests(MaskedSequenceTests):
    def setUp(self):
        super().setUp()
        # All fixtures use <=40 movies; individual tests use their own configs,
        # so explicitly exercise retrieval in a dedicated input isolation test.

    def test_retrieval_masks_hidden_values(self):
        d = long_dataset()
        table = np.random.default_rng(0).normal(size=(d['items'], d['items'])).astype('f4')
        path = self.root/'similarity.npy'; np.save(path, table)
        c = dataclasses.replace(self.c, retrieval_table=str(path), retrieval_frac=.5)
        from masked_sequence import make_keep, prepare_batch
        users = np.arange(d['users'])
        changed, _ = self.flip_hidden(d, make_keep(d, users, c, np.random.default_rng(5)), users)
        changed['probe'] = d['probe'].copy(); changed['probe']['rating'] = 6-changed['probe']['rating']
        a = prepare_batch(d, users, c, rng=np.random.default_rng(5))
        b = prepare_batch(changed, users, c, rng=np.random.default_rng(5))
        for key in ('seq_flat','seq_items','seq_item_star','seq_stars','seq_weeks','seq_features'):
            np.testing.assert_array_equal(a[key].numpy(), b[key].numpy())
        self.assertLessEqual(int(a['seq_lengths'].max()), c.seq_len)

    def test_retrieval_reserves_old_relevant_items(self):
        c = Config(seq_len=9, max_targets=1, retrieval_frac=.5)
        rows = np.zeros(20, dtype=int); days = np.arange(20)
        score = np.zeros(20); score[:3] = [3,2,1]
        chosen = window(rows, days, np.zeros(20), np.arange(20)[::-1], 1, np.array([1]), c, score)
        self.assertTrue(chosen[:3].all())
        self.assertEqual(chosen.sum(), 7)
        self.assertTrue(chosen[-2:].all())


class LatentTests(unittest.TestCase):
    def test_features_ignore_all_rating_values(self):
        d = long_dataset(); rng = np.random.default_rng(42); rank = 3
        sessions = sum(len(np.unique(d['implicit_days'][int(a):int(b)])) for a,b in zip(d['implicit_offsets'][:-1],d['implicit_offsets'][1:]))
        m = {key:rng.normal(size=size).astype('f4') for key,size in dict(
            user_bias=d['users'], user_alpha=d['users'], user_scale=d['users'],
            item_bias=d['items'], item_bin_bias=d['items']*30, item_frequency_bias=d['items']*8,
            day_bias=sessions, day_scale=sessions, user_factors=(d['users'],rank),
            item_factors=(d['items'],rank), implicit_factors=(d['items'],rank), drift=(d['users'],rank)).items()}
        a, _, _ = features(d,m,2)
        changed = dict(d); changed['probe'] = d['probe'].copy()
        changed['probe']['rating'] = 6-changed['probe']['rating']
        changed['train_ratings'] = 6-d['train_ratings']
        b, _, _ = features(changed,m,2)
        for name in a: np.testing.assert_array_equal(a[name],b[name])


if __name__ == '__main__': unittest.main()
