"""Contracts for the item-side masked-history twin: hidden-value isolation,
true learning, and final-probe isolation."""
import dataclasses
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import load
from masked_history import loss_value
from masked_item_history import (ItemConfig, ItemHistoryModel, fit, parser, predict, prepare, run,
                                 transpose)
from test_joint import dataset
from train_holdout import create


class MaskedItemHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        dataset(self.root / 'data.bin', correlated_train=True)
        self.d = load(self.root / 'data.bin')
        self.t = transpose(self.d)
        self.c = ItemConfig(rank=4, hidden=8, dropout=0., batch_items=2, nnz_budget=150)

    def tearDown(self):
        self.temp.cleanup()

    def test_transpose_preserves_every_rating_and_interaction(self):
        d, t = self.d, self.t
        owner = np.repeat(np.arange(d['users']), np.diff(d['train_offsets'].astype(np.int64)))
        source = sorted(zip(d['train_items'] - 1, owner, d['train_days'], d['train_ratings']))
        movie = np.repeat(np.arange(d['items']), np.diff(t['train_offsets']))
        moved = sorted(zip(movie, t['train_items'] - 1, t['train_days'], t['train_ratings']))
        self.assertEqual(source, moved)
        self.assertEqual(int(t['implicit_offsets'][-1]), len(d['implicit_items']))
        movie = np.repeat(np.arange(d['items']), np.diff(t['implicit_offsets']))
        self.assertTrue((np.bincount(movie, minlength=d['items']) ==
                         np.bincount(d['implicit_items'] - 1, minlength=d['items'])).all())

    def changed(self, positions):
        changed = dict(self.t)
        changed['train_ratings'] = self.t['train_ratings'].copy()
        changed['train_ratings'][positions] = 6 - changed['train_ratings'][positions]
        return changed

    def test_masked_values_cannot_change_any_input_or_prediction(self):
        movies = np.array([0, 1, 2])
        keep = np.tile([True, False, True], 100)
        changed = self.changed(np.flatnonzero(~keep))
        for c in (self.c, dataclasses.replace(self.c, context='session', suffix_prob=.5, time_bins=4)):
            a, b = prepare(self.t, movies, c, keep=keep), prepare(changed, movies, c, keep=keep)
            for key in a:
                if key != 'target_ratings':
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertFalse(torch.equal(a['target_ratings'], b['target_ratings']))
            torch.manual_seed(1)
            model = ItemHistoryModel(self.d['users'], self.d['items'], c)
            if c.context == 'session':
                with torch.no_grad():
                    model.session_net[-1].weight.normal_()
            model.eval()
            self.assertTrue(torch.equal(model.decode(model.encode(a), a), model.decode(model.encode(b), b)))
        # Suffix masking decisions depend on dates and the RNG, not ratings.
        c = dataclasses.replace(self.c, suffix_prob=.5)
        everything = self.changed(np.arange(len(self.t['train_ratings'])))
        ka = prepare(self.t, movies, c, rng=np.random.default_rng(3))['target_users']
        kb = prepare(everything, movies, c, rng=np.random.default_rng(3))['target_users']
        self.assertTrue(torch.equal(ka, kb))

    def test_losses_learn_and_persist_with_zero_context(self):
        for head, loss, context in [('scalar', 'mse', 'none'), ('distribution', 'mse', 'session'),
                                    ('distribution', 'ce', 'none'), ('distribution', 'ordinal', 'none')]:
            c = dataclasses.replace(self.c, head=head, loss=loss, context=context,
                                    time_bins=4 if context == 'session' else 0)
            torch.manual_seed(c.seed)
            model = ItemHistoryModel(self.d['users'], self.d['items'], c)
            b = prepare(self.t, np.array([1, 2]), c, keep=np.zeros(200, dtype=bool))
            raw = model.decode(model.encode(b), b)
            value = loss_value(model, raw, b['target_ratings'])
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertGreater(float(model.output.grad.abs().sum()), 0)
            model.initialize_bias(self.t)
            initial = predict(model, self.t)
            name = head + loss + context
            trained, records, epoch = fit(self.t, c, 2, self.root / name)
            self.assertEqual(epoch, 2)
            self.assertEqual(len(records), 2)
            p = predict(trained, self.t)
            self.assertGreater(float(np.max(np.abs(p - initial))), 1e-6)
            saved = torch.load(self.root / (name + '.pt'), weights_only=True)
            reloaded = ItemHistoryModel(self.d['users'], self.d['items'], c)
            reloaded.load_state_dict(saved['state'])
            np.testing.assert_array_equal(p, predict(reloaded, self.t))
            reversed_queries = self.t['probe'][::-1].copy()
            np.testing.assert_allclose(p[::-1], predict(reloaded, self.t, reversed_queries), atol=1e-6)

    def test_full_selection_and_refit_ignore_original_probe_labels(self):
        dataset(self.root / 'changed.bin', correlated_train=True, change_fold=0)
        create(self.root / 'data.bin', self.root / 'holdout.bin', .014, 7)
        reports = []
        for filename in ('data', 'changed'):
            args = parser().parse_args([
                '--data', str(self.root / (filename + '.bin')),
                '--selection-data', str(self.root / 'holdout.bin'),
                '--output', str(self.root / (filename + '-item')),
                '--epochs', '2', '--threads', '1', '--rank', '4', '--hidden', '8',
                '--batch-items', '2', '--nnz-budget', '150', '--dropout', '0', '--loss', 'mse',
                '--context', 'session', '--suffix-prob', '.5', '--time-bins', '4', '--save-epochs'])
            reports.append(run(args))
        self.assertEqual(reports[0]['selected_epoch'], reports[1]['selected_epoch'])
        self.assertEqual((self.root / 'data-item.f32').read_bytes(),
                         (self.root / 'changed-item.f32').read_bytes())
        self.assertNotEqual(reports[0]['probe_rmse'], reports[1]['probe_rmse'])
        a = torch.load(self.root / 'data-item.refit.pt', weights_only=True)['state']
        b = torch.load(self.root / 'changed-item.refit.pt', weights_only=True)['state']
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        epoch = reports[0]['selected_epoch']
        self.assertEqual((self.root / f'data-item.refit.e{epoch}.f32').read_bytes(),
                         (self.root / 'data-item.f32').read_bytes())
        with self.assertRaises(FileExistsError):
            run(args)


if __name__ == '__main__':
    unittest.main()
