"""Contracts for hidden-target inputs, true learning, and final-probe isolation."""
import dataclasses
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import load
from masked_history import Config, HistoryModel, fit, loss_value, parser, predict, prepare, run
from test_joint import dataset
from train_holdout import create


class MaskedHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        dataset(self.root / 'data.bin', correlated_train=True)
        self.d = load(self.root / 'data.bin')
        self.c = Config(rank=4, hidden=8, dropout=0., batch_users=25, nnz_budget=80)

    def tearDown(self):
        self.temp.cleanup()

    def test_masked_values_cannot_change_any_input_or_prediction(self):
        users = np.array([0, 1, 2])
        keep = np.array([True, False, True] * 3)
        changed = dict(self.d)
        changed['train_ratings'] = self.d['train_ratings'].copy()
        changed['train_ratings'][[1, 4, 7]] = 6 - changed['train_ratings'][[1, 4, 7]]
        for representation in ('scalar', 'categorical'):
            c = dataclasses.replace(self.c, representation=representation)
            a, b = prepare(self.d, users, c, keep=keep), prepare(changed, users, c, keep=keep)
            for key in a:
                if key != 'target_ratings':
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertFalse(torch.equal(a['target_ratings'], b['target_ratings']))
            model = HistoryModel(self.d['items'], c).eval()
            pa = model.decode(model.encode(a), a['features'], a['target_rows'], a['target_items'])
            pb = model.decode(model.encode(b), b['features'], b['target_rows'], b['target_items'])
            self.assertTrue(torch.equal(pa, pb))

    def test_losses_learn_and_persist_with_zero_context(self):
        for head, loss in [('scalar', 'mse'), ('distribution', 'mse'),
                           ('distribution', 'ce'), ('distribution', 'ordinal')]:
            c = dataclasses.replace(self.c, head=head, loss=loss)
            torch.manual_seed(c.seed)
            model = HistoryModel(self.d['items'], c)
            b = prepare(self.d, np.array([1, 2]), c, keep=np.zeros(6, dtype=bool))
            raw = model.decode(model.encode(b), b['features'], b['target_rows'], b['target_items'])
            value = loss_value(model, raw, b['target_ratings'])
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertGreater(float(model.output.grad.abs().sum()), 0)
            model.initialize_bias(self.d)
            initial = predict(model, self.d)
            trained, records, epoch = fit(self.d, c, 2, self.root / (head + loss))
            self.assertEqual(epoch, 2)
            self.assertEqual(len(records), 2)
            p = predict(trained, self.d)
            self.assertGreater(float(np.max(np.abs(p - initial))), 1e-6)
            saved = torch.load(self.root / (head + loss + '.pt'), weights_only=True)
            reloaded = HistoryModel(self.d['items'], c)
            reloaded.load_state_dict(saved['state'])
            np.testing.assert_array_equal(p, predict(reloaded, self.d))
            reversed_queries = self.d['probe'][::-1].copy()
            np.testing.assert_allclose(p[::-1], predict(reloaded, self.d, reversed_queries), atol=1e-6)

    def test_full_selection_and_refit_ignore_original_probe_labels(self):
        dataset(self.root / 'changed.bin', correlated_train=True, change_fold=0)
        create(self.root / 'data.bin', self.root / 'holdout.bin', .014, 7)
        reports = []
        for filename in ('data', 'changed'):
            args = parser().parse_args([
                '--data', str(self.root / (filename + '.bin')),
                '--selection-data', str(self.root / 'holdout.bin'),
                '--output', str(self.root / (filename + '-run')),
                '--epochs', '2', '--threads', '1', '--rank', '4', '--hidden', '8',
                '--batch-users', '25', '--nnz-budget', '80', '--dropout', '0'])
            reports.append(run(args))
        self.assertEqual(reports[0]['selected_epoch'], reports[1]['selected_epoch'])
        self.assertEqual((self.root / 'data-run.f32').read_bytes(),
                         (self.root / 'changed-run.f32').read_bytes())
        self.assertNotEqual(reports[0]['probe_rmse'], reports[1]['probe_rmse'])
        a = torch.load(self.root / 'data-run.refit.pt', weights_only=True)['state']
        b = torch.load(self.root / 'changed-run.refit.pt', weights_only=True)['state']
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_session_variant_isolates_hidden_values_and_probe_labels(self):
        c = dataclasses.replace(self.c, context='session', suffix_prob=.5, time_bins=4, week_days=7)
        users = np.array([0, 1, 2])
        keep = np.array([True, False, True] * 3)
        changed = dict(self.d)
        changed['train_ratings'] = self.d['train_ratings'].copy()
        changed['train_ratings'][[1, 4, 7]] = 6 - changed['train_ratings'][[1, 4, 7]]
        a, b = prepare(self.d, users, c, keep=keep), prepare(changed, users, c, keep=keep)
        for key in a:
            if key != 'target_ratings':
                self.assertTrue(torch.equal(a[key], b[key]), key)
        torch.manual_seed(1)
        model = HistoryModel(self.d['items'], c)
        with torch.no_grad():
            model.session_net[-1].weight.normal_()
        model.eval()
        pa = model.decode(model.encode(a), a['features'], a['target_rows'], a['target_items'], a)
        pb = model.decode(model.encode(b), b['features'], b['target_rows'], b['target_items'], b)
        self.assertTrue(torch.equal(pa, pb))
        # Suffix masking decisions depend on dates and the RNG, not ratings.
        ka = prepare(self.d, users, c, rng=np.random.default_rng(3))['target_items']
        kb = prepare(changed, users, c, rng=np.random.default_rng(3))['target_items']
        self.assertTrue(torch.equal(ka, kb))
        dataset(self.root / 'changed.bin', correlated_train=True, change_fold=0)
        create(self.root / 'data.bin', self.root / 'holdout.bin', .014, 7)
        reports = []
        for filename in ('data', 'changed'):
            args = parser().parse_args([
                '--data', str(self.root / (filename + '.bin')),
                '--selection-data', str(self.root / 'holdout.bin'),
                '--output', str(self.root / (filename + '-session')),
                '--epochs', '2', '--threads', '1', '--rank', '4', '--hidden', '8',
                '--batch-users', '25', '--nnz-budget', '80', '--dropout', '0',
                '--context', 'session', '--suffix-prob', '.5', '--time-bins', '4', '--week-days', '7',
                '--save-epochs'])
            reports.append(run(args))
        self.assertEqual((self.root / 'data-session.f32').read_bytes(),
                         (self.root / 'changed-session.f32').read_bytes())
        self.assertNotEqual(reports[0]['probe_rmse'], reports[1]['probe_rmse'])
        epoch = reports[0]['selected_epoch']
        self.assertEqual((self.root / f'data-session.refit.e{epoch}.f32').read_bytes(),
                         (self.root / 'data-session.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
