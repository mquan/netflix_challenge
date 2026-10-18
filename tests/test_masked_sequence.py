"""Contracts for the masked sequence encoder: hidden-target and probe-label
isolation, windowing of long histories, learning, reload and query order."""
import dataclasses
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import PROBE, load
from masked_sequence import (Config, MASK, SequenceModel, fit, make_keep, parser, predict,
                             prepare_batch, run, select_targets, sequence_inputs, weeks)
from test_joint import dataset
from train_holdout import create


def long_dataset(seed=0, users=6, items=40):
    """In-memory dataset with long, date-sorted histories (one user > seq_len)."""
    rng = np.random.default_rng(seed)
    lengths = np.array([3, 30, 1, 12, 200, 25])[:users]
    t_items, t_days, t_ratings, i_items, i_days = [], [], [], [], []
    probe = []
    for u, n in enumerate(lengths):
        chosen = rng.choice(items, size=min(n, items), replace=False) + 1
        chosen = np.resize(chosen, n)
        days = np.sort(rng.integers(0, 300, n))
        t_items.append(chosen); t_days.append(days); t_ratings.append(rng.integers(1, 6, n))
        extra = rng.choice(items, size=2, replace=False) + 1
        for item in extra:
            probe.append((u, item, 300 + u, rng.integers(1, 6)))
        i_items.append(np.r_[chosen, extra]); i_days.append(np.r_[days, [300 + u] * 2])
    offsets = np.r_[0, np.cumsum(lengths)].astype('<u8')
    imp_offsets = np.r_[0, np.cumsum([len(x) for x in i_items])].astype('<u8')
    p = np.zeros(len(probe), dtype=PROBE)
    for k, (u, i, day, r) in enumerate(probe):
        p[k]['user'], p[k]['item'], p[k]['day'], p[k]['rating'] = u, i, day, r
    return dict(users=users, items=items, mean=3.5, min_day=0, max_day=310,
                train_offsets=offsets, train_items=np.concatenate(t_items).astype('<u2'),
                train_days=np.concatenate(t_days).astype('<u2'),
                train_ratings=np.concatenate(t_ratings).astype('u1'), probe=p,
                implicit_offsets=imp_offsets, implicit_items=np.concatenate(i_items).astype('<u2'),
                implicit_days=np.concatenate(i_days).astype('<u2'))


class MaskedSequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        dataset(self.root / 'data.bin', correlated_train=True)
        self.d = load(self.root / 'data.bin')
        self.c = Config(rank=4, hidden=8, dropout=0., batch_users=25, nnz_budget=80, d_model=8,
                        heads=2, ff=16, seq_len=24, max_targets=8, batch_tokens=256, time_bins=4)

    def tearDown(self):
        self.temp.cleanup()

    def model(self, d, c, seed=1):
        torch.manual_seed(seed)
        m = SequenceModel(d['items'], c, weeks(d))
        with torch.no_grad():  # make the session shift nonzero so it is exercised
            m.session_net[-1].weight.normal_()
        return m.eval()

    def flip_hidden(self, d, keep_of_users, users):
        _, index, _, keep = keep_of_users
        changed = dict(d)
        changed['train_ratings'] = d['train_ratings'].copy()
        hidden = index[~keep]
        changed['train_ratings'][hidden] = 6 - changed['train_ratings'][hidden]
        return changed, hidden

    def test_hidden_training_ratings_never_enter_inputs_or_predictions(self):
        d = long_dataset()
        c = dataclasses.replace(self.c, week_days=7)
        users = np.arange(d['users'])
        changed, hidden = self.flip_hidden(d, make_keep(d, users, c, np.random.default_rng(5)), users)
        self.assertGreater(len(hidden), 0)
        self.assertFalse(np.array_equal(changed['train_ratings'], d['train_ratings']))
        a = prepare_batch(d, users, c, rng=np.random.default_rng(5))
        b = prepare_batch(changed, users, c, rng=np.random.default_rng(5))
        self.assertGreater(a['target_ratings'].numel(), 0)
        for key in a:
            if key == 'target_ratings':
                continue
            if torch.is_tensor(a[key]):
                self.assertTrue(torch.equal(a[key], b[key]), key)
            else:
                self.assertEqual(a[key], b[key], key)
        self.assertFalse(torch.equal(a['target_ratings'], b['target_ratings']))
        m = self.model(d, c)
        self.assertTrue(torch.equal(m.forward_batch(a), m.forward_batch(b)))
        # Targets are [MASK] tokens; every target appears in the sequence.
        stars = a['seq_stars'].numpy()
        self.assertEqual(int((stars == MASK).sum()), a['target_ratings'].numel())
        flat = a['seq_flat'].numpy()
        self.assertTrue(np.isin(a['seq_target_flat'].numpy(), flat[stars == MASK]).all())

    def test_probe_labels_never_enter_inputs_or_predictions(self):
        d = long_dataset()
        changed = dict(d)
        changed['probe'] = d['probe'].copy()
        changed['probe']['rating'] = 6 - changed['probe']['rating']
        m = self.model(d, self.c)
        np.testing.assert_array_equal(predict(m, d), predict(m, changed))
        # Query order does not matter (hash-keyed windows and ties).
        p = predict(m, d)
        np.testing.assert_allclose(p[::-1], predict(m, d, d['probe'][::-1].copy()), atol=1e-6)

    def test_long_histories_are_windowed_with_all_targets_kept(self):
        d = long_dataset()
        c = self.c
        users = np.arange(d['users'])
        rows, index, offsets, keep = make_keep(d, users, c, np.random.default_rng(2))
        chosen = select_targets(rows, keep, c, np.random.default_rng(3))
        per_row = np.bincount(rows[chosen], minlength=len(users))
        self.assertLessEqual(per_row.max(), c.max_targets)
        hidden_per_row = np.bincount(rows[~keep], minlength=len(users))
        np.testing.assert_array_equal(per_row, np.minimum(hidden_per_row, c.max_targets))
        t_rows = rows[chosen]
        t_items = d['train_items'][index[chosen]].astype(np.int64) - 1
        t_days = d['train_days'][index[chosen]].astype(np.int64)
        s, width = sequence_inputs(d, users, rows, index, keep, t_rows, t_items, t_days, c,
                                   np.random.default_rng(4))
        lengths = s['seq_lengths'].numpy()
        self.assertLessEqual(lengths.max(), c.seq_len)
        self.assertGreaterEqual(width, lengths.max())
        self.assertEqual(width % c.pad_width, 0)
        long_row = int(np.argmax(np.diff(offsets)))
        self.assertEqual(lengths[long_row], c.seq_len)
        # All targets present, and the window keeps the most recent visible ratings.
        self.assertEqual(int((s['seq_stars'].numpy() == MASK).sum()), len(chosen))
        flat = s['seq_flat'].numpy()
        row_of = flat // width
        seq_days = s['seq_weeks'].numpy()  # weeks preserve date order
        for r in range(len(users)):
            w = seq_days[row_of == r]
            self.assertTrue(np.all(np.diff(w) >= 0), 'tokens must be date ordered')
        visible_days = d['train_days'][index[keep & (rows == long_row)]].astype(np.int64)
        latest = np.sort(visible_days)[-4:] // 7
        mine = seq_days[(row_of == long_row) & (s['seq_stars'].numpy() != MASK)]
        self.assertTrue(np.isin(latest, mine).all())
        # Prediction-time windows are deterministic.
        q = (np.array([long_row]), np.array([0]), np.array([305]))
        x = prepare_batch(d, np.array([long_row]), c, queries=(np.zeros(1, dtype=np.int64), q[1], q[2]))
        y = prepare_batch(d, np.array([long_row]), c, queries=(np.zeros(1, dtype=np.int64), q[1], q[2]))
        self.assertTrue(torch.equal(x['seq_items'], y['seq_items']))
        self.assertEqual(int(x['seq_lengths'][0]), c.seq_len)

    def test_learns_persists_and_reloads(self):
        c = self.c
        torch.manual_seed(c.seed)
        m = SequenceModel(self.d['items'], c, weeks(self.d))
        m.initialize_bias(self.d)
        initial = predict(m, self.d)
        trained, records, epoch = fit(self.d, c, 2, self.root / 'seq')
        self.assertEqual((epoch, len(records)), (2, 2))
        p = predict(trained, self.d)
        self.assertGreater(float(np.max(np.abs(p - initial))), 1e-6)
        saved = torch.load(self.root / 'seq.pt', weights_only=True)
        reloaded = SequenceModel(self.d['items'], c, saved['weeks'])
        reloaded.load_state_dict(saved['state'])
        np.testing.assert_array_equal(p, predict(reloaded, self.d))
        self.assertFalse(hasattr(reloaded, 'explicit'))

    def test_selection_and_refit_ignore_original_probe_labels(self):
        dataset(self.root / 'changed.bin', correlated_train=True, change_fold=0)
        create(self.root / 'data.bin', self.root / 'holdout.bin', .014, 7)
        reports = []
        for filename in ('data', 'changed'):
            args = parser().parse_args([
                '--data', str(self.root / (filename + '.bin')),
                '--selection-data', str(self.root / 'holdout.bin'),
                '--output', str(self.root / (filename + '-seq')),
                '--epochs', '2', '--threads', '1', '--rank', '4', '--hidden', '8',
                '--batch-users', '25', '--nnz-budget', '80', '--dropout', '0',
                '--d-model', '8', '--heads', '2', '--ff', '16', '--seq-len', '24',
                '--max-targets', '8', '--batch-tokens', '256', '--time-bins', '4',
                '--week-days', '7', '--save-epochs'])
            reports.append(run(args))
        self.assertEqual(reports[0]['selected_epoch'], reports[1]['selected_epoch'])
        self.assertEqual((self.root / 'data-seq.f32').read_bytes(),
                         (self.root / 'changed-seq.f32').read_bytes())
        self.assertNotEqual(reports[0]['probe_rmse'], reports[1]['probe_rmse'])
        a = torch.load(self.root / 'data-seq.refit.pt', weights_only=True)['state']
        b = torch.load(self.root / 'changed-seq.refit.pt', weights_only=True)['state']
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        epoch = reports[0]['selected_epoch']
        self.assertEqual((self.root / f'data-seq.refit.e{epoch}.f32').read_bytes(),
                         (self.root / 'data-seq.f32').read_bytes())
        with self.assertRaises(FileExistsError):
            run(args)


if __name__ == '__main__':
    unittest.main()
