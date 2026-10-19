"""Choice-task isolation, matched-arm randomness, gradient and warm-start contracts."""
import dataclasses
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import HEADER, PROBE
import masked_sequence as ms
from selection_objective import SelectionModel, auxiliary_batch, parser, run, sample_choices, train_epoch, warm_model
import masked_history as mh
import json


def fixture():
    users, items = 6, 20
    train = np.concatenate([(np.arange(5) + 2 * user) % items + 1 for user in range(users)])
    implicit = np.concatenate([(np.arange(7) + 2 * user) % items + 1 for user in range(users)])
    probe = np.zeros(users * 2, dtype=PROBE)
    probe['user'] = np.repeat(np.arange(users), 2)
    probe['item'] = np.concatenate([(np.arange(5, 7) + 2 * user) % items + 1 for user in range(users)])
    probe['day'] = np.tile([5, 6], users)
    probe['rating'] = 3
    return dict(users=users, items=items, min_day=0, max_day=6, mean=3.,
                train_offsets=np.arange(users + 1) * 5, train_items=train.astype(np.uint16),
                train_days=np.tile(np.arange(5), users).astype(np.uint16),
                train_ratings=np.tile([1, 2, 3, 4, 5], users).astype(np.uint8),
                implicit_offsets=np.arange(users + 1) * 7, implicit_items=implicit.astype(np.uint16),
                implicit_days=np.tile(np.arange(7), users).astype(np.uint16), probe=probe)


class SelectionObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.d = fixture()
        self.c = ms.Config(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=1,
                           context='session', time_bins=4, dropout=.1, batch_users=3,
                           batch_tokens=96, seq_len=24, max_targets=8, pad_width=4, pad_rows=2)

    def model(self):
        torch.manual_seed(11)
        return SelectionModel(self.d['items'], self.c, ms.weeks(self.d))

    def assert_batch_equal(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            if torch.is_tensor(a[key]):
                self.assertTrue(torch.equal(a[key], b[key]), key)
            else:
                self.assertEqual(a[key], b[key], key)

    def test_positive_ids_and_values_are_removed_from_all_encoder_paths(self):
        users = np.arange(self.d['users'])
        b = auxiliary_batch(self.d, users, self.c, np.random.default_rng(9), 3, 5)
        positive = b['choice_labels'] == 1
        keys = b['choice_rows'][positive].numpy() * self.d['items'] + b['choice_items'][positive].numpy()
        seq_rows = b['seq_flat'].numpy() // b['seq_width']
        self.assertFalse(np.isin(seq_rows * self.d['items'] + b['seq_items'].numpy(), keys).any())
        imp_rows = np.repeat(users, np.diff(b['implicit_offsets'].numpy()))
        self.assertFalse(np.isin(imp_rows * self.d['items'] + b['implicit_index'].numpy(), keys).any())
        explicit_rows = np.repeat(users, np.diff(b['explicit_offsets'].numpy()))
        self.assertFalse(np.isin(explicit_rows * self.d['items'] + b['explicit_index'].numpy() // 5, keys).any())
        self.assertFalse((b['seq_stars'] == ms.MASK).any())
        np.testing.assert_array_equal(b['seq_target_flat'], users * b['seq_width'])
        # A held-out positive's rating/date cannot change auxiliary inputs.
        changed = dict(self.d)
        train_rows = np.repeat(users, 5)
        held = np.isin(train_rows * self.d['items'] + self.d['train_items'] - 1, keys)
        self.assertTrue(held.any())
        for field in ('train_ratings', 'train_days'):
            changed[field] = self.d[field].copy()
            changed[field][held] = 4
        ib = np.repeat(users, 7)
        held_implicit = np.isin(ib * self.d['items'] + self.d['implicit_items'] - 1, keys)
        changed['implicit_days'] = self.d['implicit_days'].copy()
        changed['implicit_days'][held_implicit] = 4
        self.assert_batch_equal(b, auxiliary_batch(changed, users, self.c, np.random.default_rng(9), 3, 5))

    def test_negative_sampling_never_calls_an_observed_movie_unobserved(self):
        t = sample_choices(self.d, np.arange(6), np.random.default_rng(4), 3, 100)
        for user in range(6):
            row = t['rows'] == user
            known = self.d['implicit_items'][user * 7:(user + 1) * 7] - 1
            self.assertTrue(np.isin(t['items'][row & (t['labels'] == 1)], known).all())
            self.assertFalse(np.isin(t['items'][row & (t['labels'] == 0)], known).any())
            self.assertAlmostEqual(float(t['weights'][row].sum()), 1 / 6, places=6)
        self.assertAlmostEqual(float(t['weights'].sum()), 1., places=6)
        full = dict(self.d, items=7, users=1, implicit_offsets=np.array([0, 7]))
        self.assertEqual(len(sample_choices(full, np.array([0]), np.random.default_rng(4))['items']), 0)

    def test_user_only_context_and_choice_gradients(self):
        c = dataclasses.replace(self.c, input_drop=.999999, dropout=0.)
        b = auxiliary_batch(self.d, np.arange(6), c, np.random.default_rng(1), 7, 3)
        self.assertEqual(len(b['seq_items']), 0)
        self.assertEqual(len(b['implicit_index']), 0)
        model = self.model()
        model.eval()
        self.assertTrue(torch.isfinite(model.choice_loss(b)))
        b = auxiliary_batch(self.d, np.arange(6), self.c, np.random.default_rng(2), 2, 3)
        before = model.choice_loss(b)
        before.backward()
        for parameter in (model.choice_output, model.code_out.weight, model.blocks[0].qkv.weight, model.implicit):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        for _ in range(10):
            optimizer.zero_grad()
            loss = model.choice_loss(b)
            loss.backward()
            optimizer.step()
        self.assertLess(float(model.choice_loss(b).detach()), float(before.detach()))

    def test_warm_start_preserves_rating_predictions_exactly(self):
        torch.manual_seed(5)
        base = ms.SequenceModel(self.d['items'], self.c, ms.weeks(self.d)).eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'warm.pt'
            torch.save(dict(config=dataclasses.asdict(self.c), items=self.d['items'],
                            weeks=ms.weeks(self.d), epoch=12, state=base.state_dict()), path)
            a, epoch = warm_model(path)
            b, _ = warm_model(path)
            self.assertEqual(epoch, 12)
            for key, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
            np.testing.assert_array_equal(ms.predict(base, self.d), ms.predict(a, self.d))

    def test_fixed_training_is_query_label_independent_and_matched_rng(self):
        changed = dict(self.d, probe=self.d['probe'].copy())
        changed['probe']['rating'] = 0
        models = []
        for data in (self.d, changed):
            model = self.model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            train_epoch(model, data, optimizer, 1, .1, 19, 2, 4)
            models.append(model)
        for key, value in models[0].state_dict().items():
            self.assertTrue(torch.equal(value, models[1].state_dict()[key]), key)
        np.testing.assert_array_equal(ms.predict(models[0], self.d), ms.predict(models[1], changed))
        states = []
        for weight in (0., .1):
            model = self.model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            rec = train_epoch(model, self.d, optimizer, 1, weight, 19, 2, 4)
            if weight:
                self.assertGreater(rec['shared_weighted_choice_gradient_norm'], 0)
            else:
                self.assertEqual(rec['shared_weighted_choice_gradient_norm'], 0)
            states.append(torch.get_rng_state())
        self.assertTrue(torch.equal(*states))

    def test_fixed_cli_rejects_wrong_training_data_and_emits_each_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data.bin'
            header = np.zeros(1, dtype=HEADER)
            for key in ('users', 'items', 'mean', 'min_day', 'max_day'):
                header[key] = self.d[key]
            header['magic'] = b'NFPDATA1'
            header['train_count'] = len(self.d['train_items'])
            header['probe_count'] = len(self.d['probe'])
            header['implicit_count'] = len(self.d['implicit_items'])
            with data.open('wb') as f:
                header.tofile(f)
                for key, dtype in [('train_offsets', '<u8'), ('train_items', '<u2'), ('train_days', '<u2'),
                                   ('train_ratings', 'u1'), ('probe', PROBE), ('implicit_offsets', '<u8'),
                                   ('implicit_items', '<u2'), ('implicit_days', '<u2')]:
                    self.d[key].astype(dtype).tofile(f)
            torch.manual_seed(5)
            base = ms.SequenceModel(self.d['items'], self.c, ms.weeks(self.d))
            warm = root / 'warm.pt'
            torch.save(dict(config=dataclasses.asdict(self.c), items=self.d['items'], weeks=ms.weeks(self.d),
                            epoch=12, state=base.state_dict()), warm)
            training = dict(config=dataclasses.asdict(self.c), selected_epoch=12, selection_sha256='wrong')
            record = root / 'warm.json'
            record.write_text(json.dumps(training))
            args = parser().parse_args(['--data', str(data), '--warm-checkpoint', str(warm),
                                        '--warm-report', str(record), '--output', str(root / 'candidate'),
                                        '--weight', '.1', '--epochs', '2', '--positives', '2', '--negatives', '4'])
            with self.assertRaisesRegex(ValueError, 'dataset must match'):
                run(args)
            training['selection_sha256'] = mh.digest(data)
            record.write_text(json.dumps(training))
            result = run(args)
            self.assertEqual([r['epoch'] for r in result['records']], [1, 2])
            for epoch in (1, 2):
                pred = np.fromfile(root / f'candidate.e{epoch}.f32', dtype='<f4')
                self.assertEqual(pred.shape, (len(self.d['probe']),))
                self.assertTrue(np.isfinite(pred).all())
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == '__main__':
    unittest.main()
