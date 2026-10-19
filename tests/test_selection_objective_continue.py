"""Preserve the choice head and prove deterministic epoch-boundary CPU resumption."""
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_objective as so
import selection_objective_continue as continuation
from gbdt_blend import HEADER, PROBE
from test_selection_objective import fixture


def legacy_parent(root, unknown=False):
    d = fixture()
    if unknown:
        d['probe']['rating'] = 0
    header = np.zeros(1, dtype=HEADER)
    for key in ('users', 'items', 'mean', 'min_day', 'max_day'):
        header[key] = d[key]
    header['magic'] = b'NFPDAT2'
    header['train_count'], header['probe_count'], header['implicit_count'] = len(d['train_items']), len(d['probe']), len(d['implicit_items'])
    data = root / 'data.bin'
    with data.open('wb') as f:
        header.tofile(f)
        for key, dtype in [('train_offsets', '<u8'), ('train_items', '<u2'), ('train_days', '<u2'),
            ('train_ratings', 'u1'), ('probe', PROBE), ('implicit_offsets', '<u8'), ('implicit_items', '<u2'), ('implicit_days', '<u2')]:
            d[key].astype(dtype).tofile(f)
    c = so.ms.Config(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=1,
        context='session', time_bins=4, dropout=.1, batch_users=3, batch_tokens=96,
        seq_len=24, max_targets=8, pad_width=4, pad_rows=2)
    torch.manual_seed(5)
    base = so.ms.SequenceModel(d['items'], c, so.ms.weeks(d))
    warm = root / 'warm.pt'
    torch.save(dict(config=dataclasses.asdict(c), items=d['items'], weeks=so.ms.weeks(d),
                    epoch=12, state=base.state_dict()), warm)
    warm_report = root / 'warm.json'
    warm_report.write_text(json.dumps(dict(config=dataclasses.asdict(c), selected_epoch=12,
                                         selection_sha256=so.mh.digest(data))))
    args = so.parser().parse_args(['--data', str(data), '--warm-checkpoint', str(warm),
        '--warm-report', str(warm_report), '--output', str(root / 'parent'), '--epochs', '1',
        '--weight', '.1', '--seed', '19', '--threads', '1', '--positives', '2', '--negatives', '4'])
    so.run(args)
    return data, root / 'parent.json', root / 'parent.e1.pt'


def args(data, report, checkpoint, output, epochs=1, mode='restart'):
    return continuation.parser().parse_args(['--data', str(data), '--parent-report', str(report),
        '--checkpoint', str(checkpoint), '--output', str(output), '--epochs', str(epochs), '--optimizer-mode', mode])


class ContinuationTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_legacy_restart_preserves_every_parameter_and_learned_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, report, checkpoint = legacy_parent(root)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertGreater(float(saved['state']['choice_bias'].abs().sum()), 0)
            model, optimizer, _, _, _, _, _ = continuation.load_parent(report, checkpoint, data, 'restart')
            self.assert_nested_equal(model.state_dict(), saved['state'])
            self.assertEqual(len(optimizer.state), 0)
            with self.assertRaisesRegex(ValueError, 'rating-only'):
                so.warm_model(checkpoint)
            with self.assertRaisesRegex(ValueError, 'lacks optimizer/RNG'):
                continuation.load_parent(report, checkpoint, data, 'resume')

    def test_split_resume_matches_uninterrupted_model_optimizer_rng_and_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, report, checkpoint = legacy_parent(root)
            continuation.run(args(data, report, checkpoint, root / 'whole', epochs=2))
            continuation.run(args(data, report, checkpoint, root / 'part', epochs=1))
            process = subprocess.run([sys.executable, str(Path(continuation.__file__).resolve()),
                '--data', str(data), '--parent-report', str(root / 'part.json'),
                '--checkpoint', str(root / 'part.e2.pt'), '--output', str(root / 'resumed'),
                '--epochs', '1', '--optimizer-mode', 'resume'], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            resumed = json.loads((root / 'resumed.json').read_text())
            a = torch.load(root / 'whole.e3.pt', weights_only=True)
            b = torch.load(root / 'resumed.e3.pt', weights_only=True)
            self.assert_nested_equal(a, b)
            self.assertEqual((root / 'whole.e3.f32').read_bytes(), (root / 'resumed.e3.f32').read_bytes())
            self.assertEqual(resumed['parent_epoch'], 2)
            self.assertEqual(resumed['fixed_final_epoch'], 3)
            self.assertEqual(resumed['state'], 'complete')
            with self.assertRaisesRegex(ValueError, 'same device'):
                continuation.load_parent(root / 'part.json', root / 'part.e2.pt', data, 'resume', 'mps')
            with self.assertRaises(FileExistsError):
                continuation.run(args(data, report, checkpoint, root / 'whole'))

    def test_changed_dataset_or_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, report, checkpoint = legacy_parent(root)
            changed = root / 'changed.bin'
            changed.write_bytes(data.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError, 'dataset differs'):
                continuation.load_parent(report, checkpoint, changed, 'restart')
            bad_checkpoint = root / 'changed.pt'
            bad_checkpoint.write_bytes(checkpoint.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError, 'last recorded epoch'):
                continuation.load_parent(report, bad_checkpoint, data, 'restart')

    def test_unknown_query_labels_do_not_produce_a_validation_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, report, checkpoint = legacy_parent(root, unknown=True)
            result = continuation.run(args(data, report, checkpoint, root / 'unknown'))
            self.assertIsNone(result['records'][0]['validation_rmse'])
            self.assertTrue(np.isfinite(np.fromfile(root / 'unknown.e2.f32', '<f4')).all())


if __name__ == '__main__':
    unittest.main()
