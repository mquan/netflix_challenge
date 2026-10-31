"""E3 contracts: lambda=0 is SO1's control byte for byte; lambda>0 changes only the loss."""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbdt_blend import HEADER, PROBE
import masked_history as mh
import masked_sequence as ms
import selection_objective as so
import e3_distribution_loss as e3
from test_selection_objective import fixture


class E3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.d = fixture()
        self.c = ms.Config(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=1,
                           context='session', time_bins=4, dropout=.1, batch_users=3,
                           batch_tokens=96, seq_len=24, max_targets=8, pad_width=4, pad_rows=2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data.bin'
        header = np.zeros(1, dtype=HEADER)
        for key in ('users', 'items', 'mean', 'min_day', 'max_day'):
            header[key] = self.d[key]
        header['magic'] = b'NFPDATA1'
        header['train_count'] = len(self.d['train_items'])
        header['probe_count'] = len(self.d['probe'])
        header['implicit_count'] = len(self.d['implicit_items'])
        with self.data.open('wb') as f:
            header.tofile(f)
            for key, dtype in [('train_offsets', '<u8'), ('train_items', '<u2'), ('train_days', '<u2'),
                               ('train_ratings', 'u1'), ('probe', PROBE), ('implicit_offsets', '<u8'),
                               ('implicit_items', '<u2'), ('implicit_days', '<u2')]:
                self.d[key].astype(dtype).tofile(f)
        torch.manual_seed(5)
        base = ms.SequenceModel(self.d['items'], self.c, ms.weeks(self.d))
        self.warm = self.root / 'warm.pt'
        torch.save(dict(config=dataclasses.asdict(self.c), items=self.d['items'], weeks=ms.weeks(self.d),
                        epoch=12, state=base.state_dict()), self.warm)
        self.record = self.root / 'warm.json'
        self.record.write_text(json.dumps(dict(config=dataclasses.asdict(self.c), selected_epoch=12,
                                               selection_sha256=mh.digest(self.data))))

    def tearDown(self):
        self.temp.cleanup()

    def argv(self, name, extra=()):
        return ['--data', str(self.data), '--warm-checkpoint', str(self.warm), '--warm-report', str(self.record),
                '--output', str(self.root / name), '--weight', '0', '--epochs', '2', '--positives', '2',
                '--negatives', '4', *extra]

    def outputs(self, name):
        pred = (self.root / f'{name}.e2.f32').read_bytes()
        state = torch.load(self.root / f'{name}.e2.pt', weights_only=False)['state']
        return pred, state

    def test_zero_weight_reproduces_so1_control_exactly(self):
        so.run(so.parser().parse_args(self.argv('control')))
        report = e3.run(e3.parser().parse_args(self.argv('e3zero', ['--ce-weight', '0'])))
        pred_c, state_c = self.outputs('control')
        pred_e, state_e = self.outputs('e3zero')
        self.assertEqual(pred_c, pred_e)
        self.assertEqual(state_c.keys(), state_e.keys())
        for key in state_c:
            self.assertTrue(torch.equal(state_c[key], state_e[key]), key)
        self.assertEqual(report['arguments']['ce_weight'], 0)
        self.assertEqual(report['arguments']['e3_script_sha256'], mh.digest(Path(e3.__file__).resolve()))
        self.assertIs(so.mh.loss_value, mh.loss_value)

    def test_positive_weight_changes_fit_and_restores_loss(self):
        original = mh.loss_value
        so.run(so.parser().parse_args(self.argv('control')))
        e3.run(e3.parser().parse_args(self.argv('e3', ['--ce-weight', '.1'])))
        self.assertIs(mh.loss_value, original)
        pred_c, _ = self.outputs('control')
        pred_e, _ = self.outputs('e3')
        self.assertNotEqual(pred_c, pred_e)
        report = json.loads((self.root / 'e3.json').read_text())
        self.assertEqual(report['arguments']['ce_weight'], .1)
        self.assertEqual([r['epoch'] for r in report['records']], [1, 2])

    def test_combined_loss_value(self):
        torch.manual_seed(3)
        model = ms.SequenceModel(self.d['items'], self.c, ms.weeks(self.d))
        raw = torch.randn(7, 5)
        ratings = torch.tensor([0, 1, 2, 3, 4, 2, 1])
        base = mh.loss_value(model, raw, ratings)
        expected = base + .25 * torch.nn.functional.cross_entropy(raw, ratings)
        self.assertTrue(torch.allclose(e3.combined_loss(mh.loss_value, .25)(model, raw, ratings), expected))
        self.assertTrue(torch.equal(e3.combined_loss(mh.loss_value, 0)(model, raw, ratings), base))

    def test_rejects_auxiliary_weight_and_bad_lambda(self):
        args = e3.parser().parse_args(self.argv('bad', ['--ce-weight', '.1']))
        args.weight = .1
        with self.assertRaisesRegex(ValueError, 'auxiliary weight'):
            e3.run(args)
        for value in ('-1', 'nan'):
            with self.assertRaisesRegex(ValueError, 'ce-weight'):
                e3.run(e3.parser().parse_args(self.argv('bad' + value, ['--ce-weight', value])))
        self.assertFalse(list(self.root.glob('bad*')))


if __name__ == '__main__':
    unittest.main()
