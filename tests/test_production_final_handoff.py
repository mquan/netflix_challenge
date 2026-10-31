"""Real tiny blender CLI fit/infer/export with synthetic production-stage receipts."""
import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from production_final_handoff import export, run_stage
from production_rbm_audit import digest
from production_submission import ORDER, QUERY
from test_production_blend import fixture


class FinalHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name).resolve()
        bank, offset = fixture(cls.root)
        data = cls.root / 'train-probe-qualifying.bin'
        data.write_bytes((cls.root / 'query.bin').read_bytes())
        bank['qualifying']['dataset'] = dict(path=str(data), sha256=digest(data))
        inverse = np.arange(1001, 1101, dtype='<u4')
        queries = np.memmap(data, dtype=QUERY, mode='r', offset=offset, shape=(200,))
        order = np.zeros(200, ORDER)
        order['raw_user'], order['item'], order['day'] = inverse[queries['user']], queries['item'], queries['day']
        np.save(cls.root / 'dense-to-raw-user.npy', inverse)
        np.save(cls.root / 'qualifying-order.npy', order)
        raw = cls.root / 'qualifying.txt'
        lines = []
        for row in order:
            date = datetime.date(1999, 1, 1) + datetime.timedelta(days=int(row['day']))
            lines += [f'{row["item"]}:', f'{row["raw_user"]},{date.isoformat()}']
        raw.write_text('\n'.join(lines) + '\n')
        cls.manifest = cls.root / 'manifest.json'
        cls.manifest.write_text(json.dumps(dict(state='complete', purpose='production_only_unlabeled_queries',
            users=100, items=5, qualifying_count=200, qualifying=str(raw), qualifying_sha256=digest(raw),
            outputs={name: digest(cls.root / name) for name in
                     ('train-probe-qualifying.bin', 'dense-to-raw-user.npy', 'qualifying-order.npy')})))
        receipts = {}
        for index in range(82):
            path = cls.root / f'synthetic-production-audit-{index}.json'
            path.write_text(json.dumps(dict(state='verified_complete', metric=None)))
            receipts[str(path)] = digest(path)
        bank.update(state='complete_audited_qualifying_bank_ready_for_fixed_blender', sources={},
                    production_provenance=dict(production_stage_count=82, stage_audits=receipts,
                                               audited_bindings={}, metric=None))
        cls.bank = cls.root / 'bank.json'
        cls.bank.write_text(json.dumps(bank))
        cls.model, cls.inference, cls.replay = [cls.root / name for name in ('model', 'inference', 'replay')]
        cls.fit_receipt = run_stage(cls.bank, 'fit', cls.model)
        cls.infer_receipt = run_stage(cls.bank, 'infer', cls.inference, cls.model)
        cls.replay_receipt = run_stage(cls.bank, 'infer', cls.replay, cls.model)

    def setUp(self):
        self.output = self.root / self._testMethodName

    def test_real_fit_two_inferences_and_raw_order_text_export(self):
        self.assertIsNone(self.fit_receipt['metric'])
        self.assertEqual(self.infer_receipt['model_bindings'], self.replay_receipt['model_bindings'])
        result = export(self.bank, self.manifest, self.model, self.inference, self.replay, self.output)
        self.assertEqual(result['state'], 'verified_final_production_submission')
        self.assertEqual(result['rows'], 200)
        self.assertIsNone(result['metric'])
        text = (self.output / 'submission.txt').read_text().splitlines()
        values = np.array([float(line) for line in text if not line.endswith(':')], '<f4')
        np.testing.assert_array_equal(values, np.fromfile(self.inference / 'predictions.f32', '<f4'))
        with self.assertRaises(FileExistsError):
            export(self.bank, self.manifest, self.model, self.inference, self.replay, self.output)

    def test_changed_replay_predictions_are_rejected_before_export(self):
        path = self.replay / 'predictions.f32'
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(original[:-4])
        with self.assertRaises(ValueError):
            export(self.bank, self.manifest, self.model, self.inference, self.replay, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_model_audit_or_changed_budget_prevents_inference(self):
        audit = self.model / 'audit.json'
        original = audit.read_bytes()
        audit.unlink()
        try:
            with self.assertRaises(FileNotFoundError):
                run_stage(self.bank, 'infer', self.output, self.model)
        finally:
            audit.write_bytes(original)
        self.assertFalse(self.output.exists())
        report = self.model / 'model.json'
        original = report.read_bytes()
        self.addCleanup(report.write_bytes, original)
        altered = json.loads(original)
        altered['networks'][0]['steps'] += 1
        report.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'fixed budgets'):
            run_stage(self.bank, 'infer', self.output, self.model)
        self.assertFalse(self.output.exists())

    def test_inference_cannot_substitute_another_fitted_model_or_same_directory(self):
        with self.assertRaisesRegex(ValueError, 'Independent'):
            export(self.bank, self.manifest, self.model, self.inference, self.inference, self.output)
        path = self.replay / 'audit.json'
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        record = json.loads(original)
        target = str(self.model / 'model.json')
        record['model_bindings'][target] = 'wrong-model'
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            export(self.bank, self.manifest, self.model, self.inference, self.replay, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
