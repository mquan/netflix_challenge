"""Real K1/user-kNN audit handoffs and corrupt-export rejection."""
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from production_user_knn_audit import audit, prepare
from verify_production_user_knn_fixtures import exercise


class UserKnnAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'fixture'
        self.proof = exercise(self.folder)
        self.path = self.folder / '0/plan.json'
        self.plan = json.loads(self.path.read_text())
        self.root = Path(self.plan['stages'][1]['output_dir'])

    def test_actual_full_budget_and_inference_handoffs(self):
        self.assertEqual(len(self.proof['commands']), 10)
        self.assertTrue(self.proof['query_label_isolation'])
        self.assertTrue(self.proof['original_rebuilt_inference_exact'])
        parent = json.loads((self.folder / '0/K1/audit.json').read_text())
        self.assertEqual(parent['final_epoch'], 41)
        record = json.loads((self.root / 'audit.json').read_text())
        self.assertEqual(record['qualifying_count'], 80)
        self.assertEqual(len(record['outputs']), 9)
        self.assertIsNone(record['metric'])
        with self.assertRaises(FileExistsError):
            audit(self.path, 1)

    def test_reject_changed_predictions_counts_and_supports(self):
        (self.root / 'audit.json').unlink()
        cases = [('.f32', lambda b: b[:-4]),
                 ('.f32', lambda b: struct.pack('<f', float('nan')) + b[4:]),
                 ('.count.f32', lambda b: struct.pack('<f', 1.5) + b[4:]),
                 ('.count.f32', lambda b: struct.pack('<f', 1000) + b[4:]),
                 ('.support.f32', lambda b: struct.pack('<f', -1) + b[4:])]
        for suffix, change in cases:
            paths = [self.root / (prefix + suffix) for prefix in ('model', 'replay')]
            originals = [path.read_bytes() for path in paths]
            for path, body in zip(paths, originals):
                path.write_bytes(change(body))
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                audit(self.path, 1)
            for path, body in zip(paths, originals):
                path.write_bytes(body)
        audit(self.path, 1)

    def test_reject_changed_inference_parameters_and_query_coverage(self):
        (self.root / 'audit.json').unlink()
        for name, old, new in [('run.log', b'threads=3', b'threads=1'),
                               ('run.log', b'neighbors=100', b'neighbors=50'),
                               ('replay.log', b'queries=80', b'queries=79')]:
            path = self.root / name
            body = path.read_bytes()
            self.assertIn(old, body)
            path.write_bytes(body.replace(old, new))
            with self.subTest(name=name, old=old), self.assertRaises(ValueError):
                audit(self.path, 1)
            path.write_bytes(body)
        audit(self.path, 1)

    def test_prepare_rejects_missing_or_mutated_K1_before_output_creation(self):
        shutil.rmtree(self.root)
        parent = self.folder / '0/K1'
        receipt = parent / 'audit.json'
        body = receipt.read_bytes()
        receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.path, 1)
        self.assertFalse(self.root.exists())
        receipt.write_bytes(body)
        with (parent / 'model.bin').open('ab') as stream:
            stream.write(b'x')
        with self.assertRaises(ValueError):
            prepare(self.path, 1)
        self.assertFalse(self.root.exists())


if __name__ == '__main__':
    unittest.main()
