"""Reject broken kernel exports, coverage, budgets and J3 handoffs."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from production_kernel_audit import audit, prepare
from production_rbm_audit import digest
from verify_production_kernel_fixtures import exercise


class KernelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'fixture'
        self.proof = exercise(self.folder)
        self.path = self.folder / '0/kernel-plan.json'
        self.plan = json.loads(self.path.read_text())

    def test_all_real_stages_have_complete_audits(self):
        self.assertEqual(self.proof['audited_kernel_stages'], 8)
        self.assertEqual(len(self.proof['commands']), 20)
        for stage in self.plan['stages']:
            record = json.loads((Path(stage['output_dir']) / 'audit.json').read_text())
            self.assertIsNone(record['metric'])
            self.assertEqual(record['qualifying_count'], 24)
            self.assertEqual(len(record['outputs']), 15)
        with self.assertRaises(FileExistsError):
            audit(self.path, 0)

    def test_gp_rejects_changed_coverage_noise_exports_and_logs(self):
        root = Path(self.plan['stages'][0]['output_dir'])
        (root / 'audit.json').unlink()
        cases = [('model.mask.f32', lambda b: struct.pack('<f', 0) + b[4:]),
                 ('model.0.f32', lambda b: b[:-4]),
                 ('model.1.f32', lambda b: struct.pack('<f', float('nan')) + b[4:]),
                 ('replay.2.f32', lambda b: struct.pack('<f', 0) + b[4:]),
                 ('run.log', lambda b: b.replace(b'neighbors=32', b'neighbors=16')),
                 ('replay.log', lambda b: b.replace(b'count=24', b'count=23'))]
        for name, mutate in cases:
            path = root / name
            original = path.read_bytes()
            path.write_bytes(mutate(original))
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit(self.path, 0)
            self.assertFalse((root / 'audit.json').exists())
            path.write_bytes(original)
        audit(self.path, 0)

    def test_local_rejects_partial_schedule_and_changed_epoch_exports(self):
        root = Path(self.plan['stages'][3]['output_dir'])
        (root / 'audit.json').unlink()
        for name, mutate in [('run.log', lambda b: b.replace(b'epoch=6 ', b'epoch=5 ')),
                             ('replay.log', lambda b: b.replace(b'groups=8', b'groups=1')),
                             ('model.epoch3.f32', lambda b: b[:-4])]:
            path = root / name
            original = path.read_bytes()
            path.write_bytes(mutate(original))
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit(self.path, 3)
            path.write_bytes(original)
        audit(self.path, 3)

    def test_prepare_rejects_unaudited_or_changed_J3_before_creating_output(self):
        new_root = self.folder / 'new-kernel'
        self.plan['stages'][0]['output_dir'] = str(new_root)
        self.path.write_text(json.dumps(self.plan))
        spec = self.plan['source_phase']
        source = json.loads(Path(spec['plan']).read_text())
        receipt = Path(source['stages'][0]['output_dir']) / 'audit.json'
        body = receipt.read_bytes()
        receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.path, 0)
        self.assertFalse(new_root.exists())
        receipt.write_bytes(body)
        model = Path(spec['model'])
        model.write_bytes(model.read_bytes() + b'x')
        with self.assertRaises(ValueError):
            prepare(self.path, 0)
        self.assertFalse(new_root.exists())


if __name__ == '__main__':
    unittest.main()
