"""Fixed-budget AutoRec must not select or score unknown query labels."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_joint import dataset
from test_native_fixed_epochs import query_labels

ROOT = Path(__file__).resolve().parents[1]


class AutoRecProductionTests(unittest.TestCase):
    def invoke(self, data, output, *extra):
        command = [sys.executable, str(ROOT / 'scripts/autorec.py'), '--data', str(data),
                   '--predictions', str(output), '--hidden', '8', '--layers', '1',
                   '--batch', '32', '--epochs', '6', '--patience', '1', '--device', 'cpu', *extra]
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        return subprocess.run(command, capture_output=True, text=True, env=environment)

    def test_fixed_budget_ignores_query_labels_and_omits_unknown_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            for label in (0, 5):
                data, output = p / f'{label}.bin', p / f'{label}.f32'
                dataset(data, correlated_train=True)
                query_labels(data, label)
                result = self.invoke(data, output, '--fixed-epoch')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(sum(line.startswith('epoch=') for line in result.stdout.splitlines()), 6)
                self.assertNotIn('early_stop', result.stdout)
                if label == 0:
                    self.assertNotIn('probe_rmse', result.stdout)
                    self.assertIn('query_labels=unknown', result.stdout)
            self.assertEqual((p / '0.f32').read_bytes(), (p / '5.f32').read_bytes())
            self.assertEqual((p / '0.f32').stat().st_size, 200 * 4)

    def test_unknown_labels_cannot_enter_development_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            dataset(p / 'data.bin')
            query_labels(p / 'data.bin', 0)
            result = self.invoke(p / 'data.bin', p / 'invalid.f32')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('require --fixed-epoch', result.stderr)
            self.assertFalse((p / 'invalid.f32').exists())

    def test_saved_trajectory_epoch_matches_stopping_at_that_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            dataset(p / 'data.bin', correlated_train=True)
            query_labels(p / 'data.bin', 0)
            full = self.invoke(p / 'data.bin', p / 'full.f32', '--fixed-epoch', '--save-epoch', '2')
            short = self.invoke(p / 'data.bin', p / 'short.f32', '--fixed-epoch', '--epochs', '2')
            self.assertEqual(full.returncode, 0, full.stdout + full.stderr)
            self.assertEqual(short.returncode, 0, short.stdout + short.stderr)
            self.assertEqual((p / 'full.e2.f32').read_bytes(), (p / 'short.f32').read_bytes())
            self.assertNotEqual((p / 'full.e2.f32').read_bytes(), (p / 'full.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
