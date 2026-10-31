"""Checkpoint instrumentation must preserve the frozen AutoRec learning path."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_autorec_source import prepare
from production_rbm_audit import digest
from test_joint import dataset
from test_native_fixed_epochs import query_labels

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class AutoRecSourceTests(unittest.TestCase):
    def invoke(self, source, data, output, *extra):
        command = [sys.executable, str(ROOT / 'scripts/production_autorec_source.py'), 'run',
                   '--source', str(source), '--threads', '1', '--', '--data', str(data),
                   '--predictions', str(output), '--hidden', '8', '--layers', '1', '--batch', '32',
                   '--epochs', '6', '--fixed-epoch', '--save-epoch', '2', '--device', 'cpu', *extra]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        return subprocess.run(command, text=True, capture_output=True, env=env)

    def test_capture_preserves_final_and_shared_snapshot_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = prepare(root / 'source')
            data = root / 'data.bin'
            dataset(data, correlated_train=True)
            query_labels(data, 0)
            a = self.invoke(ROOT / 'scripts/autorec.py', data, root / 'original.f32')
            b = self.invoke(root / 'source/autorec.py', data, root / 'captured.f32')
            self.assertEqual(a.returncode, 0, a.stdout + a.stderr)
            self.assertEqual(b.returncode, 0, b.stdout + b.stderr)
            for suffix in ('.f32', '.e2.f32'):
                self.assertEqual((root / ('original' + suffix)).read_bytes(), (root / ('captured' + suffix)).read_bytes())
            self.assertEqual(sum(line.startswith('epoch=') for line in b.stdout.splitlines()), 6)
            self.assertNotIn('probe_rmse', b.stdout)
            for epoch, filename in [(2, 'captured.e2.pt'), (6, 'captured.pt')]:
                saved = torch.load(root / filename, map_location='cpu', weights_only=True)
                self.assertEqual(saved['epoch'], epoch)
                self.assertEqual(saved['source_sha256'], manifest['source_sha256'])
                self.assertEqual(saved['torch_threads'], 1)
                self.assertEqual(saved['config']['epochs'], 6)
            self.assertEqual(digest(ROOT / 'scripts/autorec.py'), manifest['original_source_sha256'])

    def test_final_and_snapshot_reload_match_and_wrong_config_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare(root / 'source')
            data = root / 'data.bin'
            dataset(data, correlated_train=True)
            query_labels(data, 0)
            source = root / 'source/autorec.py'
            train = self.invoke(source, data, root / 'model.f32')
            self.assertEqual(train.returncode, 0, train.stdout + train.stderr)
            for name, checkpoint in [('final', root / 'model.pt'), ('snapshot', root / 'model.e2.pt')]:
                output = root / (name + '.f32')
                reload = self.invoke(source, data, output, '--reload-checkpoint', str(checkpoint))
                self.assertEqual(reload.returncode, 0, reload.stdout + reload.stderr)
                expected = root / ('model.f32' if name == 'final' else 'model.e2.f32')
                self.assertEqual(output.read_bytes(), expected.read_bytes())
                self.assertNotIn('epoch=', reload.stdout.replace('reload_epoch=', 'reload='))
            bad = self.invoke(source, data, root / 'invalid.f32', '--reload-checkpoint', str(root / 'model.pt'), '--hidden', '9')
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('configuration differs', bad.stderr)
            self.assertFalse((root / 'invalid.f32').exists())

    def test_checkpoint_capture_remains_query_label_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare(root / 'source')
            saved = []
            for label in (0, 5):
                data = root / f'data-{label}.bin'
                dataset(data, correlated_train=True)
                query_labels(data, label)
                result = self.invoke(root / 'source/autorec.py', data, root / f'model-{label}.f32')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                saved.append(torch.load(root / f'model-{label}.pt', map_location='cpu', weights_only=True))
            self.assertEqual((root / 'model-0.f32').read_bytes(), (root / 'model-5.f32').read_bytes())
            self.assertEqual(set(saved[0]['state']), set(saved[1]['state']))
            for name in saved[0]['state']:
                self.assertTrue(torch.equal(saved[0]['state'][name], saved[1]['state'][name]), name)


if __name__ == '__main__':
    unittest.main()
