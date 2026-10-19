"""Validate the preserved Bayes binary and version-one posterior reader."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from recover_bayes_recipes import posterior
from test_bayes import dataset
from test_native_fixed_epochs import query_labels

BIN = ROOT / 'artifacts/r8/bin/netflix-bayes-0c1f29d'


def run(*args):
    result = subprocess.run([str(BIN), 'bayes', *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)


class BayesProductionTests(unittest.TestCase):
    @unittest.skipUnless(BIN.is_file(), 'preserved historical Bayes binary unavailable')
    def test_unknown_labels_resume_and_serialized_posterior_single_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            for label in (0, 5):
                prefix = p / str(label)
                data = Path(str(prefix) + '.data')
                initial = Path(str(prefix) + '.initial')
                model = Path(str(prefix) + '.model')
                prediction = Path(str(prefix) + '.f32')
                dataset(data, users=12, items=10, per_user=8)
                query_labels(data, label)
                run('--data', data, '--model', initial, '--factors', 3, '--seed', 13,
                    '--samples', 2, '--burn-in', 1, '--threads', 1, '--item-bin', 2, '--drift', 2)
                first, _ = posterior(initial)
                self.assertEqual((first['sweep'], first['kept']), (2, 1))
                run('--data', data, '--init-model', initial, '--model', model,
                    '--samples', 6, '--burn-in', 1, '--threads', 1, '--predictions', prediction)
                final, mean = posterior(model)
                self.assertEqual((final['sweep'], final['kept']), (6, 5))
                self.assertEqual(mean.tobytes(), prediction.read_bytes())
                again = Path(str(prefix) + '.again.f32')
                run('--data', data, '--init-model', model, '--model', Path(str(prefix) + '.again.model'),
                    '--samples', 6, '--burn-in', 1, '--threads', 1, '--predictions', again)
                self.assertEqual(prediction.read_bytes(), again.read_bytes())
            self.assertEqual((p / '0.model').read_bytes()[16:], (p / '5.model').read_bytes()[16:])
            self.assertEqual((p / '0.f32').read_bytes(), (p / '5.f32').read_bytes())

    @unittest.skipUnless(BIN.is_file(), 'preserved historical Bayes binary unavailable')
    def test_thread_count_change_preserves_average_and_reemission(self):
        # Historical multi-threaded bag updates are explicitly Hogwild. Verify
        # serialization/resume invariants, not bitwise equality across retrains.
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            dataset(p / 'data', users=12, items=10, per_user=8)
            query_labels(p / 'data', 0)
            run('--data', p / 'data', '--model', p / 'initial', '--factors', 3, '--seed', 13,
                '--samples', 2, '--burn-in', 1, '--threads', 4, '--item-bin', 2, '--drift', 2)
            run('--data', p / 'data', '--init-model', p / 'initial', '--model', p / 'final',
                '--samples', 6, '--burn-in', 1, '--threads', 5, '--predictions', p / 'pred.f32')
            info, mean = posterior(p / 'final')
            self.assertEqual((info['sweep'], info['kept']), (6, 5))
            self.assertEqual(mean.tobytes(), (p / 'pred.f32').read_bytes())
            run('--data', p / 'data', '--init-model', p / 'final', '--model', p / 'again',
                '--samples', 6, '--burn-in', 1, '--threads', 5, '--predictions', p / 'again.f32')
            self.assertEqual((p / 'pred.f32').read_bytes(), (p / 'again.f32').read_bytes())

    def test_reader_rejects_wrong_version_and_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'bad'
            p.write_bytes(struct.pack('<QQIII', 0, 1, 3, 4, 2))
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                posterior(p)
            p.write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                posterior(p)


if __name__ == '__main__':
    unittest.main()
