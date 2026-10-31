"""Contracts for `netflix joint --row-weights` (E1 in NEXT_EXPLORATIONS.md).

Set NETFLIX_BIN to test a binary other than build/netflix (for example a
separately built build/netflix-rowweights while other jobs use build/netflix).
"""
import array
import os
import pathlib
import subprocess
import tempfile
import unittest

from test_joint import dataset

BIN = str(pathlib.Path(os.environ.get('NETFLIX_BIN', 'build/netflix')).resolve())
TRAIN_ROWS = 300  # test_joint.dataset: 100 users x 3 training ratings


def joint(data, model, predictions, *extra, ok=True):
    result = subprocess.run(
        [BIN, 'joint', '--data', str(data), '--model', str(model), '--predictions', str(predictions),
         '--factors', '4', '--iterations', '3', '--fixed-epoch', '--threads', '1', '--seed', '7', *map(str, extra)],
        capture_output=True, text=True)
    if ok and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not ok and not result.returncode:
        raise AssertionError('expected failure')
    return result


def write_weights(path, values):
    path.write_bytes(array.array('f', values).tobytes())


class RowWeightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.temp.name)
        self.data = self.p / 'd.bin'
        dataset(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def train(self, tag, *extra):
        model, pred = self.p / (tag + '.bin'), self.p / (tag + '.f32')
        out = joint(self.data, model, pred, *extra)
        return model.read_bytes(), pred.read_bytes(), out.stdout

    def test_unit_weights_reproduce_unweighted_run_exactly(self):
        write_weights(self.p / 'ones.f32', [1.0] * TRAIN_ROWS)
        base_model, base_pred, base_out = self.train('base')
        model, pred, out = self.train('ones', '--row-weights', self.p / 'ones.f32')
        self.assertEqual(model, base_model)
        self.assertEqual(pred, base_pred)
        self.assertIn('non_unit=0', out)
        epochs = lambda text: [l for l in text.splitlines() if l.startswith('epoch=')]
        strip = lambda lines: [l.rsplit(' seconds=', 1)[0] for l in lines]
        self.assertEqual(strip(epochs(out)), strip(epochs(base_out)))

    def test_non_unit_weights_change_the_fit(self):
        write_weights(self.p / 'w.f32', [3.0 if n % 3 == 2 else 1.0 for n in range(TRAIN_ROWS)])
        base_model, base_pred, _ = self.train('base')
        model, pred, out = self.train('w', '--row-weights', self.p / 'w.f32')
        self.assertNotEqual(pred, base_pred)
        self.assertNotEqual(model, base_model)
        self.assertIn('non_unit=100', out)

    def test_weighted_model_keeps_unweighted_fingerprint(self):
        # eval-joint on the same dataset must accept a weighted model unchanged.
        write_weights(self.p / 'w.f32', [2.0] * TRAIN_ROWS)
        _, pred, _ = self.train('w', '--row-weights', self.p / 'w.f32')
        again = self.p / 'again.f32'
        subprocess.run([BIN, 'eval-joint', '--data', str(self.data), '--model', str(self.p / 'w.bin'),
                        '--predictions', str(again)], check=True, capture_output=True)
        self.assertEqual(again.read_bytes(), pred)

    def test_malformed_weights_are_rejected(self):
        cases = {'short': [1.0] * (TRAIN_ROWS - 1), 'long': [1.0] * (TRAIN_ROWS + 1),
                 'negative': [1.0] * (TRAIN_ROWS - 1) + [-1.0],
                 'nan': [1.0] * (TRAIN_ROWS - 1) + [float('nan')],
                 'huge': [1.0] * (TRAIN_ROWS - 1) + [17.0]}
        for name, values in cases.items():
            with self.subTest(name):
                write_weights(self.p / (name + '.f32'), values)
                joint(self.data, self.p / (name + '.bin'), self.p / (name + '.pred'),
                      '--row-weights', self.p / (name + '.f32'), ok=False)
                self.assertFalse((self.p / (name + '.bin')).exists())
        joint(self.data, self.p / 'missing.bin', self.p / 'missing.pred',
              '--row-weights', self.p / 'absent.f32', ok=False)


if __name__ == '__main__':
    unittest.main()
