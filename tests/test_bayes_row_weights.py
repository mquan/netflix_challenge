"""Contracts for `netflix bayes --row-weights` (E1-Bayes, NEXT_EXPLORATIONS.md).

A weight w on training row p sets its noise precision to alpha*w. Set
NETFLIX_BIN to test a binary other than build/netflix.
"""
import array
import os
import pathlib
import random
import re
import subprocess
import tempfile
import unittest

from test_bayes import dataset, floats

BIN = str(pathlib.Path(os.environ.get('NETFLIX_BIN', 'build/netflix')).resolve())
TRAIN_ROWS = 120 * 10  # test_bayes.dataset: 120 users x (12 - 2 probe) ratings


def run(*args, ok=True):
    result = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if ok and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not ok and not result.returncode:
        raise AssertionError('expected failure')
    return result


def write_weights(path, values):
    path.write_bytes(array.array('f', values).tobytes())


class BayesRowWeightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def bayes(self, data, tag, *extra, threads=1, samples=4):
        out = run('bayes', '--data', data, '--model', self.p / (tag + '.bin'), '--predictions',
                  self.p / (tag + '.f32'), '--factors', 3, '--samples', samples, '--burn-in', 1,
                  '--threads', threads, '--session', 2, '--item-freq', 2, *extra).stdout
        return out, (self.p / (tag + '.f32')).read_bytes(), (self.p / (tag + '.bin')).read_bytes()

    def test_unit_weights_reproduce_unweighted_chain_exactly(self):
        dataset(self.p / 'd.bin')
        write_weights(self.p / 'ones.f32', [1.0] * TRAIN_ROWS)
        for threads in (1, 3):
            base_out, base_pred, base_model = self.bayes(self.p / 'd.bin', 'b%d' % threads, threads=threads)
            out, pred, model = self.bayes(self.p / 'd.bin', 'o%d' % threads, '--row-weights', self.p / 'ones.f32',
                                          threads=threads)
            if threads == 1:  # multi-thread bag draws race, so only one thread is bit-reproducible
                self.assertEqual(pred, base_pred)
                self.assertEqual(model, base_model)
                sweeps = lambda text: [l.rsplit(' seconds=', 1)[0] for l in text.splitlines() if l.startswith('sweep=')]
                self.assertEqual(sweeps(out), sweeps(base_out))
            self.assertIn('non_unit=0', out)

    def test_weighted_residuals_stay_exact(self):
        dataset(self.p / 'd.bin')
        rng = random.Random(3)
        write_weights(self.p / 'w.f32', [rng.choice([1.0, 2.0, 5.0]) for _ in range(TRAIN_ROWS)])
        for threads in (1, 3):
            out, _, _ = self.bayes(self.p / 'd.bin', 'w%d' % threads, '--row-weights', self.p / 'w.f32',
                                   '--check-residuals', threads=threads, samples=3)
            drift = [float(x) for x in re.findall(r'residual_drift=(\S+)', out)]
            self.assertEqual(len(drift), 3)
            self.assertLess(max(drift), 1e-6, out)

    def test_heavily_weighted_rows_are_fitted_more_closely(self):
        # Probe rows duplicate the training rows, so probe predictions are the
        # posterior-mean fit of each training row.
        probe = dataset(self.p / 'd.bin', probe_is_train=True)
        self.assertEqual(len(probe), TRAIN_ROWS)
        heavy = [n % 4 == 0 for n in range(TRAIN_ROWS)]
        write_weights(self.p / 'h.f32', [8.0 if h else 1.0 for h in heavy])
        _, base, _ = self.bayes(self.p / 'd.bin', 'base', samples=40)
        _, weighted, _ = self.bayes(self.p / 'd.bin', 'heavy', '--row-weights', self.p / 'h.f32', samples=40)

        def mse(raw, chosen):
            pred = array.array('f'); pred.frombytes(raw)
            err = [(r - p) ** 2 for p, (_, _, _, r), c in zip(pred, probe, chosen) if c]
            return sum(err) / len(err)
        self.assertLess(mse(weighted, heavy), 0.8 * mse(base, heavy))

    def test_malformed_weights_are_rejected(self):
        dataset(self.p / 'd.bin')
        cases = {'short': [1.0] * (TRAIN_ROWS - 1), 'long': [1.0] * (TRAIN_ROWS + 1),
                 'zero': [1.0] * (TRAIN_ROWS - 1) + [0.0], 'negative': [1.0] * (TRAIN_ROWS - 1) + [-1.0],
                 'nan': [1.0] * (TRAIN_ROWS - 1) + [float('nan')], 'huge': [1.0] * (TRAIN_ROWS - 1) + [17.0]}
        for name, values in cases.items():
            with self.subTest(name):
                write_weights(self.p / (name + '.f32'), values)
                run('bayes', '--data', self.p / 'd.bin', '--model', self.p / (name + '.bin'), '--factors', 2,
                    '--samples', 2, '--burn-in', 1, '--row-weights', self.p / (name + '.f32'), ok=False)
                self.assertFalse((self.p / (name + '.bin')).exists())


if __name__ == '__main__':
    unittest.main()
