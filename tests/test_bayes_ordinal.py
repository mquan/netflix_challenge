"""Contracts for `netflix bayes --likelihood ordinal` (E5, NEXT_EXPLORATIONS.md).

Set NETFLIX_BIN to test a binary other than build/netflix.
"""
import array
import math
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

from test_bayes import dataset, floats, rmse

BIN = str(pathlib.Path(os.environ.get('NETFLIX_BIN', 'build/netflix')).resolve())


def run(*args, ok=True):
    result = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if ok and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not ok and not result.returncode:
        raise AssertionError('expected failure')
    return result


class OrdinalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.temp.name)
        self.probe = dataset(self.p / 'd.bin')

    def tearDown(self):
        self.temp.cleanup()

    def bayes(self, tag, *extra, samples=4, threads=1):
        out = run('bayes', '--data', self.p / 'd.bin', '--model', self.p / (tag + '.bin'), '--predictions',
                  self.p / (tag + '.f32'), '--factors', 3, '--samples', samples, '--burn-in', 1,
                  '--threads', threads, *extra).stdout
        return out, (self.p / (tag + '.f32')).read_bytes(), (self.p / (tag + '.bin')).read_bytes()

    def test_sampler_self_check(self):
        self.assertIn('matches analytic moments', run('check-truncated-normal').stdout)

    def test_explicit_gaussian_is_the_unchanged_default(self):
        base = self.bayes('base')
        explicit = self.bayes('gauss', '--likelihood', 'gaussian')
        self.assertEqual(base[1], explicit[1])
        self.assertEqual(base[2], explicit[2])

    def test_ordinal_residuals_stay_exact(self):
        for threads in (1, 3):
            out, _, _ = self.bayes('o%d' % threads, '--likelihood', 'ordinal', '--check-residuals',
                                   '--session', 2, threads=threads, samples=3)
            self.assertIn('likelihood=ordinal', out)
            drift = [float(x) for x in re.findall(r'residual_drift=(\S+)', out)]
            self.assertEqual(len(drift), 3)
            self.assertLess(max(drift), 1e-5, out)

    def test_ordinal_learns_and_predicts_within_star_range(self):
        _, gauss, _ = self.bayes('g', samples=60)
        out, ordinal, _ = self.bayes('o', '--likelihood', 'ordinal', samples=60)
        pred = floats(self.p / 'o.f32')
        self.assertEqual(len(pred), len(self.probe))
        self.assertTrue(all(1 <= x <= 5 for x in pred))
        raw = (self.p / 'd.bin').read_bytes()
        import struct
        mean = struct.unpack_from('<d', raw, 48)[0]
        self.assertLess(rmse(pred, self.probe), rmse([mean] * len(self.probe), self.probe) - .1)
        self.assertLess(rmse(pred, self.probe), rmse(floats(self.p / 'g.f32'), self.probe) + .05)
        self.assertNotEqual(gauss, ordinal)

    def test_rejects_unknown_likelihood_and_ordinal_resume(self):
        run('bayes', '--data', self.p / 'd.bin', '--model', self.p / 'x.bin', '--factors', 2, '--samples', 2,
            '--burn-in', 1, '--likelihood', 'poisson', ok=False)
        self.bayes('seed')
        run('bayes', '--data', self.p / 'd.bin', '--model', self.p / 'r.bin', '--factors', 3, '--samples', 6,
            '--burn-in', 1, '--likelihood', 'ordinal', '--init-model', self.p / 'seed.bin', ok=False)


if __name__ == '__main__':
    unittest.main()
