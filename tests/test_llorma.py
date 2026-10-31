"""Contracts for `netflix llorma` (L1 local low-rank ensemble, FAMILY_AUDIT.md).

Set NETFLIX_BIN to test a binary other than build/netflix.
"""
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


class LlormaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.p = pathlib.Path(cls.temp.name)
        cls.probe = dataset(cls.p / 'd.bin', users=200, items=40, per_user=16)
        run('joint', '--data', cls.p / 'd.bin', '--model', cls.p / 'j.bin', '--factors', 4, '--iterations', 20,
            '--fixed-epoch', '--predictions', cls.p / 'j.f32')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def llorma(self, tag, *extra):
        settings = {'--anchors': 12, '--rank': 3, '--iterations': 30, '--bandwidth': 1.2, '--learning-rate': .02}
        pairs = list(extra)
        for flag, value in zip(pairs[::2], pairs[1::2]):
            settings[flag] = value  # later values override defaults (the CLI reads a flag's first occurrence)
        args = [x for kv in settings.items() for x in kv]
        out = run('llorma', '--data', self.p / 'd.bin', '--joint-model', self.p / 'j.bin', '--predictions',
                  self.p / (tag + '.f32'), '--support', self.p / (tag + '.support.f32'), *args).stdout
        return out, (self.p / (tag + '.f32')).read_bytes()

    def test_learns_beyond_its_baseline(self):
        out, _ = self.llorma('a', '--anchors', 30, '--bandwidth', 1.6)
        final = float(re.search(r'final probe_rmse=(\S+)', out).group(1))
        baseline = float(re.search(r'baseline_probe_rmse=(\S+)', out).group(1))
        self.assertLess(final, baseline - .03, out)
        pred = floats(self.p / 'a.f32')
        self.assertEqual(len(pred), len(self.probe))
        self.assertTrue(all(abs(x) < 10 for x in pred))
        self.assertAlmostEqual(rmse(pred, self.probe), final, places=5)
        self.assertEqual(len(re.findall(r'^anchor=', out, re.M)), 30)

    def test_reproducible_for_fixed_threads(self):
        for threads in (1, 3):
            _, x = self.llorma('r%da' % threads, '--threads', threads)
            _, y = self.llorma('r%db' % threads, '--threads', threads)
            self.assertEqual(x, y)

    def test_uncovered_rows_fall_back_to_baseline(self):
        # A tiny bandwidth leaves most probe rows without any anchor weight.
        out, _ = self.llorma('narrow', '--bandwidth', 1e-4)
        uncovered = int(re.search(r'uncovered_probe_rows=(\d+)', out).group(1))
        self.assertGreater(uncovered, 0)
        support = floats(self.p / 'narrow.support.f32')
        self.assertEqual(sum(1 for s in support if s <= 0), uncovered)

    def test_pca_geometry_and_coverage_quantile(self):
        out, _ = self.llorma('pca', '--geometry-pca', 2, '--coverage', .3, '--bandwidth', 9)
        self.assertIn('geometry_dims=2 coverage=0.3', out)
        rows = [int(x) for x in re.findall(r' rows=(\d+)', out)]
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r > 0 for r in rows))
        # bandwidth comes from the quantile, not the (huge) --bandwidth flag
        self.assertTrue(all(float(h) < 3.2 for h in re.findall(r'h_user=(\S+)', out)))
        final = float(re.search(r'final probe_rmse=(\S+)', out).group(1))
        baseline = float(re.search(r'baseline_probe_rmse=(\S+)', out).group(1))
        self.assertLess(final, baseline)

    def test_rejects_bad_settings_and_mismatched_model(self):
        for flag, value in (('--rank', 0), ('--anchors', 0), ('--bandwidth', 0), ('--learning-rate', -1),
                            ('--coverage', 1.5), ('--geometry-pca', 99)):
            with self.subTest(flag):
                run('llorma', '--data', self.p / 'd.bin', '--joint-model', self.p / 'j.bin', '--predictions',
                    self.p / 'bad.f32', flag, value, ok=False)
        other = self.p / 'other.bin'
        dataset(other, users=200, items=40, per_user=16, seed=99)
        run('llorma', '--data', other, '--joint-model', self.p / 'j.bin', '--predictions', self.p / 'x.f32', ok=False)


if __name__ == '__main__':
    unittest.main()
