"""round9_latent_features must reproduce the C++ joint prediction term by term."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from test_joint import dataset, run  # noqa: E402
import round9_latent_features as lat  # noqa: E402


class LatentFeatureTests(unittest.TestCase):
    def test_features_reproduce_cpp_joint(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            dataset(p/'a.bin')
            run('joint', '--data', p/'a.bin', '--model', p/'a.model', '--iterations', 2,
                '--factors', 4, '--day-factors', 2, '--predictions', p/'a.f32')
            common = ['--data', str(p/'a.bin'), '--model', str(p/'a.model'),
                      '--reference', str(p/'a.f32'), '--components', '3', '--tolerance', '1e-5']
            check = lat.main(common + ['--check'])
            self.assertLess(check['max_abs_diff'], 1e-5)
            manifest = lat.main(common + ['--output', str(p/'out')])
            cols = {pathlib.Path(c).stem[4:]: np.fromfile(c, '<f4') for c in manifest['columns']}
            self.assertEqual(len(cols), 3 + 3 + len(lat.BIAS_COLUMNS))
            d = lat.load(str(p/'a.bin'))
            total = d['mean'] + sum(cols[k].astype(np.float64)
                                    for k in ('bu', 'bu_dev', 'but', 'bi_scaled', 'bif', 'dot'))
            np.testing.assert_allclose(total, np.fromfile(p/'a.f32', '<f4'), atol=1e-5)
            np.testing.assert_allclose(cols['bi_scaled'], cols['bi']*cols['cu'], atol=1e-6)

    def test_fingerprint_rejects_other_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            dataset(p/'a.bin'); dataset(p/'b.bin', change_fold=0)
            run('joint', '--data', p/'a.bin', '--model', p/'a.model', '--iterations', 1,
                '--factors', 4, '--predictions', p/'a.f32')
            with self.assertRaises(SystemExit):
                lat.main(['--data', str(p/'b.bin'), '--model', str(p/'a.model'),
                          '--reference', str(p/'a.f32'), '--check'])


if __name__ == '__main__':
    unittest.main()
