"""Numeric solve and held-out-label isolation for local GP inference."""
import array
import pathlib
import subprocess
import tempfile
import unittest
from test_joint import dataset, run

GP = str(pathlib.Path('build/moonshot_gp').resolve())


class LocalGPTests(unittest.TestCase):
    def test_analytic_posterior(self):
        subprocess.run([GP, '--self-test'], check=True)

    def test_probe_label_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            for name in ('a', 'b'):
                dataset(p/(name+'.bin'), change_fold=0 if name == 'b' else None)
                run('joint', '--data', p/(name+'.bin'), '--model', p/(name+'.model'),
                    '--iterations', 1, '--factors', 4)
                subprocess.run([GP, '--data', str(p/(name+'.bin')),
                                '--model', str(p/(name+'.model')),
                                '--output', str(p/name), '--neighbors', '2',
                                '--time-scale', '30', '--persistent', '.5', '--implicit-mix', '.5'], check=True, capture_output=True)
            for suffix in ('.0.f32', '.1.f32', '.2.f32', '.3.f32', '.base.f32'):
                self.assertEqual((p/('a'+suffix)).read_bytes(), (p/('b'+suffix)).read_bytes())
            mask = array.array('f', (p/'a.mask.f32').read_bytes())
            self.assertEqual(list(mask), [1.0]*200)


if __name__ == '__main__':
    unittest.main()
