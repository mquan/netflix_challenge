"""Verify local-expert predictions cannot consume held-out rating labels."""
import pathlib
import subprocess
import tempfile
import unittest
from test_joint import dataset, run

BIN = str(pathlib.Path('build/moonshot_local').resolve())

class LocalTests(unittest.TestCase):
    def test_label_isolation_and_learning(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            for name in ('a', 'b'):
                dataset(p/(name+'.bin'), change_fold=0 if name == 'b' else None)
                run('joint', '--data', p/(name+'.bin'), '--model', p/(name+'.model'),
                    '--iterations', 1, '--factors', 4, '--predictions', p/(name+'.base'))
                for groups in (1, 3):
                    subprocess.run([BIN, '--data', str(p/(name+'.bin')),
                                    '--model', str(p/(name+'.model')), '--output', str(p/(name+str(groups))),
                                    '--groups', str(groups), '--rank', '2', '--epochs', '2'],
                                   check=True, capture_output=True)
            for groups in (1, 3):
                for epoch in (1, 2):
                    suffix = f'{groups}.epoch{epoch}.f32'
                    a = (p/('a'+suffix)).read_bytes()
                    self.assertEqual(a, (p/('b'+suffix)).read_bytes())
                    self.assertNotEqual(a, (p/'a.base').read_bytes())
                self.assertNotEqual((p/f'a{groups}.epoch1.f32').read_bytes(),
                                    (p/f'a{groups}.epoch2.f32').read_bytes())

if __name__ == '__main__':
    unittest.main()
