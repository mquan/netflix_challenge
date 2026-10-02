"""Training-only session features must ignore probe labels."""
import pathlib
import subprocess
import tempfile
import unittest

from test_joint import dataset, run


BIN = str(pathlib.Path('build/moonshot_session_meta').resolve())


class SessionMetaTests(unittest.TestCase):
    def test_probe_label_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for name, changed in (('a', None), ('b', 0)):
                dataset(root/(name+'.bin'), change_fold=changed)
                subprocess.run([BIN, '--data', str(root/(name+'.bin')),
                                '--output', str(root/name)], check=True,
                               capture_output=True, text=True)
            for j in range(6):
                self.assertEqual((root/f'a.{j}.f32').read_bytes(),
                                 (root/f'b.{j}.f32').read_bytes())

    def test_residual_features_ignore_probe_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for name, changed in (('a', None), ('b', 0)):
                dataset(root/(name+'.bin'), change_fold=changed)
                run('joint', '--data', root/(name+'.bin'),
                    '--model', root/(name+'.model'), '--iterations', 1,
                    '--factors', 4)
                subprocess.run([BIN, '--data', str(root/(name+'.bin')),
                                '--model', str(root/(name+'.model')),
                                '--output', str(root/name)], check=True,
                               capture_output=True, text=True)
            for j in range(8):
                self.assertEqual((root/f'a.{j}.f32').read_bytes(),
                                 (root/f'b.{j}.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
