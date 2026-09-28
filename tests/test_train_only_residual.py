"""The correction must never use real probe labels to fit predictions."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from test_joint import dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import load  # noqa: E402
from train_holdout import create  # noqa: E402


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'train_only_residual.py'


class TrainOnlyResidualTests(unittest.TestCase):
    def test_target_labels_do_not_change_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dataset(root / 'a.bin')
            dataset(root / 'b.bin', change_fold=0)
            create(root / 'a.bin', root / 'holdout.bin', .014, 7)
            h = load(root / 'holdout.bin')
            t = load(root / 'a.bin')
            np.full(len(h['probe']), 3.6, dtype='<f4').tofile(root / 'holdout.f32')
            np.full(len(t['probe']), 3.6, dtype='<f4').tofile(root / 'target.f32')
            for name in ('a', 'b'):
                result = subprocess.run([sys.executable, str(SCRIPT),
                    '--holdout-data', str(root / 'holdout.bin'),
                    '--holdout-pred', str(root / 'holdout.f32'),
                    '--target-data', str(root / (name + '.bin')),
                    '--target-pred', str(root / 'target.f32'),
                    '--output', str(root / (name + '.f32')),
                    '--rounds', '8', '--min-leaf', '5'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / 'a.f32').read_bytes(), (root / 'b.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
