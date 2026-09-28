"""Outer-fold labels must not affect normalization, fitting, or epoch selection."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path('scripts').resolve()))
from moonshot_nn_blend import cross_fit
from gbdt_blend import blend_fold

class NeuralBlendTests(unittest.TestCase):
    def test_outer_label_isolation(self):
        rng = np.random.default_rng(8)
        users = np.repeat(np.arange(1000, dtype=np.uint32), 3)
        pred = rng.uniform(1.5, 4.5, size=(len(users), 3)).astype(np.float32)
        y = np.clip(pred[:, 0]+.3*np.sin(pred[:, 1]*3), 1, 5)
        c = np.column_stack([np.ones(len(users)), rng.normal(size=len(users))]).astype(np.float32)
        first, control, records = cross_fit(pred, c, y, users, hidden=4, epochs=3)
        self.assertGreater(records[0]['chosen_epochs'], 0)
        changed = y.copy(); held = blend_fold(users) == 0; changed[held] = 6-changed[held]
        second, second_control, second_records = cross_fit(pred, c, changed, users, hidden=4, epochs=3)
        np.testing.assert_array_equal(first[held], second[held])
        np.testing.assert_array_equal(control[held], second_control[held])
        self.assertEqual(records[0]['chosen_epochs'], second_records[0]['chosen_epochs'])

if __name__ == '__main__': unittest.main()
