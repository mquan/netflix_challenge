import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from e3_all_fold_gate import compare_all_folds
from gbdt_blend import blend_fold
from selection_objective_gate import compare


class AllFoldGateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261031)
        self.users = np.repeat(np.arange(150, dtype=np.uint32), 3)
        self.y = rng.integers(1, 6, len(self.users)).astype(float)
        self.bank = self.y[:, None] + rng.normal(0, 1.3, (len(self.y), 2))
        self.control = self.y + rng.normal(0, 1, len(self.y))
        self.treatment = self.y + rng.normal(0, .2, len(self.y))

    def run_gate(self, y=None, treatment=None):
        return compare_all_folds(self.bank, self.control,
            self.treatment if treatment is None else treatment,
            self.y if y is None else y, self.users)

    def test_fold_zero_exactly_matches_shared_gate(self):
        shared, old = compare(self.bank, self.control, self.treatment, self.y, self.users)
        report, new = self.run_gate()
        held = blend_fold(self.users) == 0
        for name in new:
            np.testing.assert_array_equal(new[name][held], old[name])
            self.assertEqual(report['selected_penalties'][name][0], shared['selected_penalties'][name])
        self.assertTrue(report['passes_all_fold_gate'])
        self.assertTrue(all(delta < 0 for delta in report['fold_deltas']))

    def test_outer_labels_cannot_affect_their_predictions(self):
        _, original = self.run_gate()
        for outer in range(5):
            held = blend_fold(self.users) == outer
            changed = self.y.copy()
            changed[held] = 6 - changed[held]
            _, altered = self.run_gate(y=changed)
            for name in original:
                np.testing.assert_array_equal(original[name][held], altered[name][held])

    def test_duplicate_has_no_gain_and_missing_fold_is_rejected(self):
        report, _ = self.run_gate(treatment=self.control)
        self.assertFalse(report['passes_all_fold_gate'])
        self.assertEqual(report['delta_rmse'], 0)
        keep = blend_fold(self.users) != 4
        with self.assertRaisesRegex(ValueError, 'all five'):
            compare_all_folds(self.bank[keep], self.control[keep], self.treatment[keep],
                              self.y[keep], self.users[keep])


if __name__ == '__main__':
    unittest.main()
