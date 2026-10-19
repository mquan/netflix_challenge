import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import blend_fold
from selection_objective_gate import compare


class SelectionGateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.users = np.repeat(np.arange(100), 3)
        self.y = rng.uniform(1, 5, len(self.users))
        self.bank = self.y[:, None] + rng.normal(0, .3, (len(self.users), 3))
        self.control = self.y + rng.normal(0, .3, len(self.users))

    def test_identical_candidate_has_exactly_zero_gain(self):
        result, pred = compare(self.bank, self.control, self.control.copy(), self.y, self.users)
        np.testing.assert_array_equal(pred['duplicate_control'], pred['control_plus_treatment'])
        self.assertEqual(result['primary']['delta_rmse'], 0.)

    def test_evaluation_labels_cannot_change_predictions_or_penalty(self):
        treatment = self.control + .02
        result, pred = compare(self.bank, self.control, treatment, self.y, self.users)
        changed = self.y.copy()
        changed[blend_fold(self.users) == 0] = 6 - changed[blend_fold(self.users) == 0]
        other, after = compare(self.bank, self.control, treatment, changed, self.users)
        self.assertEqual(result['selected_penalties'], other['selected_penalties'])
        for key in pred:
            np.testing.assert_array_equal(pred[key], after[key])


if __name__ == '__main__':
    unittest.main()
