import copy
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from e1_fixed_score import arm_spec, decide
from gbdt_blend import blend_fold


class E1FixedScoreTests(unittest.TestCase):
    def test_only_declared_b_columns_change(self):
        spec = dict(fit={'columns': ['untouched-fit']}, a={'columns': ['untouched-a']},
                    b=dict(columns=['other', 'artifacts/rehearsal/B-K1.f32',
                        'artifacts/rehearsal/B-knn256td.netflix.f32',
                        'artifacts/rehearsal/B-knn64.netflix.f32'],
                        support=['artifacts/rehearsal/B-knn256td.netflix.support.f32']))
        original = copy.deepcopy(spec)
        changed = arm_spec(spec, 'w2')
        self.assertEqual(spec, original)
        self.assertEqual(changed['fit'], spec['fit'])
        self.assertEqual(changed['a'], spec['a'])
        self.assertEqual(changed['b']['columns'][0], 'other')
        with self.assertRaises(ValueError):
            arm_spec(spec, 'w4')

    def fixture(self):
        users = np.arange(200, dtype=np.uint32)
        y = np.full(200, 3.)
        predictions = {name: np.full(200, value) for name, value in
                       [('original', 3.5001), ('w1', 3.5), ('w2', 3.49), ('w3', 3.51)]}
        return y, users, predictions

    def test_gain_requires_magnitude_noise_and_all_folds(self):
        y, users, predictions = self.fixture()
        result = decide(y, users, predictions)
        self.assertTrue(result['arms']['w2']['passes_gate'])
        self.assertFalse(result['arms']['w3']['passes_gate'])
        predictions['w2'][blend_fold(users) == 0] = 3.501
        result = decide(y, users, predictions)
        self.assertLess(result['arms']['w2']['comparison']['delta_rmse'], -0.0005)
        self.assertFalse(result['passes_gate'])

    def test_run_noise_can_reject_apparent_gain(self):
        y, users, predictions = self.fixture()
        predictions['original'][:] = 3.6
        result = decide(y, users, predictions)
        self.assertFalse(result['passes_gate'])

    def test_small_gain_does_not_pass(self):
        y, users, predictions = self.fixture()
        predictions['w2'][:] = 3.4999
        self.assertFalse(decide(y, users, predictions)['passes_gate'])


if __name__ == '__main__':
    unittest.main()
