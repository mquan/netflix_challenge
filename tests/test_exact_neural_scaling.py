"""Compare the optimized memory operations with the frozen complete scaler."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import upstream_nonlinear as original
import upstream_exact_neural_scaling as compact


class ExactScalingTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(11503)
        values = rng.normal(size=(10003, 400)).astype(np.float32)
        values[:, 3] = np.float32(0)
        values[:, 7] = np.float32(3.6)
        values[0, 11] = np.finfo(np.float32).max
        values[1, 11] = -np.finfo(np.float32).max
        return values

    def test_original_float32_baseline_construction_exact(self):
        values = self.fixture()[:, :399]
        baseline = np.linspace(1.05, 4.95, len(values), dtype=np.float64)
        expected = np.column_stack([values, baseline]).astype(np.float32)
        actual = compact.append_baseline(values, baseline)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertTrue(actual.flags.c_contiguous)

    def test_full_eligible_statistics_and_every_normalized_value_exact(self):
        values = self.fixture()
        for mask in [np.arange(len(values)) % 2 == 0, np.arange(len(values)) % 5 != 3]:
            expected, reference = original.scaled(values, mask)
            actual, scaler = compact.scaled(values, mask)
            for name in ['lo', 'hi', 'mean', 'std']:
                np.testing.assert_array_equal(scaler[name].view(np.uint64), reference[name].view(np.uint64))
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_transform_block_boundaries_do_not_change_any_values(self):
        values = self.fixture()
        expected, scaler = original.scaled(values, np.arange(len(values)) % 2 == 0)
        for rows in [1, 137, 8192, len(values) + 1]:
            actual = compact.apply_scaler(values, scaler, block_rows=rows)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        with self.assertRaises(ValueError):
            compact.apply_scaler(values, scaler, block_rows=0)


if __name__ == '__main__':
    unittest.main()
