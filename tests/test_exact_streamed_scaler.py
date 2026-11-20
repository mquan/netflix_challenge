from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upstream_exact_streamed_scaler as streamed


class CompleteOriginalScaler(unittest.TestCase):
    def original(self, values, eligible):
        selected = values[eligible]
        lo, hi = np.percentile(selected, [1,99], axis=0)
        clipped = np.clip(selected, lo, hi).astype(np.float64)
        return dict(lo=lo, hi=hi, mean=clipped.mean(0), std=np.maximum(clipped.std(0), 1e-6))

    def test_all_statistics_match_original_full_matrix_across_boundaries(self):
        generator = np.random.default_rng(45)
        values = generator.normal(size=(5007,579)).astype(np.float32)
        values[:, 0] = 0; values[:, 1] = 7
        values[:, 2] = np.round(values[:, 2])
        eligible = generator.random(len(values)) < .8
        expected = self.original(values, eligible)
        for block in [1,257,8192]:
            actual = streamed.fit_scaler(values, eligible, block)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_original_order_is_preserved_with_cancellation_and_sparse_eligibility(self):
        generator = np.random.default_rng(2026)
        values = (generator.normal(size=(2033,579))*1e12).astype(np.float32)
        values[::2] *= -1; values[:,3] = np.float32(-0.)
        eligible = np.arange(len(values)) % 7 == 0
        expected = self.original(values, eligible)
        actual = streamed.fit_scaler(values, eligible, 63)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_reduced_feature_population_or_wrong_precision_rejected(self):
        eligible = np.ones(5, dtype=bool)
        for values in [np.zeros((5,578), dtype=np.float32), np.zeros((5,579), dtype=np.float64)]:
            with self.assertRaises(ValueError):
                streamed.fit_scaler(values, eligible)


if __name__ == '__main__':
    unittest.main()
