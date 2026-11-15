import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from streaming_blend import design, grouped_statistics, predict_ridge, weights


class StreamingTests(unittest.TestCase):
    def test_sufficient_statistics_match_dense_and_exclude_held_labels(self):
        rng = np.random.default_rng(7)
        p = rng.normal(3.6, .3, (120, 3))
        c = np.column_stack([np.ones(120), rng.normal(size=(120, 2))])
        y = rng.normal(3.6, .7, 120)
        g = np.arange(120)%10
        stats = grouped_statistics(p, c, y, g, chunks=7)
        for inner in (False, True):
            fit = g//2 != 2
            if inner: fit &= g%2 == 0
            x = design(p, c)
            a = x[fit].T@x[fit]
            penalty = np.ones(a.shape[0])*fit.sum()*1e-5; penalty[0] = 0
            a.flat[::len(a)+1] += penalty
            expected = np.linalg.solve(a, x[fit].T@(y[fit]-3.6))
            np.testing.assert_allclose(weights(stats, 2, inner), expected, atol=1e-9)
            changed = y.copy(); changed[g//2 == 2] = 100
            other = grouped_statistics(p, c, changed, g, chunks=11)
            np.testing.assert_allclose(weights(other, 2, inner), expected, atol=1e-9)
            np.testing.assert_allclose(predict_ridge(p, c, expected, chunks=11), 3.6+x@expected)


if __name__ == '__main__': unittest.main()
