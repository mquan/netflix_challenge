import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ease_residual import ease_weights


class EaseTests(unittest.TestCase):
    def test_closed_form_matches_separate_constrained_regressions(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(50, 9)); gram = x.T@x
        w = ease_weights(gram, 3)
        np.testing.assert_array_equal(np.diag(w), 0)
        for j in range(9):
            mask = np.arange(9) != j
            separate = np.linalg.solve(gram[np.ix_(mask, mask)]+3*np.eye(8), gram[mask, j])
            np.testing.assert_allclose(w[mask, j], separate, atol=1e-12)
        # Altering the query item's value cannot alter its own reconstruction.
        changed = x.copy(); changed[:, 3] += 100
        np.testing.assert_allclose((changed@w)[:, 3], (x@w)[:, 3], atol=1e-12)


if __name__ == '__main__': unittest.main()
