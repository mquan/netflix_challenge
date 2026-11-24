"""Synthetic checks for rcb_blend: context weights recover a planted context effect
and are not preferred when none exists."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import rcb_blend as b

def make(effect, n=40000, seed=0):
    r = np.random.default_rng(seed); users = r.integers(0, 4000, n)
    ctx = {k: r.normal(size=n).astype(np.float32) for k in b.CTX_NAMES}
    truth = 3.6 + r.normal(0, .8, n); p1 = truth + r.normal(0, .5, n); p2 = truth + r.normal(0, .5, n)
    hi = ctx['log_user_support'] > 0.6744  # top quartile trusts p1 more
    if effect: p2 = np.where(hi, truth + r.normal(0, 1.5, n), p2)
    return np.column_stack([p1, p2]).astype(np.float32), truth, users, ctx

class T(unittest.TestCase):
    def run_cv(self, effect):
        X, y, u, ctx = make(effect); folds = b.splitmix_folds(u, 1, 5); g = c = 0.0
        for f in range(5):
            tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            d, w = b.fit_global(X, y, [0, 1], tr); g += np.sum((d.predict(te, w) - y[te]) ** 2)
            d, w, lam, _ = b.fit_context(X, y, u, ctx, [0, 1], tr, 7 + f); c += np.sum((d.predict(te, w) - y[te]) ** 2)
        return np.sqrt(g / len(y)), np.sqrt(c / len(y))
    def test_planted(self):
        g, c = self.run_cv(True); self.assertLess(c, g - 0.005)
    def test_null(self):
        g, c = self.run_cv(False); self.assertLess(abs(c - g), 0.002)
    def test_intercept_unpenalized(self):
        d = b.Design(np.zeros((1, 2), np.float32), None, None, [0, 1]); p = d.penalty(0)
        self.assertEqual(p[2, 2], 0); self.assertEqual(p[0, 0], b.GLOBAL_LAMBDA)
unittest.main()
