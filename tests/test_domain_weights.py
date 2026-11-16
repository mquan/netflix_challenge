import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from domain_weights import fit_classifier


class DomainWeightTests(unittest.TestCase):
    def test_density_ratio_recovers_known_gaussian_shift_on_unseen_rows(self):
        rng=np.random.default_rng(104)
        shift=np.array([.5,-.35])
        source=rng.normal(size=(50000,2))
        target=rng.normal(size=(50000,2))+shift
        beta=fit_classifier(np.r_[source,target],np.r_[np.zeros(len(source)),np.ones(len(target))])
        held=rng.normal(size=(100000,2))
        weights=np.exp(held@beta[:-1]+beta[-1])
        corrected=np.average(held,axis=0,weights=weights)
        np.testing.assert_allclose(corrected,shift,atol=.025)
        self.assertLess(np.linalg.norm(corrected-shift),np.linalg.norm(held.mean(0)-shift)*.1)


if __name__=='__main__':unittest.main()
