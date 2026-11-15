import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from item_calibration import moments, fit


class ItemTests(unittest.TestCase):
    def test_itemwise_fit_and_label_isolation(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(500,7)); y=rng.normal(size=500)
        items=np.arange(500)%11; mask=np.arange(500)%5!=0
        beta=fit(moments(x,y,items,mask,11),10)
        for i in range(11):
            take=mask & (items==i)
            expected=np.linalg.solve(x[take].T@x[take]+10*np.eye(7),x[take].T@y[take])
            np.testing.assert_allclose(beta[i],expected,atol=1e-12)
        y[~mask]=1000
        np.testing.assert_array_equal(beta,fit(moments(x,y,items,mask,11),10))


if __name__=='__main__': unittest.main()
