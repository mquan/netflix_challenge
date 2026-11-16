import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from projected_fwls import encoding,feature_stats,solutions,fit
from streaming_blend import grouped_statistics


class ProjectedFwlsTests(unittest.TestCase):
    def test_disagreement_encoding_and_penalized_solution(self):
        rng=np.random.default_rng(92)
        p=rng.normal(size=(250,12))+3.6
        raw=np.column_stack([p,rng.normal(size=(250,3))]).astype(np.float32)
        c=np.ones((250,1));y=rng.normal(size=250)+3.6;groups=np.arange(250)%10
        g,r,n=grouped_statistics(p,c,y,groups)
        scaler=feature_stats(raw,12,dict(gram=g))
        x=encoding(raw,12,scaler)
        self.assertEqual(x.shape,(250,1+12+3+9*3))
        self.assertTrue(np.isfinite(x).all())
        gram=x.T@x;rhs=x.T@(y-3.6)
        for alpha,w in solutions(gram,rhs,len(y)):
            penalty=np.eye(x.shape[1])*len(y)*alpha;penalty[0,0]=0
            np.testing.assert_allclose(w,np.linalg.solve(gram+penalty,rhs),rtol=1e-5,atol=1e-6)

    def test_inner_choice_ignores_other_user_groups(self):
        rng=np.random.default_rng(73);x=np.column_stack([np.ones(100),rng.normal(size=(100,4))])
        y=rng.normal(size=100);groups=np.arange(100)%10
        g=np.array([x[groups==k].T@x[groups==k] for k in range(10)])
        r=np.array([x[groups==k].T@y[groups==k] for k in range(10)])
        n=np.bincount(groups);train=np.arange(10)<4;valid=(np.arange(10)>=4)&(np.arange(10)<6)
        alpha,w=fit((g,r,n),train,valid)
        altered=r.copy();altered[6:]+=1000
        other,ow=fit((g,altered,n),train,valid)
        self.assertEqual(alpha,other);np.testing.assert_array_equal(w,ow)


if __name__=='__main__': unittest.main()
