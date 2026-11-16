import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ridge_selection import candidates,choose,refit
from streaming_blend import grouped_statistics,design


class RidgeSelectionTests(unittest.TestCase):
    def test_statistics_solution_matches_direct_penalized_fit(self):
        rng=np.random.default_rng(731)
        p=rng.normal(size=(200,3));c=np.column_stack([np.ones(200),rng.normal(size=(200,4))])
        x=design(p,c);y=rng.normal(size=200)+3.6
        g=x.T@x;r=x.T@(y-3.6)
        for recipe,w in candidates(g,r,len(y)):
            width=4 if recipe['context']=='global' else len(r)
            penalty=np.eye(width)*len(y)*recipe['alpha'];penalty[0,0]=0
            expected=np.linalg.solve(g[:width,:width]+penalty,r[:width])
            np.testing.assert_allclose(w[:width],expected,rtol=1e-6,atol=1e-6)
            np.testing.assert_array_equal(w[width:],0)

    def test_outer_labels_cannot_change_selection_or_refit(self):
        rng=np.random.default_rng(920)
        p=rng.normal(size=(300,2));c=np.column_stack([np.ones(300),rng.normal(size=(300,4))])
        groups=np.arange(300)%10;y=rng.normal(size=300)+3.6
        first=grouped_statistics(p,c,y,groups)
        changed=y.copy();changed[groups//2==2]+=1000
        second=grouped_statistics(p,c,changed,groups)
        ids=np.arange(10);train=ids//2!=2
        recipe=choose(first,train&(ids%2==0),train&(ids%2==1))
        self.assertEqual(recipe,choose(second,train&(ids%2==0),train&(ids%2==1)))
        np.testing.assert_array_equal(refit(first,train,recipe),refit(second,train,recipe))


if __name__=='__main__': unittest.main()
