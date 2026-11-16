import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from public_newsplit import choose_moved
from gbdt_blend import PROBE
from newsplit_weights import inclusion_weights


class NewSplitTests(unittest.TestCase):
    def test_analytic_weights_recover_population_in_expectation_and_ignore_labels(self):
        q=np.zeros(6,dtype=PROBE);q['user']=[0,0,1,1,1,2];q['day']=[2,1,2,3,1,5];q['rating']=1
        d=dict(users=3,probe=q,train_offsets=np.array([0,1,2,3]),train_days=np.array([1,1,1]),
               train_ratings=np.ones(3))
        w=inclusion_weights(d)
        np.testing.assert_allclose(w,[10,1,1/.55,1/.55,1,1])
        d['probe']['rating'][:]=5;d['train_ratings'][:]=100
        np.testing.assert_array_equal(w,inclusion_weights(d))
        values=np.array([2.,7.,4.]);sw=np.array([1/.55,1/.55,1.])
        expectation=.1*np.sum(sw*values)+.45*np.sum((sw*values)[[1,2]])+.45*np.sum((sw*values)[[0,2]])
        self.assertAlmostEqual(expectation,values.sum())

    def test_selection_ignores_values_and_respects_new_day_and_multi_rating_rules(self):
        q=np.zeros(9,dtype=PROBE)
        q['user']=[0,0,1,1,2,2,2,3,4]
        q['item']=np.arange(1,10)
        q['day']=[2,3,1,1,2,3,4,4,4]
        q['rating']=1
        d=dict(users=5,train_offsets=np.array([0,1,2,3,4,5]),
               train_days=np.array([1,1,1,1,1]),train_ratings=np.ones(5,dtype='u1'),probe=q)
        original_map=np.arange(9,dtype='u4')
        moved=choose_moved(d,original_map)
        self.assertTrue(set(q['user'][moved]).issubset({0,2}))
        self.assertEqual(len(np.unique(q['user'][moved])),len(moved))
        d['train_ratings'][:]=5; d['probe']['rating'][:]=5
        np.testing.assert_array_equal(moved,choose_moved(d,original_map))
        self.assertTrue(np.all(q['day'][moved]!=1))


if __name__=='__main__': unittest.main()
