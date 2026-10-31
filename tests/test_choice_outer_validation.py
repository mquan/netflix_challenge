from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from choice_outer_validation import stacked_bases,fit_outer
from headroom_choices import features
from gbdt_blend import PROBE


class OuterValidationTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(19)
        folds=np.tile(np.arange(5),200)
        y=rng.integers(1,6,len(folds)).astype(float)
        bank=y[:,None]+rng.normal(0,.8,(len(y),3))
        choices=rng.normal(size=(len(y),4))
        return bank,choices,y,folds

    def test_outer_labels_cannot_change_any_fitted_predictions_or_round_selection(self):
        bank,choices,y,folds=self.data()
        params=dict(objective='regression',learning_rate=.05,num_leaves=7,min_data_in_leaf=20,
            feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,seed=20261031,
            num_threads=1,verbose=-1)
        pred,rounds,_=fit_outer(bank,choices,y,folds,params,rounds=30,patience=5)
        changed=y.copy();changed[folds==0]=(changed[folds==0]%5)+1
        other,other_rounds,_=fit_outer(bank,choices,changed,folds,params,rounds=30,patience=5)
        self.assertEqual(rounds,other_rounds)
        for key in pred:np.testing.assert_array_equal(pred[key],other[key])
        inner,outer=stacked_bases(bank,y,folds)
        b,c=stacked_bases(bank,changed,folds)
        np.testing.assert_array_equal(inner,b);np.testing.assert_array_equal(outer,c)

    def test_selection_labels_cannot_enter_inner_training_base(self):
        bank,_,y,folds=self.data()
        a,_=stacked_bases(bank,y,folds)
        changed=y.copy();changed[folds==1]=(changed[folds==1]%5)+1
        b,_=stacked_bases(bank,changed,folds)
        np.testing.assert_array_equal(a[folds>=1],b[folds>=1])
        self.assertTrue(np.isnan(a[folds==0]).all())
        changed=y.copy();changed[folds==2]=(changed[folds==2]%5)+1
        b,_=stacked_bases(bank,changed,folds)
        np.testing.assert_array_equal(a[folds==2],b[folds==2])
        self.assertFalse(np.array_equal(a[folds==3],b[folds==3]))

    def test_choice_features_ignore_all_query_rating_labels(self):
        probe=np.zeros(4,dtype=PROBE)
        probe['user']=[0,0,1,1];probe['item']=[5,6,7,8]
        probe['day']=[3,4,3,4];probe['rating']=3
        d=dict(users=2,items=40,probe=probe,train_offsets=np.array([0,2,4]),
            train_items=np.array([1,2,3,4]),train_days=np.array([1,2,1,2]),
            train_ratings=np.array([1,2,3,4]),implicit_offsets=np.array([0,4,8]),
            implicit_items=np.array([1,2,5,6,3,4,7,8]),implicit_days=np.array([1,2,3,4,1,2,3,4]))
        rng=np.random.default_rng(17)
        q=rng.normal(size=(40,32));emb=rng.normal(size=(40,32))
        a,names=features(d,q,emb)
        changed=dict(d,probe=d['probe'].copy());changed['probe']['rating']=5
        b,other=features(changed,q,emb)
        np.testing.assert_array_equal(a,b);self.assertEqual(names,other)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a.shape,(len(d['probe']),12))


if __name__ == '__main__':
    unittest.main()
