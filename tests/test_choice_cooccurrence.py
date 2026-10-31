from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from choice_cooccurrence import choice_matrix,embeddings,fit_pair
from choice_outer_validation import fit_outer


class CooccurrenceTests(unittest.TestCase):
    def test_presence_matrix_idf_normalization_and_rating_independence(self):
        d=dict(users=3,items=6,train_offsets=np.arange(4),train_items=np.array([1,2,3]),
            train_ratings=np.array([1,3,5]),probe=np.array([2,4,1]),
            implicit_offsets=np.arange(4)*4,
            implicit_items=np.array([1,4,4,5,2,4,5,6,3,5,6,6]),implicit_days=np.arange(12))
        a=choice_matrix(d)
        presence=np.array([[0,0,0,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]],dtype=float)
        expected=presence*np.log1p(3/(1+presence.sum(0)))
        expected/=np.linalg.norm(expected,axis=1,keepdims=True)
        np.testing.assert_allclose(a.toarray(),expected,rtol=0,atol=1e-12)
        self.assertEqual(a.nnz,7)
        changed=dict(d,train_ratings=np.array([5,1,2]),probe=np.array([5,5,5]))
        np.testing.assert_array_equal(a.toarray(),choice_matrix(changed).toarray())
        x,y,s=embeddings(a,rank=2,oversample=1,power=2)
        xx,yy,ss=embeddings(choice_matrix(changed),rank=2,oversample=1,power=2)
        for first,second in [(x,xx),(y,yy),(s,ss)]:np.testing.assert_array_equal(first,second)

    def test_randomized_factorization_matches_exact_low_rank_geometry(self):
        rng=np.random.default_rng(17)
        a=rng.uniform(.1,1,(12,2))@rng.uniform(.1,1,(2,9))
        raw,unit,s=embeddings(sparse.csr_matrix(a),rank=2,oversample=2,power=2,seed=19)
        _,exact,vt=np.linalg.svd(a,full_matrices=False)
        np.testing.assert_allclose(s,exact[:2],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(raw@raw.T,(vt[:2].T*exact[:2])@vt[:2],rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(unit,axis=1),1,rtol=0,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            embeddings(sparse.csr_matrix(a),rank=20)

    def test_nested_control_replay_and_outer_label_independence(self):
        rng=np.random.default_rng(19);folds=np.tile(np.arange(5),200)
        y=rng.integers(1,6,len(folds)).astype(float)
        bank=y[:,None]+rng.normal(0,.8,(len(y),3))
        control=rng.normal(size=(len(y),4));extra=rng.normal(size=(len(y),3))
        params=dict(objective='regression',num_leaves=7,min_data_in_leaf=20,
            feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,seed=19,num_threads=1,verbose=-1)
        original,_,_=fit_outer(bank,control,y,folds,params,rounds=20,patience=5)
        pred,iterations=fit_pair(bank,control,extra,y,folds,params,rounds=20,patience=5)
        np.testing.assert_array_equal(pred['h8'],original['choice_tree'])
        changed=y.copy();changed[folds==0]=(changed[folds==0]%5)+1
        other,counts=fit_pair(bank,control,extra,changed,folds,params,rounds=20,patience=5)
        self.assertEqual(iterations,counts)
        for key in pred:np.testing.assert_array_equal(pred[key],other[key])


if __name__=='__main__':unittest.main()
