"""Synthetic full-width CPU terminal-model extraction; no actual device/data."""
import copy
from dataclasses import asdict
import hashlib
import os
import unittest
from unittest.mock import patch
import numpy as np
from test_upstream_factored_rbm_rows import fixture
from upstream_factored_rbm_rows import UserRows
from upstream_factored_rbm_training_metal import Config,State
import upstream_factored_rbm_terminal_metal as terminal
import upstream_rbm400_latent_extraction as extraction

DIGEST='1'*64


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
        tr,q=fixture();cls.reader=UserRows(tr,q,5)
        state=State(cls.reader,DIGEST,Config(),'cpu')
        for epoch in range(1,16):state.fit_epoch(epoch,lambda:None)
        cls.config=asdict(Config());cls.record=terminal.freeze(state,'known',DIGEST,cls.config)
        cls.expected=terminal.replay(cls.record,cls.reader,'known',DIGEST,cls.config,lambda:None,'cpu')

    def run_extract(self,record=None,expected=None,reader=None,guard=lambda:None):
        return extraction.extract(record or self.record,reader or self.reader,'known',DIGEST,self.config,
                                  self.expected if expected is None else expected,guard,'cpu')

    def test_full50factor_caches_and_native_query_bits(self):
        value=self.run_extract();meta=value['metadata']
        self.assertEqual(value['users'].shape,(4,50));self.assertEqual(value['items'].shape,(5,50))
        self.assertTrue(value['covered'].all());self.assertEqual(meta['native_query_rows'],6)
        self.assertEqual(meta['cold_query_users'],2);self.assertTrue(meta['raw_prediction_bits_verified'])
        self.assertFalse(meta['actual_MPS_predictions_replayed']);self.assertFalse(meta['production_provider_admitted'])
        self.assertFalse(meta['qualifying_labels_read']);self.assertEqual(meta['models_fitted'],0)

    def test_empty_histories_keep_bias_and_conditional_addresses(self):
        value=self.run_extract();model={k:v.numpy().astype(np.float64) for k,v in self.record['model'].items()}
        # User1 has no training rows, query movie2. User3 has only a masked
        # training movie1 plus query movie0. Both must retain conditional D.
        for user,items in [(1,[2]),(3,[0,1])]:
            hidden=2/(1+np.exp(-(model['bh']+model['d'][items].sum(axis=0))))-1
            expected=hidden@model['q'].T
            np.testing.assert_allclose(value['users'][user],expected,rtol=2e-5,atol=2e-7)
            self.assertGreater(float(np.linalg.norm(value['users'][user])),1e-8)

    def test_all_user_factors_against_independent_scalar_posterior(self):
        from test_upstream_factored_rbm_prediction import scalar
        batch=self.reader.batch(np.arange(self.reader.n_users,dtype=np.int64))
        model={k:v.numpy().astype(np.float64) for k,v in self.record['model'].items()}
        posterior=scalar(model,batch['support'],batch['query'])['hidden_expected']
        expected=posterior@model['q'].T
        np.testing.assert_allclose(self.run_extract()['users'],expected,rtol=2e-5,atol=2e-7)

    def test_model_tensors_and_terminal_rng_remain_unchanged(self):
        def digest():return hashlib.sha256(b''.join(v.numpy().tobytes() for v in self.record['model'].values())+self.record['terminal_rng_state'].numpy().tobytes()).hexdigest()
        before=digest();a=self.run_extract();b=self.run_extract()
        self.assertEqual(before,digest());np.testing.assert_array_equal(a['users'],b['users'])
        np.testing.assert_array_equal(a['items'],b['items'])

    def test_one_bit_prediction_mismatch_is_not_tolerated(self):
        wrong=self.expected.copy();wrong[3]=np.nextafter(wrong[3],np.float32(np.inf))
        with self.assertRaises(ValueError):self.run_extract(expected=wrong)

    def test_literal_native_bit_check_distinguishes_signed_zero(self):
        original=extraction.predict
        def signed_zero(*args,**kwargs):
            value=original(*args,**kwargs)
            value['prediction']=self.torch.zeros_like(value['prediction'])
            value['prediction'][0]=-0.0
            return value
        # Signed zero cannot arise from the admitted rating expectation;
        # this fixture still checks the literal uint32 comparison contract.
        with patch.object(extraction,'predict',side_effect=signed_zero):
            with self.assertRaises(ValueError):self.run_extract(expected=np.zeros(6,dtype=np.float32))

    def test_partial_or_wrong_phase_terminal_model_rejects(self):
        for key,value in [('completed_epoch',14),('completed_batches',0),('phase','qualifying')]:
            wrong=copy.deepcopy(self.record);wrong[key]=value
            with self.assertRaises(ValueError):self.run_extract(record=wrong)

    def test_query_outcomes_are_never_requested(self):
        tr,q=fixture()
        class NoLabels(dict):
            def __getitem__(self,key):
                if key=='ratings':raise AssertionError('Forbidden query outcomes')
                return super().__getitem__(key)
        reader=UserRows(tr,NoLabels(q),5);self.run_extract(reader=reader)

    def test_users_without_queries_remain_uncovered_with_nan_cache(self):
        tr,q=fixture();keep=q['user_idxs']!=1
        q={k:v[keep] if k!='user_cnts' else np.array([2,0,2,1],dtype=np.int32) for k,v in q.items()}
        reader=UserRows(tr,q,5)
        expected=terminal.replay(self.record,reader,'known',DIGEST,self.config,lambda:None,'cpu')
        value=self.run_extract(reader=reader,expected=expected)
        np.testing.assert_array_equal(value['covered'],[True,False,True,True])
        self.assertTrue(np.isnan(value['users'][1]).all())
        self.assertEqual(value['metadata']['native_query_rows'],5)

    def test_duplicate_or_outside_query_row_addresses_are_rejected(self):
        for bad in (np.array([0,0,2,3,4,5]),np.array([-1,1,2,3,4,5])):
            reader=copy.copy(self.reader);original=self.reader.batch
            def broken(ids,original=original,bad=bad):
                value=original(ids);value['query_rows']=bad;return value
            reader.batch=broken
            with self.assertRaises(ValueError):self.run_extract(reader=reader)

    def test_wrong_native_vector_and_resource_failure_reject_without_result(self):
        for expected in (self.expected.astype(np.float64),self.expected[:-1],np.full(6,np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):self.run_extract(expected=expected)
        def fail():raise RuntimeError('Synthetic resource failure')
        with self.assertRaises(RuntimeError):self.run_extract(guard=fail)

    def test_mps_fallback_rejected_before_any_gpu_operation(self):
        with patch.dict(os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'1','PYTORCH_MPS_FAST_MATH':'0'}):
            with self.assertRaises(ValueError):extraction.extract(self.record,self.reader,'known',DIGEST,self.config,self.expected,lambda:None,'mps')


if __name__=='__main__':unittest.main()
