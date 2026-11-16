import sys
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from streaming_nn_blend import standardized, train_device, predict_device
from round8_nn_blend import standardize
from feature_adaptation import relative_unrated


class NormalizationTests(unittest.TestCase):
    def test_relative_missing_count_replacement_matches_explicit_features(self):
        rng=np.random.default_rng(94);x=rng.normal(size=(60,16)).astype('f4')
        x[:,3]=np.log1p(20+np.arange(60));x[:,-1]=np.log1p(9)
        bank=dict(names=['prediction','support'],extra_meta_names=['a','log_unrated_n'])
        column,values=relative_unrated(x,bank)
        np.testing.assert_allclose(values,np.log1p(9/(25+np.arange(60))),rtol=4e-7)
        explicit=x.astype(np.float64);explicit[:,column]=values
        fit=np.arange(60)%3!=0
        np.testing.assert_array_equal(standardized(x,fit,replacement_columns={column:values}),standardized(explicit,fit))
    def test_suppressed_feature_is_zero_in_fit_and_query_transform(self):
        rng=np.random.default_rng(77);x=rng.normal(size=(100,7)).astype('f4');mask=np.arange(100)%3!=0
        explicit=x.copy();explicit[:,3]=0
        actual,scaler=standardized(x,mask,zero_columns=[3],return_scaler=True)
        np.testing.assert_array_equal(actual,standardized(explicit,mask))
        self.assertEqual(scaler['lo'][3],0);self.assertEqual(scaler['hi'][3],0)
        shifted=x+100
        query=(np.clip(shifted,scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std']
        np.testing.assert_array_equal(query[:,3],0)

    def test_weighted_full_batch_step_matches_replicating_examples(self):
        rng=np.random.default_rng(52);x=rng.normal(size=(16,4)).astype('f4');y=rng.normal(size=16)
        sw=np.tile([1,2],8)
        a=SimpleNamespace(hidden=8,layers=2,dropout=0.,lr=.0003,weight_decay=.01,
                          batch=1000,patience=5,device='cpu')
        model,_=train_device(x,y,a,83,1,sample_weights=sw)
        repeated,_=train_device(np.repeat(x,sw,axis=0),np.repeat(y,sw),a,83,1)
        np.testing.assert_allclose(predict_device(model,x,'cpu'),predict_device(repeated,x,'cpu'),atol=1e-7)

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_resident_gather_matches_host_gather(self):
        rng=np.random.default_rng(4);x=np.asfortranarray(rng.normal(size=(128,9)).astype('f4'));y=rng.normal(size=128)
        a=SimpleNamespace(hidden=16,layers=2,dropout=0.,lr=.0003,weight_decay=.01,
                          batch=64,patience=5,device='mps',resident_features=False)
        host,_=train_device(x,y,a,65,2);a.resident_features=True
        resident,_=train_device(x,y,a,65,2)
        np.testing.assert_array_equal(predict_device(host,x,'mps'),predict_device(resident,x,'mps'))

    def test_memory_and_disk_scaling_match_original(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(1000,37)); x[:,0]=1
        fit=np.arange(1000)%5!=0
        expected=standardize(x,fit)
        np.testing.assert_array_equal(standardized(x,fit,block=7),expected)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'norm.npy'
            actual=standardized(np.asfortranarray(x),fit,block=7,output=p)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(np.load(p),expected)

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_numeric_agreement_without_dropout(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(128,9)).astype('f4'); y=rng.normal(size=128)
        a=SimpleNamespace(hidden=16,layers=2,dropout=0.,lr=.0003,weight_decay=.01,
                          batch=64,patience=5,device='cpu')
        cpu,_=train_device(x,y,a,12,2)
        a.device='mps'
        gpu,_=train_device(x,y,a,12,2)
        np.testing.assert_allclose(predict_device(gpu,x,'mps'),predict_device(cpu,x,'cpu'),atol=2e-5)


if __name__=='__main__': unittest.main()
