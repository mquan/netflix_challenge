from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_exact391_scaler as bounded
import upstream_own391_paired_mps as study
import verify_upstream_own391_paired_mps as independent


class Matched391Recipe(unittest.TestCase):
    def original(self,values,mask):
        selected=values[mask];lo,hi=np.percentile(selected,[1,99],axis=0)
        clipped=np.clip(selected,lo,hi).astype(np.float64)
        return dict(lo=lo,hi=hi,mean=clipped.mean(0),std=np.maximum(clipped.std(0),1e-6))

    def test_all_original_and_independent_statistics_match(self):
        rng=np.random.default_rng(391);values=rng.normal(size=(3007,618)).astype(np.float32)
        values[:,0]=0;values[:,1]=7;values[:,2]=np.round(values[:,2]);values[:,3]=np.float32(-0.)
        mask=rng.random(len(values))<.7;expected=self.original(values,mask)
        for actual in [bounded.fit_scaler(values,mask,1),bounded.fit_scaler(values,mask,257),bounded.fit_scaler(values,mask),independent.independent_scaler(values,mask)]:
            for key in expected:np.testing.assert_array_equal(actual[key].view('u8'),expected[key].view('u8'))

    def test_large_cancellation_and_sparse_eligibility_preserve_original_order(self):
        rng=np.random.default_rng(2026);values=(rng.normal(size=(2033,618))*1e12).astype(np.float32);values[::2]*=-1
        mask=np.arange(len(values))%7==0;expected=self.original(values,mask)
        for actual in [bounded.fit_scaler(values,mask,63),independent.independent_scaler(values,mask)]:
            for key in expected:np.testing.assert_array_equal(actual[key].view('u8'),expected[key].view('u8'))

    def test_reduced_population_or_wrong_precision_rejected(self):
        mask=np.ones(5,dtype=bool)
        for values in [np.zeros((5,617),dtype=np.float32),np.zeros((5,618),dtype=np.float64)]:
            with self.assertRaises(ValueError):bounded.fit_scaler(values,mask)

    def test_full_partition_and_both_original_budget_arms(self):
        import torch
        self.assertEqual(sorted(study.SIGNALS+study.CONTEXT),list(range(618)))
        self.assertEqual((len(study.SIGNALS),len(study.CONTEXT)),(393,225))
        for name,hidden in [('h128',128),('h512',512)]:
            cfg=study.CONFIGS[name]
            self.assertEqual((cfg['nn_hidden'],cfg['context_rank'],cfg['nn_epochs'],cfg['nn_patience'],cfg['nn_batch']),(hidden,32,40,4,8192))
            self.assertEqual(cfg['seeds'],[2026,7,13])
            model=study.model_factory(618,cfg)
            with torch.no_grad():torch.testing.assert_close(model(torch.randn(5,618)),torch.zeros((5,1)),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
