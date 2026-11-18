"""Exercise the new neural path without loading either native training tree."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_context_blend as c
from verify_upstream_context_partial import independent_rmse


class ContextBlendTest(unittest.TestCase):
    def test_training_and_saved_model_prediction(self):
        import torch
        torch.set_num_threads(2);torch.manual_seed(23)
        features=np.random.default_rng(23).normal(size=(256,222)).astype(np.float32)
        original=features.copy()
        target=(.5+.1*features[:,0]*features[:,63]).astype(np.float32)
        model=c.model_factory(222,c.CONFIG)
        np.testing.assert_array_equal(c.n.nn_predict(model,features),np.zeros(256))
        model,epochs=c.train_nn(features,target,c.CONFIG,23,2)
        expected=c.n.nn_predict(model,features)
        self.assertEqual(epochs,2)
        self.assertLess(np.mean((expected-target)**2),np.mean(target**2))
        self.assertGreater(float(model.gate[-1].weight.detach().abs().sum()),0)
        np.testing.assert_array_equal(features,original)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            torch.save(dict(config=c.CONFIG,feature_width=222,state=model.state_dict()),path)
            actual=c.n.nn_predict(c.neural_model(path),features)
            np.testing.assert_array_equal(actual,expected)
            wrong=dict(c.CONFIG,context_rank=99)
            torch.save(dict(config=wrong,feature_width=222,state=model.state_dict()),path)
            with self.assertRaisesRegex(ValueError,'recipe changed'):c.neural_model(path)

    def test_width_and_semantic_partition(self):
        with self.assertRaisesRegex(ValueError,'layout'):c.model_factory(228,c.CONFIG)
        # The five query-context columns and 130 voting columns never enter
        # the predictor branch; the appended training-only ridge does.
        self.assertEqual(set(c.CONTEXT),set(range(63,68))|set(range(91,221)))
        self.assertEqual(set(c.SIGNALS),set(range(222))-set(c.CONTEXT))
        self.assertEqual(len(set(c.SIGNALS)&set(c.CONTEXT)),0)

    def test_partial_scoring_clips_after_double_conversion(self):
        # Float32 cannot represent the exact clipping limits. Both values
        # below lie just outside them, so the exact residual magnitude is .05.
        upper=np.nextafter(np.float32(4.95),np.float32(np.inf))
        prediction=np.array([np.float32(1.05),upper],dtype=np.float32)
        self.assertAlmostEqual(independent_rmse(prediction,[1.,5.]),.05,places=14)
        with self.assertRaisesRegex(ValueError,'Malformed'):
            independent_rmse(prediction,[1.])


if __name__=='__main__':unittest.main()
