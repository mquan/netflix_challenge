from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import upstream_extension_context as c


class ExtensionContextTest(unittest.TestCase):
    def test_layout_and_saved_trained_model_replay(self):
        self.assertEqual(sorted(c.SIGNALS+c.CONTEXT),list(range(228)))
        self.assertTrue(set(range(221,228)).issubset(c.SIGNALS))
        self.assertFalse(set(c.SIGNALS)&set(c.CONTEXT))
        with self.assertRaises(ValueError):c.model_factory(223,c.CONFIG)
        with self.assertRaises(ValueError):c.model_factory(228,dict(c.CONFIG,context_rank=8))
        c.configure();rng=np.random.default_rng(55)
        features=rng.normal(size=(16,228)).astype(np.float32)
        target=rng.normal(size=16).astype(np.float32)
        model,epochs=c.context.train_nn(features,target,c.CONFIG,9,2)
        self.assertEqual(epochs,2)
        model.eval();expected=c.n.nn_predict(model,features)
        self.assertTrue(np.isfinite(expected).all());self.assertTrue(np.any(expected!=0))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'nn.pt'
            torch.save(dict(feature_width=228,config=c.CONFIG,state=model.state_dict()),path)
            restored=c.neural_model(path)
            np.testing.assert_array_equal(c.n.nn_predict(restored,features),expected)


if __name__=='__main__':unittest.main()
