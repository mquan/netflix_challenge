from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import upstream_cf48_102_context as c


class CF48ContextTest(unittest.TestCase):
    def test_new_signal_layout_and_actual_saved_model_replay(self):
        targets=[(c.base,['RUN','REPORT','CONFIG','SIGNALS','CONTEXT','model_factory','code_hashes','prepare','predecessors','__file__']),
                 (c.n,['RUN','REPORT','CONFIG','code_hashes','train_nn']),
                 (c.base.context,['model_factory']),(c.base.verifier,['neural_model']),
                 (c.base.preceding,['RUN','REPORT'])]
        saved=[(obj,{key:getattr(obj,key) for key in keys}) for obj,keys in targets]
        try:
            self.assertEqual(sorted(c.SIGNALS+c.CONTEXT),list(range(239)))
            self.assertTrue({229,237,238}.issubset(c.SIGNALS))
            with self.assertRaises(ValueError):c.model_factory(228,c.CONFIG)
            c.configure();rng=np.random.default_rng(81)
            features=rng.normal(size=(16,239)).astype(np.float32);labels=rng.normal(size=16).astype(np.float32)
            model,epochs=c.base.context.train_nn(features,labels,c.CONFIG,13,2)
            self.assertEqual(epochs,2);expected=c.n.nn_predict(model,features)
            self.assertTrue(np.isfinite(expected).all());self.assertTrue(np.any(expected!=0))
            with tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/'nn.pt';torch.save(dict(feature_width=239,config=c.CONFIG,state=model.state_dict()),path)
                restored=c.base.neural_model(path)
                np.testing.assert_array_equal(c.n.nn_predict(restored,features),expected)
        finally:
            for obj,fields in saved:
                for key,value in fields.items():setattr(obj,key,value)


if __name__=='__main__':unittest.main()
