"""Check the extra-signal layout, model replay and held-user isolation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_cf12_context as m


class Cf12ContextTest(unittest.TestCase):
    def test_signal_context_partition_and_width(self):
        self.assertEqual(set(m.SIGNALS)|set(m.CONTEXT),set(range(223)))
        self.assertFalse(set(m.SIGNALS)&set(m.CONTEXT))
        self.assertIn(221,m.SIGNALS)  # Newly appended CF12 prediction.
        self.assertIn(222,m.SIGNALS)  # New training-only ridge prediction.
        self.assertEqual(m.CONTEXT,m.context.CONTEXT)
        with self.assertRaisesRegex(ValueError,'layout'):m.model_factory(222,m.CONFIG)

    def test_training_and_saved_model_replay(self):
        import torch
        torch.set_num_threads(2);torch.manual_seed(23)
        values=np.random.default_rng(23).normal(size=(256,223)).astype(np.float32)
        target=(.5+.1*values[:,221]*values[:,63]).astype(np.float32)
        model=m.model_factory(223,m.CONFIG)
        np.testing.assert_array_equal(m.n.nn_predict(model,values),np.zeros(256))
        with patch.object(m.context,'model_factory',m.model_factory):
            model,count=m.context.train_nn(values,target,m.CONFIG,23,2)
        expected=m.n.nn_predict(model,values)
        self.assertEqual(count,2)
        self.assertLess(np.mean((expected-target)**2),np.mean(target**2))
        self.assertGreater(float(model.gate[-1].weight.detach().abs().sum()),0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pt';torch.save(dict(config=m.CONFIG,feature_width=223,state=model.state_dict()),p)
            np.testing.assert_array_equal(m.n.nn_predict(m.neural_model(p),values),expected)
            torch.save(dict(config=dict(m.CONFIG,context_rank=3),feature_width=223,state=model.state_dict()),p)
            with self.assertRaisesRegex(ValueError,'recipe changed'):m.neural_model(p)

    def test_outer_scoring_labels_cannot_change_actual_fold_fit(self):
        rng=np.random.default_rng(42);rows=120
        features=rng.normal(size=(rows,222)).astype(np.float32)
        z=rng.normal(size=(rows,162));z[:,-1]=1
        groups=np.tile(np.arange(10),rows//10)
        labels=3.6+.05*features[:,221]
        changed=labels.copy();changed[groups//2==0]=100
        cfg=dict(m.CONFIG,seeds=[23],nn_epochs=2,nn_patience=1,nn_batch=32)
        records=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(m.context,'model_factory',m.model_factory),patch.object(m.n,'train_nn',m.context.train_nn):
            for i,y in enumerate([labels,changed]):
                grams,rhs=m.n.grouped_stats(z,y,groups);root=Path(tmp)/str(i)
                records.append(m.n.fit_fold(features,z,y,groups,grams,rhs,0,cfg,'nn',root))
            self.assertEqual(records[0]['chosen_iterations'],records[1]['chosen_iterations'])
            for name in ['prediction.f32','ridge.npz','scalers.npz']:
                self.assertEqual((Path(tmp)/'0'/name).read_bytes(),(Path(tmp)/'1'/name).read_bytes())


if __name__=='__main__':unittest.main()
