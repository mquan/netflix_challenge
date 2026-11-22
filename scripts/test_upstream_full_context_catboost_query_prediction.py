"""Real CPU saved-model inference; synthetic blocks, no production admission."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_full_context_catboost_query_prediction as module
import upstream_full_context_catboost_artifacts as artifacts


class PredictionTests(unittest.TestCase):
    def test_actual_five_saved_models_zero_slots_and_ordered_fold_bag_bits(self):
        guard=lambda **kw:None
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();rng=np.random.default_rng(318)
            x=rng.normal(size=(23,713)).astype('<f4');x[:,-1]=0
            b=np.empty((5,2,len(x)),dtype='<f8')
            folds=[]
            for fold in range(5):
                b[fold,0]=1.3+fold*.1;b[fold,1]=2.1+fold*.2
                matrix=x.copy();matrix[:,-1]=b[fold,1].astype('<f4')
                y=np.asarray(.4*matrix[:,0]+.1*matrix[:,2],dtype='<f8')
                model=module.backend.train(matrix,y,2026+fold,3,guard)
                record=artifacts.save_model(root/('fold-%s.cbm'%fold),model,seed=2026+fold,iterations=3,width=713,root=root,guard=guard)
                folds.append(dict(fold=fold,seed_order=[2026,7,13],seed_offset=fold,chosen_iterations=[3,0,0],models={'2026':record},qualifying_labels_read=False))
            original=x.tobytes();base=b.tobytes()
            with module.models(folds,713,guard) as loaded:
                vectors,bag=module.predict(x,b,loaded,guard)
                expected=[]
                for fold in range(5):
                    matrix=x.copy();matrix[:,-1]=b[fold,1].astype('<f4')
                    correction=module.backend.predict(loaded[fold][2026],matrix,guard)/3
                    expected.append(np.clip(b[fold,1]+correction,1.05,4.95).astype('<f4'))
                for a,c in zip(vectors,expected):self.assertEqual(a.tobytes(),c.tobytes())
                total=np.zeros(len(x),dtype='<f8')
                for value in expected:total+=value.astype('<f8')
                self.assertEqual(bag.tobytes(),np.clip(total/5,1.05,4.95).astype('<f4').tobytes())
            self.assertEqual(x.tobytes(),original);self.assertEqual(b.tobytes(),base)
            changed=copy.deepcopy(folds);changed[2]['models']={}
            with self.assertRaises(ValueError):
                with module.models(changed,713,guard):self.fail('Missing saved positive slot admitted')

    def test_all_zero_slots_keep_baseline_and_both_denominators_at_8192_rows(self):
        x=np.zeros((8192,864),dtype='<f4');b=np.full((5,2,len(x)),2.2,dtype='<f8')
        folds=[dict(fold=f,seed_order=[2026,7,13],seed_offset=f,chosen_iterations=[0,0,0],models={},qualifying_labels_read=False) for f in range(5)]
        with module.models(folds,864,lambda:None) as loaded:
            values,bag=module.predict(x,b,loaded,lambda:None)
            for v in values:self.assertEqual(v.tobytes(),np.full(len(x),2.2,dtype='<f4').tobytes())
            self.assertEqual(bag.tobytes(),values[0].tobytes())
            with self.assertRaises(ValueError):module.predict(np.zeros((8193,864),dtype='<f4'),np.full((5,2,8193),2.2,dtype='<f8'),loaded,lambda:None)


if __name__=='__main__':unittest.main()
