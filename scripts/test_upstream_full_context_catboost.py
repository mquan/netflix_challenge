"""Small real CPU backend/serialization fixtures; no full-source study admission."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_full_context_catboost as backend
from upstream_factored_rbm_admission import file_identity


class BackendTests(unittest.TestCase):
    def data(self,rows=96,width=713):
        rng=np.random.default_rng(128);x=rng.normal(size=(rows,width)).astype('<f4')
        return x,(.5*x[:,0]-.25*x[:,1]).astype('<f8')

    def test_original_budget_and_process_isolation_are_explicit(self):
        self.assertEqual(backend.MAX_ROUNDS,2000);self.assertEqual(backend.PATIENCE,100)
        self.assertEqual(backend.PARAMETERS['depth'],8);self.assertEqual(backend.PARAMETERS['thread_count'],2)
        self.assertFalse(backend.PARAMETERS['allow_writing_files'])
        with patch.dict(sys.modules,{'torch':object()}):
            with self.assertRaisesRegex(ValueError,'isolated'):backend.runtime()
        with patch.object(backend.importlib.metadata,'version',return_value='0.0'):
            with self.assertRaises(ValueError):backend.runtime()

    def test_real_refit_callbacks_and_authenticated_saved_prediction_bits(self):
        x,y=self.data();calls=[];guard=lambda:calls.append(1)
        model=backend.train(x,y,2026,4,guard)
        self.assertEqual(model.tree_count_,4);self.assertGreaterEqual(len(calls),7)
        query,_=self.data(8193)
        expected=model.predict(query,thread_count=2)
        self.assertEqual(backend.predict(model,query,guard).tobytes(),expected.tobytes())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'model.cbm';model.save_model(str(path))
            record={'identity':file_identity(path.stat())};params=backend.model_signature(model)
            with backend.actual_model(path,record,params,x.shape[1],guard) as loaded:
                self.assertEqual(backend.predict(loaded,query,guard).tobytes(),expected.tobytes())
            wrong=dict(params,flat_parameters=dict(params['flat_parameters'],random_seed=7))
            with self.assertRaises(ValueError):
                with backend.actual_model(path,record,wrong,x.shape[1],guard):pass

    def test_real_inner_selection_and_fresh_positive_refit(self):
        x,y=self.data(192);x[:,2:]=0.;fit=np.arange(len(y))%2==0;valid=~fit
        # Only the temporary CPU fixture uses16 rounds. The production ceiling
        # remains2000 and is checked separately; this is not a full study.
        with patch.object(backend,'MAX_ROUNDS',16):
            model=backend.train(x[fit],y[fit],7,16,lambda:None,validation=(x[valid],y[valid]))
            chosen,receipt=backend.selected_iterations(model,y[valid])
            self.assertGreater(chosen,0);self.assertLessEqual(chosen,16)
            actual=backend.train(x,y,7,chosen,lambda:None)
            self.assertEqual(actual.tree_count_,chosen)
            self.assertFalse(receipt['qualifying_labels_read'])

    def test_zero_correction_keeps_zero_slot_and_invalid_inventory_rejected(self):
        class Model:
            tree_count_=4
            def get_best_iteration(self):return 3
            def get_best_score(self):return {'validation':{'RMSE':1.}}
        chosen,receipt=backend.selected_iterations(Model(),np.zeros(21,dtype='<f8'))
        self.assertEqual(chosen,0);self.assertEqual(receipt['selected_iterations'],0)
        model=Model();model.tree_count_=3
        with self.assertRaises(ValueError):backend.selected_iterations(model,np.ones(21,dtype='<f8'))

    def test_invalid_schema_values_seed_or_budget_rejected_before_training(self):
        x,y=self.data();guard=lambda:None
        for changed in (x.astype('<f8'),np.asfortranarray(x),x[:,:712]):
            with self.assertRaises(ValueError):backend.train(changed,y,7,4,guard)
        for count in (0,2001):
            with self.assertRaises(ValueError):backend.train(x,y,7,count,guard)
        with self.assertRaises(ValueError):backend.train(x,y,-1,4,guard)
        with self.assertRaises(ValueError):backend.train(x,y,7,4,guard,validation=(x,y))
        x[0,0]=np.nan
        with self.assertRaises(ValueError):backend.train(x,y,7,4,guard)

    def test_model_identity_is_checked_even_when_prediction_body_fails(self):
        x,y=self.data();model=backend.train(x,y,13,2,lambda:None)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'model.cbm';model.save_model(str(path));record={'identity':file_identity(path.stat())}
            with self.assertRaisesRegex(ValueError,'changed during'):
                with backend.actual_model(path,record,backend.model_signature(model),713,lambda:None):
                    replacement=path.with_suffix('.new');replacement.write_bytes(path.read_bytes());replacement.replace(path)
                    raise RuntimeError('Synthetic prediction failure')

    def test_model_identity_is_checked_when_native_deserialization_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'model.cbm';path.write_bytes(b'synthetic failure fixture')
            record={'identity':file_identity(path.stat())}
            class FailedLoader:
                def load_model(self,**kwargs):
                    replacement=path.with_suffix('.new');replacement.write_bytes(path.read_bytes());replacement.replace(path)
                    raise RuntimeError('Synthetic native deserialization failure')
            with patch.object(backend,'runtime',return_value=FailedLoader):
                with self.assertRaisesRegex(ValueError,'changed during'):
                    with backend.actual_model(path,record,{},713,lambda:None):pass


if __name__=='__main__':unittest.main()
