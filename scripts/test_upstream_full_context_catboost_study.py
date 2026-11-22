"""Durable study glue on synthetic source proofs and mocked fold scheduling."""
from contextlib import contextmanager,ExitStack
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import upstream_full_context_catboost_study as module
import test_upstream_full_context_catboost_consumer as fixtures


class StudyTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with fixtures.ConsumerTests().fixture() as f,ExitStack() as stack:
            root=f['root'];stack.enter_context(patch.object(module,'ROOT',root));stack.enter_context(patch.object(module,'ROWS',20));stack.enter_context(patch.object(module,'QUALIFYING_ROWS',23))
            source=root/'fixture-study-code.py';source.write_text('# synthetic study source\n')
            bindings={'fixture-study-code.py':module.sha(source)}
            stack.enter_context(patch.object(module,'code_bindings',return_value=bindings))
            weights=np.zeros((5,2,562),dtype='<f8');metadata=f['metadata']
            metadata['weights_sha256']=hashlib.sha256(weights.tobytes()).hexdigest()
            metadata_sha=hashlib.sha256(module.wire.encoded(metadata)).hexdigest()
            contract=module.consumer.contract(metadata,metadata_sha,root)
            prepared=dict(metadata=metadata,contract=contract,weights=weights)
            owner={}
            for name in ['plan','monitor','baseline','gate','ownership']:
                path=root/'control'/(name+'.json');path.parent.mkdir(exist_ok=True)
                path.write_bytes(module.BASELINE.read_bytes() if name=='baseline' else b'{"synthetic_ownership":true}\n')
                owner[name]=dict(path=str(path),sha256=module.sha(path))
            yield f,prepared,owner

    def test_durable_preparation_load_original_budgets_and_no_retry(self):
        with self.fixture() as (f,prepared,owner):
            receipt=module.prepare_arm('scalar',prepared,owner,lambda **kw:None)
            loaded,metadata,contract=module.load('scalar',receipt['sha256'],lambda **kw:None)
            self.assertEqual(loaded['max_rounds'],2000);self.assertEqual(loaded['patience'],100)
            self.assertEqual(loaded['seed_order'],[2026,7,13]);self.assertEqual(loaded['folds'],5)
            self.assertEqual(metadata,prepared['metadata']);self.assertEqual(contract,prepared['contract'])
            self.assertFalse(loaded['production_study_admitted'])
            with self.assertRaises(ValueError):module.prepare_arm('scalar',prepared,owner,lambda **kw:None)
            changed=f['root']/'fixture-study-code.py';changed.write_text('changed source')
            with patch.object(module,'code_bindings',return_value={'fixture-study-code.py':module.sha(changed)}):
                with self.assertRaises(ValueError):module.load('scalar',receipt['sha256'],lambda **kw:None)
            with self.assertRaises(ValueError):module.load('scalar','0'*64,lambda **kw:None)

    def test_changed_weights_or_original_baseline_cannot_prepare(self):
        with self.fixture() as (f,prepared,owner):
            prepared['weights'][0,0,0]=1
            with self.assertRaises(ValueError):module.prepare_arm('scalar',prepared,owner,lambda **kw:None)
        with self.fixture() as (f,prepared,owner):
            path=Path(owner['baseline']['path']);path.write_text('{}');owner['baseline']['sha256']=module.sha(path)
            with self.assertRaisesRegex(ValueError,'byte-identical'):module.prepare_arm('scalar',prepared,owner,lambda **kw:None)

    def test_partial_started_marker_and_changed_weight_payload_cannot_be_reused(self):
        with self.fixture() as (f,prepared,owner):
            receipt=module.prepare_arm('scalar',prepared,owner,lambda **kw:None)
            run,report=module.paths('scalar');(report/'fit-started.json').write_text('{}')
            with patch.object(module.wire,'receive',side_effect=AssertionError('partial fit restarted')):
                with self.assertRaises(ValueError):module.fit('scalar',receipt['sha256'],io.BytesIO(),producer_receipt=f['receipt'],producer_receipt_sha256=f['rh'],producer_identity=f['captured'],guard=lambda **kw:None)
            weights=np.ones_like(prepared['weights']);np.save(run/'weights.npy',weights)
            with self.assertRaises(ValueError):module.load('scalar',receipt['sha256'],lambda **kw:None)

    def test_full_five_fold_actual_saved_model_inventory_and_predictions(self):
        with self.fixture() as (f,prepared,owner):
            receipt=module.prepare_arm('scalar',prepared,owner,lambda **kw:None)
            wire_proof=dict(exact_footer_and_EOF=True)
            def fit(features,baselines,metadata,metadata_sha,proof,**callbacks):
                # Source/row admission and iteration selection are synthetic
                # here; every model saved by these callbacks is a real pinned
                # CPU CatBoost predictor with the declared full raw width.
                result=[]
                for fold in range(5):
                    for stage in ['inner','full']:callbacks['save_population'](fold,stage,{'fixture':True})
                    counts=[2,0,3]
                    for seed,count in zip(module.SEEDS,counts):
                        if count:
                            model=module.backend.train(features,np.linspace(-.1,.1,20,dtype=np.float64),seed+fold,count,lambda **kw:None)
                            callbacks['save_model'](fold,model,seed,count)
                    prediction=np.full(4,2.5,dtype='<f4')
                    row=dict(fold=fold,seed_order=list(module.SEEDS),chosen_iterations=counts)
                    callbacks['save_fold'](fold,row,prediction);result.append(row)
                return dict(prediction=np.full(20,2.5,dtype='<f4'),folds=result)
            events=[]
            def receive(*args,**kwargs):
                events.append('complete-stream');return f['x'],f['b'],wire_proof
            def exit_gate(proof):
                self.assertEqual(events,['complete-stream']);self.assertIs(proof,wire_proof)
                events.append('producer-exit-verified')
                return dict(path=f['receipt'],sha256=f['rh'],producer=f['captured'])
            with patch.object(module.wire,'receive',side_effect=receive),patch.object(module.consumer,'fit',side_effect=fit):
                result=module.fit('scalar',receipt['sha256'],io.BytesIO(),producer_exit_gate=exit_gate,guard=lambda **kw:None)
            self.assertEqual(events,['complete-stream','producer-exit-verified'])
            self.assertEqual(result['actual_saved_model_count'],10);self.assertEqual(len(result['fold_proof_sha256']),5)
            self.assertEqual(result['original_seed_slots'],15);self.assertTrue(result['zero_slots_retained_in_denominator'])
            self.assertFalse(result['production_study_admitted'])
            run,report=module.paths('scalar')
            for fold in range(5):
                row=json.loads((report/'folds'/('%s.json'%fold)).read_text())
                self.assertEqual(set(row['models']),{'2026','13'})
                self.assertFalse((run/'folds'/str(fold)/'tree-7.cbm').exists())
                for record in row['models'].values():
                    self.assertEqual(module.sha(record['path']),record['sha256'])
            with self.assertRaises(ValueError):module.fit('scalar',receipt['sha256'],io.BytesIO(),producer_receipt=f['receipt'],producer_receipt_sha256=f['rh'],producer_identity=f['captured'],guard=lambda **kw:None)


if __name__=='__main__':unittest.main()
