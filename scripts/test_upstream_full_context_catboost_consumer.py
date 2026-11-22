"""Consumer barriers on temporary files; CPU proof/row counts are synthetic."""
from contextlib import ExitStack,contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_full_context_catboost_consumer as module


class ConsumerTests(unittest.TestCase):
    def sha(self, path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return self.sha(path)

    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary).resolve();n=20
            stack.enter_context(patch.object(module,'ROWS',n));stack.enter_context(patch.object(module,'QUALIFYING_ROWS',23))
            compact={};bindings={};prepared={}
            for name,value in [('labels.npy',(np.arange(n)%5+1).astype('<f8')),
                               ('users.npy',np.arange(n,dtype='<i4')),('groups.npy',(np.arange(n)%10).astype('i1'))]:
                path=root/module.ARRAYS/name;path.parent.mkdir(parents=True,exist_ok=True);np.save(path,value)
                relative=str(path.relative_to(root));bindings[relative]=self.sha(path)
                record=module.verify_files(root,{relative:bindings[relative]},lambda:None)[relative]
                compact[name]=dict(path=str(path),dtype=value.dtype.str,shape=[n],**record)
                prepared[name]=dict(path=str(path),bytes=record['bytes'],sha256=record['sha256'])
            x=np.zeros((n,713),dtype='<f4');x[:,0]=np.arange(n);b=np.full((5,2,n),2.5,dtype='<f8')
            op=['0'*64,hashlib.sha256(x[:,:-1].tobytes()).hexdigest()];dims=module.operators.dimensions(False,False)
            manifest=dict(state='complete_both_context_arms_data_and_known_ridge_pending_independent_full_replay',rows=n,qualifying_rows=23,
                qualifying_labels_read=False,neural_models_fitted=0,nonlinear_component_OOF_predictions_used_as_features=False,
                lambda_fixed=1000.,folds=5,groups=10,alias={'retain_scalar':False},arms={'scalar':dims},
                virtual_c_order_sha256={'probex':{'arm_operator_sha256':{'scalar':op}}},source_sha256=dict(bindings),prepared=prepared)
            mp=root/module.DATA/'manifest.json';mh=self.write(mp,manifest);bindings[str(mp.relative_to(root))]=mh
            proof=dict(state='complete_independent_both_context_arms_all_operators_aliases_statistics_and_known_ridge',manifest_sha256=mh,
                source_sha256=manifest['source_sha256'],qualifying_labels_read=False,neural_models_fitted=0,
                both_query_alias_independently_verified=True,every_both_arm_operator_bit_verified=True,
                every_group_statistic_and_ridge_prediction_bit_verified=True,rows=n,qualifying_rows=23,ridge_equations=2810)
            pp=root/module.REPLAY/'independent-verification.json';ph=self.write(pp,proof);bindings[str(pp.relative_to(root))]=ph
            cp=root/module.CONTROL/'completion.json';bindings[str(cp.relative_to(root))]=self.write(cp,dict(
                state='complete_both_RBM400_context_arms_data_and_independent_known_ridge_replay',verification_sha256=ph,qualifying_labels_read=False))
            metadata=dict(format='own_full_context_tree_original_source_v1',arm='scalar',rows=n,qualifying_rows=23,retain_scalar=False,
                dimensions=dims,source_sha256=bindings,known_arrays=compact,parent_CPU_manifest_sha256=mh,parent_CPU_verification_sha256=ph,
                weights_sha256='1'*64,operator_sha256=op,ridge_penalty=1000.,ridge_equations=10,
                nonlinear_component_OOF_predictions_used_as_features=False,qualifying_labels_read=False,models_fitted=0)
            mh=hashlib.sha256(module.wire.encoded(metadata)).hexdigest();contract=module.contract(metadata,mh,root)
            wire_proof=dict(contract_sha256=hashlib.sha256(module.wire.encoded(contract)).hexdigest(),admission_sha256=mh,operator_sha256=op,
                exact_footer_and_EOF=True,models_fitted=0,qualifying_labels_read=False,raw_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                baseline_sha256=[hashlib.sha256(b[f,p].tobytes()).hexdigest() for f in range(5) for p in range(2)])
            # A real short-lived child is observed before exit and reaped by
            # this test parent. It does not train or admit any production data.
            process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.15)'])
            captured=module.identity.process(process.pid);assert captured is not None
            self.assertEqual(process.wait(timeout=10),0)
            receipt=dict(state='complete_owned_original_tree_producer_zero_exit',exit_code=0,producer=captured,
                contract_sha256=wire_proof['contract_sha256'],source_admission_sha256=mh,qualifying_labels_read=False)
            rp=root/'producer-exit.json';rh=self.write(rp,receipt)
            yield dict(root=root,x=x,b=b,metadata=metadata,mh=mh,contract=contract,wire=wire_proof,receipt=rp,rh=rh,captured=captured)

    def fit(self,f,**kwargs):
        calls=[]
        def train(x,b,y,users,groups,fold,**callbacks):
            calls.append(fold);self.assertEqual(x.shape,(20,713));self.assertEqual(b.shape,(2,20))
            self.assertEqual(y.dtype,np.dtype('<f8'))
            return dict(fold=fold,chosen_iterations=[0,0,0],prediction=b[1,groups//2==fold].astype('<f4'))
        args=dict(producer_receipt=f['receipt'],producer_receipt_sha256=f['rh'],producer_identity=f['captured'],root=f['root'],guard=lambda:None,
            save_model=lambda *a:self.fail('zero slots saved a model'),save_population=lambda *a:None,save_fold=lambda *a:None,
            release=lambda:None,train_fold=train);args.update(kwargs)
        result=module.fit(f['x'],f['b'],f['metadata'],f['mh'],f['wire'],**args)
        return result,calls

    def test_complete_authenticated_known_files_and_all_five_fold_dispatches(self):
        with self.fixture() as f:
            result,calls=self.fit(f)
            self.assertEqual(calls,list(range(5)));self.assertEqual(len(result['prediction']),20)
            self.assertTrue((result['prediction']==np.float32(2.5)).all())
            self.assertFalse(result['production_study_admitted'])
            self.assertNotIn('torch',sys.modules)

    def test_changed_source_raw_handoff_or_exit_receipt_prevents_fit(self):
        with self.fixture() as f:
            f['x'][0,0]=.01
            with self.assertRaisesRegex(ValueError,'Transferred raw'):self.fit(f)
        with self.fixture() as f:
            metadata=f['metadata'];p=Path(metadata['known_arrays']['labels.npy']['path']);p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):self.fit(f)
        with self.fixture() as f:
            value=json.loads(f['receipt'].read_text());value['exit_code']=1;f['rh']=self.write(f['receipt'],value)
            with self.assertRaisesRegex(ValueError,'successful original producer exit'):self.fit(f)

    def test_live_original_producer_cannot_be_authorized_by_EOF_or_forged_exit(self):
        with self.fixture() as f:
            captured=module.identity.process(os.getpid());value=json.loads(f['receipt'].read_text());value['producer']=captured
            digest=self.write(f['receipt'],value)
            with self.assertRaisesRegex(ValueError,'still live'):
                module.producer_exit(f['receipt'],digest,captured,f['contract'],f['mh'],f['root'],lambda:None)

    def test_exec_command_change_cannot_make_live_producer_look_reaped(self):
        with self.fixture() as f:
            captured=module.identity.process(os.getpid());captured['worker_command']=captured['command']
            captured['command']='captured bootstrap before exec'
            value=json.loads(f['receipt'].read_text());value['producer']=captured
            digest=self.write(f['receipt'],value)
            with self.assertRaisesRegex(ValueError,'still live'):
                module.producer_exit(f['receipt'],digest,captured,f['contract'],f['mh'],f['root'],lambda:None)

    def test_dimensions_paths_and_parent_proof_semantics_are_required(self):
        with self.fixture() as f:
            metadata=copy.deepcopy(f['metadata']);metadata['dimensions']['context_columns']-=1
            digest=hashlib.sha256(module.wire.encoded(metadata)).hexdigest()
            with self.assertRaises(ValueError):module.contract(metadata,digest,f['root'])
            metadata=copy.deepcopy(f['metadata']);metadata['known_arrays']['labels.npy']['path']=str(f['root']/'wrong.npy')
            digest=hashlib.sha256(module.wire.encoded(metadata)).hexdigest()
            with self.assertRaises(ValueError):module.contract(metadata,digest,f['root'])
            path=f['root']/module.REPLAY/'independent-verification.json';proof=json.loads(path.read_text())
            proof['every_both_arm_operator_bit_verified']=False;new_sha=self.write(path,proof)
            metadata=copy.deepcopy(f['metadata']);metadata['source_sha256'][str(path.relative_to(f['root']))]=new_sha
            metadata['parent_CPU_verification_sha256']=new_sha
            cp=f['root']/module.CONTROL/'completion.json';done=json.loads(cp.read_text());done['verification_sha256']=new_sha
            metadata['source_sha256'][str(cp.relative_to(f['root']))]=self.write(cp,done)
            digest=hashlib.sha256(module.wire.encoded(metadata)).hexdigest()
            with self.assertRaisesRegex(ValueError,'Actual complete original CPU'):
                module.authenticate_sources(metadata,digest,f['root'],lambda:None)


if __name__=='__main__':unittest.main()
