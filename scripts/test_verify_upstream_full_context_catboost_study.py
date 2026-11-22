"""Actual complete synthetic tree study/reload; full data proofs are mocked."""
from contextlib import ExitStack
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import upstream_full_context_catboost_study as study
import verify_upstream_full_context_catboost_study as module
import test_upstream_full_context_catboost_consumer as fixtures
import upstream_full_context_catboost_canonical as canonical


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=fixtures.ConsumerTests().fixture();cls.f=cls.fixture.__enter__();f=cls.f;root=f['root'];cls.root=root
        cls.stack=ExitStack();cls.stack.enter_context(patch.object(study,'ROOT',root));cls.stack.enter_context(patch.object(study,'ROWS',20));cls.stack.enter_context(patch.object(study,'QUALIFYING_ROWS',23))
        code=root/'fixture-code.py';code.write_text('# synthetic admitted code\n');binding={'fixture-code.py':study.sha(code)}
        cls.stack.enter_context(patch.object(study,'code_bindings',return_value=binding));cls.stack.enter_context(patch.object(module,'code_bindings',return_value=binding));cls.stack.enter_context(patch.object(canonical,'code_bindings',return_value=binding))
        sha=study.sha
        labels=np.load(root/study.consumer.ARRAYS/'labels.npy');users=np.arange(20,dtype=np.int32);groups=(users%10).astype('i1')
        z=np.zeros((20,562),dtype='<f8');z[:,0]=np.sin(users);z[:,-1]=1
        gram=np.stack([z[groups==g].T@z[groups==g] for g in range(10)])
        rhs=np.stack([z[groups==g].T@labels[groups==g] for g in range(10)])
        source=dict(f['metadata']['source_sha256']);mp=root/study.consumer.DATA/'manifest.json'
        manifest=json.loads(mp.read_text());original=dict(manifest['source_sha256'])
        for name,value in [('gram.npy',gram),('rhs.npy',rhs)]:
            path=root/study.consumer.ARRAYS/name;np.save(path,value);relative=str(path.relative_to(root));original[relative]=sha(path);source[relative]=sha(path)
            manifest['prepared'][name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        x=f['x'];x[:,0]=labels;x[:,1]=np.cos(users)
        weights=np.empty((5,2,562),dtype='<f8')
        reg=np.eye(562)*1000.;reg[-1,-1]=0
        for fold in range(5):
            for population in range(2):
                keep=np.asarray([g//2!=fold and (population==1 or g%2==0) for g in range(10)])
                weights[fold,population]=np.linalg.solve(gram[keep].sum(0)+reg,rhs[keep].sum(0))
        baselines=np.stack([[np.clip(z@weights[f,p],1.05,4.95) for p in range(2)] for f in range(5)])
        op=[hashlib.sha256(z.tobytes()).hexdigest(),hashlib.sha256(x[:,:-1].tobytes()).hexdigest()]
        manifest['source_sha256']=original;manifest['virtual_c_order_sha256']['probex']['arm_operator_sha256']['scalar']=op
        mp.write_text(json.dumps(manifest));source[str(mp.relative_to(root))]=sha(mp)
        pp=root/study.consumer.REPLAY/'independent-verification.json';proof=json.loads(pp.read_text());proof.update(source_sha256=original,manifest_sha256=sha(mp));pp.write_text(json.dumps(proof));source[str(pp.relative_to(root))]=sha(pp)
        cp=root/study.consumer.CONTROL/'completion.json';done=json.loads(cp.read_text());done['verification_sha256']=sha(pp);cp.write_text(json.dumps(done));source[str(cp.relative_to(root))]=sha(cp)
        metadata=copy.deepcopy(f['metadata']);metadata.update(source_sha256=source,parent_CPU_manifest_sha256=sha(mp),parent_CPU_verification_sha256=sha(pp),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),operator_sha256=op)
        metadata_sha=hashlib.sha256(study.wire.encoded(metadata)).hexdigest();contract=study.consumer.contract(metadata,metadata_sha,root)
        cls.metadata=metadata;cls.contract=contract;cls.weights=weights;cls.gram=gram;cls.rhs=rhs;cls.x=x.copy();cls.baselines=baselines
        owner={}
        for name in ['plan','monitor','baseline','gate','ownership']:
            p=root/'owner'/(name+'.json');p.parent.mkdir(exist_ok=True);p.write_bytes(study.BASELINE.read_bytes() if name=='baseline' else b'{"synthetic_owner":true}\n');owner[name]=dict(path=str(p),sha256=sha(p))
        receipt=study.prepare_arm('scalar',dict(metadata=metadata,contract=contract,weights=weights),owner,lambda **kw:None)
        cls.manifest_sha=receipt['sha256'];cls.stream_proof=dict(contract_sha256=hashlib.sha256(study.wire.encoded(contract)).hexdigest(),admission_sha256=metadata_sha,
            operator_sha256=op,exact_footer_and_EOF=True,models_fitted=0,qualifying_labels_read=False,raw_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            baseline_sha256=[hashlib.sha256(baselines[f,p].tobytes()).hexdigest() for f in range(5) for p in range(2)])
        rp=f['receipt'];row=json.loads(rp.read_text());row.update(contract_sha256=cls.stream_proof['contract_sha256'],source_admission_sha256=metadata_sha);rp.write_text(json.dumps(row));cls.receipt_sha=sha(rp)
        with patch.object(study.wire,'receive',return_value=(x.copy(),baselines.copy(),cls.stream_proof)):
            study.fit('scalar',cls.manifest_sha,io.BytesIO(),producer_receipt=rp,producer_receipt_sha256=cls.receipt_sha,producer_identity=f['captured'],guard=lambda **kw:None)
        cls.fit_sha=sha(study.paths('scalar')[1]/'fit-completion.json')

    @classmethod
    def tearDownClass(cls):
        cls.stack.close();cls.fixture.__exit__(None,None,None)

    def verify(self):
        with patch.object(module.wire,'receive',return_value=(self.x.copy(),self.baselines.copy(),self.stream_proof)):
            return module.verify('scalar',self.manifest_sha,self.fit_sha,io.BytesIO(),guard=lambda **kw:None,
                producer_exit_gate=lambda proof:dict(path=self.f['receipt'],sha256=self.receipt_sha,producer=self.f['captured']))

    def test_all_actual_folds_models_ridge_systems_and_OOF_bits_replayed(self):
        result=self.verify()
        self.assertEqual(result['original_seed_slots'],15);self.assertEqual(result['ridge_equations'],10)
        self.assertEqual(len(result['fold_replays']),5);self.assertGreater(result['actual_saved_models_reopened'],0)
        self.assertTrue(result['every_full_OOF_join_bit_verified']);self.assertFalse(result['production_study_admitted'])
        run,report=study.paths('scalar')
        value=canonical.verify('scalar',self.manifest_sha,self.fit_sha,study.sha(report/'independent-verification.json'),lambda **kw:None)
        actual=np.load(run/'prediction.npy');labels=np.load(self.root/study.consumer.ARRAYS/'labels.npy')
        self.assertEqual(value['tree_only_CV_rmse'],canonical.score(actual,labels))
        self.assertTrue(value['score_is_complete_five_user_fold_CV'])
        self.assertFalse(value['production_study_admitted'])
        with self.assertRaises(ValueError):canonical.verify('scalar',self.manifest_sha,self.fit_sha,study.sha(report/'independent-verification.json'),lambda **kw:None)

        with self.assertRaises(ValueError):self.verify()

    def test_changed_one_coefficient_bit_rejected_independently_of_fit_helper(self):
        result=module.coefficients(self.gram,self.rhs,self.weights,lambda:None);self.assertEqual(len(result),10)
        changed=self.weights.copy();changed[3,0,0]=np.nextafter(changed[3,0,0],np.inf)
        with self.assertRaisesRegex(ValueError,'coefficient bit'):module.coefficients(self.gram,self.rhs,changed,lambda:None)

    def test_missing_original_seed_inventory_rejected(self):
        # Binding checks can run after a completed proof; no numerical replay
        # or new source/model admission is performed by this mutation fixture.
        root=self.root;run,report=study.paths('scalar');p=report/'folds/0.json';before=p.read_bytes()
        row=json.loads(before);row['models']={};p.write_text(json.dumps(row))
        fit_path=report/'fit-completion.json';fit_before=fit_path.read_bytes();fit=json.loads(fit_before)
        fit['fold_proof_sha256'][str(p.relative_to(root))]=study.sha(p);fit_path.write_text(json.dumps(fit))
        try:
            with self.assertRaises(ValueError):module.bindings('scalar',self.manifest_sha,study.sha(fit_path),lambda:None)
        finally:p.write_bytes(before);fit_path.write_bytes(fit_before)


if __name__=='__main__':unittest.main()
