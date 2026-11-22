"""Independent NN preprocessing/FD/inventory fixtures; no real MPS admission."""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import verify_upstream_rbm400_neural_study_reference_v4 as verification
import upstream_rbm400_neural_study_reference_v4 as study
import upstream_rbm400_neural_scaler as scaling
import test_upstream_rbm400_neural_study_reference_v4 as study_fixtures


class ReplayTests(unittest.TestCase):
    @contextmanager
    def complete_fixture(self,arms=('latent',)):
        # Source/device/training are mocked in this temporary fixture. Real
        # materialization, all eligible scalers, durable tensors, every fold,
        # independent reconstruction and forward predictions run on CPU.
        import torch
        original_factory=study.architecture.model_factory
        class DeviceFixture:
            def __init__(self,width,cfg):self.model=original_factory(width,cfg)
            def parameters(self):
                from types import SimpleNamespace
                return iter([SimpleNamespace(device=SimpleNamespace(type='mps'))])
            def state_dict(self):return self.model.state_dict()
            def load_state_dict(self,*a,**k):return self.model.load_state_dict(*a,**k)
            @property
            def base(self):return self.model.base
            def to(self,device):
                if device!='mps':raise AssertionError('Expected original declared device')
                return self
        def train(x,y,cfg,seed,epochs,valid=None):
            torch.manual_seed(seed);model=DeviceFixture(x.shape[1],cfg)
            selected=0 if seed>=2000 else (1 if seed<13 else 2)
            return model,selected if valid is not None else epochs
        def predict(model,x):
            model.model.eval()
            with torch.no_grad():return model.model(torch.from_numpy(x)).squeeze(1).numpy().copy()
        @contextmanager
        def query(q,declaration,records,guard,*,retain_scalar,latent):
            if q!='probex' or not retain_scalar:raise AssertionError('Exact synthetic known retained pair')
            raw=fixture_manifest['_fixture_operators']
            class Reader:
                prediction=np.zeros(100,dtype='<f4')
                def block(self,lo,hi):return raw['z'][lo:hi],np.ascontiguousarray(raw['f'][lo:hi,:863 if latent else 713])
            yield Reader()
        helper=study_fixtures.StudyTests()
        with helper.fixture(full_operators=True) as (root,fixture_manifest,_):
            study.prepare(lambda **kw:None)
            with patch.object(study,'runtime',return_value=torch),patch.object(study.inputs.context,'open_query',side_effect=query),\
                 patch.object(study.backend,'train_nn',side_effect=train),patch.object(study.backend,'predict',side_effect=predict),\
                 patch.object(study.architecture,'model_factory',side_effect=DeviceFixture),patch.object(torch.mps,'empty_cache'):
                results={}
                for arm in arms:
                    study.fit(arm,lambda **kw:None);results[arm]=verification.verify(arm,lambda **kw:None)
                yield root,results

    def test_complete_synthetic_fit_scalers_inputs_saved_models_and_full_replay(self):
        with self.complete_fixture() as (root,results):
            result=results['latent']
            self.assertEqual(result['state'],verification.STATE);self.assertEqual(result['actual_saved_models_reopened'],10)
            self.assertTrue(result['every_held_prediction_bit_equal']);self.assertEqual(len(result['records']),5)
            run,report=study.paths('latent')
            nn=np.load(run/'nn.npy',allow_pickle=False);half=np.load(run/'fixed-half-tree154.npy',allow_pickle=False)
            expected=np.clip(.5*nn.astype(np.float64)+.5*np.float32(3.6),1.05,4.95).astype('<f4')
            self.assertEqual(half.tobytes(),expected.tobytes())
            with self.assertRaises(ValueError):verification.verify('latent',lambda **kw:None)

    def test_independent_full_width_eligible_scalers_match_every_original_bit(self):
        rng=np.random.default_rng(128)
        for width in (713,714,863,864):
            raw=rng.normal(size=(321,width)).astype('<f4');raw[:,-2]=3.;mask=np.arange(321)%3!=0
            expected=scaling.fit_scaler(raw,mask,guard=lambda:None)
            actual=verification.independent_scaler(raw,mask,lambda:None)
            for name in expected:self.assertEqual(expected[name].tobytes(),actual[name].tobytes(),(width,name))

    def checkpoint_fixture(self,root):
        import torch
        cfg=study.architecture.CONFIGS['latent'];torch.manual_seed(128)
        model=study.architecture.model_factory(864,cfg)
        saved=dict(state=model.state_dict(),config=cfg,feature_width=864,arm='latent',fold=2,seed_base=7,seed=9,
                   epochs=1,backend='torch-mps-float32',optimizer_state_retained=False,training_resume_supported=False)
        manifest=dict(arm='latent',config=cfg,dimensions={'nn_inputs':864},
            layout=study.architecture.ops.layout(study.architecture.original.signal_indices(713),study.architecture.original.CONTEXT,
                                                retain_scalar=True,latent=True))
        path=root/'nn.pt';torch.save(saved,path)
        record=study.inputs.cpu.inputs.verify_files(root,{'nn.pt':study.inputs.cpu.inputs.sha(path)},lambda:None)['nn.pt']
        return path,saved,manifest,record,model

    def fake_device_factory(self):
        original=study.architecture.model_factory
        class CPUDeviceMoveFixture:
            def __init__(self,width,cfg):self.model=original(width,cfg)
            def state_dict(self):return self.model.state_dict()
            def load_state_dict(self,*args,**kwargs):return self.model.load_state_dict(*args,**kwargs)
            def to(self,device):
                if device!='mps':raise AssertionError('Expected declared original device request')
                return self
        return CPUDeviceMoveFixture

    def test_authenticated_loader_reopens_complete_real_CPU_state_with_mocked_device_move(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path,saved,manifest,record,_=self.checkpoint_fixture(root)
            with patch.object(study.architecture,'model_factory',self.fake_device_factory()):
                with verification.actual_model(path,record,manifest,2,7,1,lambda:None) as reopened:
                    for name,value in reopened.state_dict().items():self.assertTrue(torch.equal(value,saved['state'][name]))

    def test_loader_rejects_metadata_dtype_and_layout_without_silent_coercion(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path,saved,manifest,_,_=self.checkpoint_fixture(root)
            variants=[]
            wrong=copy.deepcopy(saved);wrong['seed']=10;variants.append(wrong)
            wrong=copy.deepcopy(saved);wrong['state']['base.0.weight']=wrong['state']['base.0.weight'].double();variants.append(wrong)
            wrong=copy.deepcopy(saved);wrong['state']['context'][0]=-1;variants.append(wrong)
            wrong=copy.deepcopy(saved);del wrong['state']['base.6.bias'];variants.append(wrong)
            with patch.object(study.architecture,'model_factory',self.fake_device_factory()):
                for value in variants:
                    torch.save(value,path);record=study.inputs.cpu.inputs.verify_files(root,{'nn.pt':study.inputs.cpu.inputs.sha(path)},lambda:None)['nn.pt']
                    with self.assertRaises(ValueError):
                        with verification.actual_model(path,record,manifest,2,7,1,lambda:None):self.fail('Invalid checkpoint accepted')

    def test_loader_checks_original_identity_on_exceptional_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path,_,manifest,record,_=self.checkpoint_fixture(root)
            with patch.object(study.architecture,'model_factory',self.fake_device_factory()):
                with self.assertRaisesRegex(ValueError,'changed during full replay'):
                    with verification.actual_model(path,record,manifest,2,7,1,lambda:None):
                        with path.open('ab') as stream:stream.write(b'changed')
                        raise RuntimeError('Synthetic inference failure')

    def test_all_five_fit_receipts_exact_inventory_and_zero_slot_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'artifacts').mkdir();(root/'experiments').mkdir()
            with patch.object(study,'ROOT',root),patch.object(study.inputs.context.provider.original,'ROOT',root):
                run,report=study.paths('latent');run.mkdir();report.mkdir();(report/'nn').mkdir()
                manifest=dict(config=study.architecture.CONFIGS['latent'])
                (report/'manifest.json').write_text(json.dumps(manifest));sha=study.inputs.cpu.inputs.sha
                msha=sha(report/'manifest.json');proofs={}
                for fold in range(5):
                    directory=run/'nn'/str(fold);directory.mkdir(parents=True)
                    files={stage+'-'+name for stage in ('inner','full') for name in ('weights.npy','lo.npy','hi.npy','mean.npy','std.npy','pipeline.json')}|{'prediction.npy','nn-7.pt','nn-13.pt'}
                    outputs={}
                    for name in files:
                        path=directory/name;path.write_bytes(b'synthetic inventory-only fixture')
                        outputs[name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
                    receipt=dict(fold=fold,chosen_iterations=[0,1,2],manifest_sha256=msha,seed_order=[2026,7,13],seed_offset=fold,
                        actual_MPS_fit=True,qualifying_labels_read=False,actual_saved_model_count=2,outputs=outputs,
                        schedule='all_original_independent_inner_selections_then_fresh_outer_refits')
                    path=report/'nn'/('%s.json'%fold);path.write_text(json.dumps(receipt));proofs[str(path.relative_to(root))]=sha(path)
                complete=dict(state=study.FIT_STATE,arm='latent',manifest_sha256=msha,all_five_folds_complete=True,
                    all_three_seed_slots_each_fold=True,actual_MPS_fit=True,qualifying_labels_read=False,fold_proof_sha256=proofs)
                (report/'fit-completion.json').write_text(json.dumps(complete))
                receipts,bindings=verification.receipt_bindings('latent',manifest)
                self.assertEqual(len(receipts),5);self.assertGreater(len(bindings),70)
                (run/'nn'/'0'/'nn-2026.pt').write_bytes(b'Cannot fabricate a zero-selected model')
                with self.assertRaises(ValueError):verification.receipt_bindings('latent',manifest)


if __name__=='__main__':unittest.main()
