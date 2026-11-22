"""Durable neural orchestration fixtures; source/MPS gates explicitly mocked."""
from contextlib import contextmanager,ExitStack
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_rbm400_neural_study_reference_v2 as study


class StudyTests(unittest.TestCase):
    def test_execution_sources_pin_owner_prepare_and_baseline_without_result_cycle(self):
        import upstream_rbm400_neural_execution_reference_v2 as execution
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();report=root/'control';report.mkdir()
            names=('plan.json','monitor.json','first-scientific-baseline.json','prerequisites.json','prepare-gate.json','prepare-ownership.json')
            for name in names:(report/name).write_text('{}')
            (report/'prepare.json').write_text('{"later_result_must_not_be_an_input":true}')
            protocol=root/'experiments/upstream-rbm400-neural-execution-protocol-reference-v2/verification.json';protocol.parent.mkdir(parents=True);protocol.write_text('{}')
            with patch.object(study,'ROOT',root),patch.object(execution,'REPORT',report),patch.object(execution,'check_sources'):
                bindings=study.execution_bindings()
            self.assertEqual(set(bindings),{str((report/name).relative_to(root)) for name in names}|{str(protocol.relative_to(root))})
            self.assertNotIn('control/prepare.json',bindings)

    @contextmanager
    def fixture(self,full_operators=False):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder).resolve();base_art=root/'base-art';base_report=root/'base-report'
            base_art.mkdir();base_report.mkdir();(root/'artifacts').mkdir();(root/'experiments').mkdir()
            (root/'fixture-code.py').write_text('# synthetic source fixture\n')
            (root/'tree.f32').write_bytes(np.full(100,3.6,dtype='<f4').tobytes())
            tree_manifest=root/'tree-manifest.json'
            tree=dict(id='tree154',prediction='tree.f32',family='tree',origin='synthetic-fixture')
            tree_manifest.write_text(json.dumps({'controls':[tree]}))
            for name in ('manifest.json','independent-verification.json'):(base_report/name).write_text('{}')
            arrays={'labels.npy':np.full(100,5.,dtype='<f8'),'users.npy':np.arange(100,dtype='<i4'),
                'groups.npy':(np.arange(100)%10).astype('i1'),'gram.npy':np.tile(np.eye(563),(10,1,1)),
                'rhs.npy':np.zeros((10,563))}
            if full_operators:
                rng=np.random.default_rng(128);z=rng.normal(size=(100,563));z[:,-1]=1
                features=rng.normal(size=(100,863)).astype('<f4')
                for group in range(10):
                    keep=arrays['groups.npy']==group;a=z[keep]
                    arrays['gram.npy'][group]=a.T@a;arrays['rhs.npy'][group]=a.T@arrays['labels.npy'][keep]
            for name,value in arrays.items():np.save(base_art/name,value)
            sha=study.inputs.cpu.inputs.sha
            required=list(base_art.iterdir())+list(base_report.iterdir())+[root/'tree.f32',tree_manifest,root/'fixture-code.py']
            bindings={str(p.relative_to(root)):sha(p) for p in required}
            manifest=dict(alias={'retain_scalar':True},arms={arm:study.architecture.ops.dimensions(True,arm=='latent') for arm in study.ARMS},
                prepared={name:dict(path=str(base_art/name),sha256=sha(base_art/name)) for name in arrays})
            if full_operators:
                import hashlib
                manifest['_fixture_operators']={'z':z,'f':features}
                manifest['virtual_c_order_sha256']={'probex':{'arm_operator_sha256':{
                    arm:[hashlib.sha256(z.tobytes()).hexdigest(),hashlib.sha256(np.ascontiguousarray(features[:,:713 if arm=='scalar' else 863]).tobytes()).hexdigest()]
                    for arm in study.ARMS}}}
            def sources(guard):
                declaration={'source_sha256':dict(bindings)}
                return declaration,copy.deepcopy(manifest),study.inputs.cpu.inputs.verify_files(root,bindings,guard)
            for module,name,value in [(study,'ROOT',root),(study,'TREE_MANIFEST',tree_manifest),(study,'ROWS',100),
                (study.inputs.data,'ART',base_art),(study.inputs.data,'REPORT',base_report),
                (study.inputs.context.provider.original,'ROOT',root)]:stack.enter_context(patch.object(module,name,value))
            stack.enter_context(patch.object(study.inputs,'sources',side_effect=sources))
            stack.enter_context(patch.object(study,'code_bindings',return_value={'fixture-code.py':sha(root/'fixture-code.py')}))
            stack.enter_context(patch.object(study,'execution_bindings',return_value={}))
            yield root,manifest,tree

    def test_prepare_both_arm_manifests_and_load_exact_contracts(self):
        with self.fixture() as (root,manifest,tree):
            receipts=study.prepare(lambda **kw:None)
            self.assertEqual(set(receipts),set(study.ARMS))
            for arm in study.ARMS:
                value,_,_,_=study.load(arm,lambda **kw:None)
                self.assertEqual(value['dimensions'],manifest['arms'][arm]);self.assertEqual(value['controls'],[tree])
                self.assertEqual(value['config']['seeds'],[2026,7,13]);self.assertEqual(value['config']['nn_epochs'],40)
            with self.assertRaises(ValueError):study.prepare(lambda **kw:None)
            _,report=study.paths('latent');path=report/'manifest.json';value=json.loads(path.read_text())
            value['layout']['context'][0]=-1;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):study.load('latent',lambda **kw:None)

    def test_runtime_rejects_fallback_before_any_actual_device_request(self):
        with patch.dict(study.os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'1'}):
            with self.assertRaises(ValueError):study.runtime()

    def test_checkpoint_rejects_CPU_or_zero_or_misdeclared_model_slot(self):
        import torch
        model=study.architecture.model_factory(864,study.architecture.CONFIGS['latent'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'nn.pt'
            for count,width in ((0,864),(1,865),(1,864)):
                with self.assertRaises(ValueError):study.checkpoint(path,model,'latent',0,2026,count,width,lambda **kw:None)
            self.assertFalse(path.exists())

    def test_actual_width_and_buffer_validation_is_independent_of_device_gate(self):
        class DeviceFixture:
            def __init__(self,width):self.model=study.architecture.model_factory(width,study.architecture.CONFIGS['latent'])
            def parameters(self):return iter([SimpleNamespace(device=SimpleNamespace(type='mps'))])
            def state_dict(self):return self.model.state_dict()
            @property
            def base(self):return self.model.base
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'nn.pt'
            wrong=DeviceFixture(863)
            with self.assertRaisesRegex(ValueError,'Actual model input width'):
                study.checkpoint(path,wrong,'latent',0,2026,1,864,lambda **kw:None)
            for buffer in ('signals','context'):
                value=DeviceFixture(864);getattr(value.model,buffer)[0]=-1
                with self.assertRaisesRegex(ValueError,'Actual signal/context buffers'):
                    study.checkpoint(path,value,'latent',0,2026,1,864,lambda **kw:None)
            self.assertFalse(path.exists())

    def test_all_five_folds_durable_inventory_counts_checkpoint_bits_and_no_restart(self):
        # This is an IO orchestration fixture: the device gate and fold trainer
        # are mocked. The genuine stored CPU H128 state tests serialization,
        # not actual MPS fitting or scientific source acceptance.
        import torch
        class SyntheticDeviceModel:
            def __init__(self):
                torch.manual_seed(128);self.model=study.architecture.model_factory(864,study.architecture.CONFIGS['latent'])
            def parameters(self):return iter([SimpleNamespace(device=SimpleNamespace(type='mps'))])
            def state_dict(self):return self.model.state_dict()
            @property
            def base(self):return self.model.base
        with self.fixture() as (root,_,_):
            study.prepare(lambda **kw:None)
            def fixture_fold(arm,fold,labels,users,groups,inner_weights,full_weights,**callbacks):
                for stage,weights in [('inner',inner_weights),('full',full_weights)]:
                    scaler={name:np.full(864,1. if name=='std' else 0.,dtype=np.float64) for name in ('lo','hi','mean','std')}
                    callbacks['save_preprocessing'](stage,weights,scaler,dict(synthetic_fixture=True))
                callbacks['save_model'](SyntheticDeviceModel(),7,1);callbacks['save_model'](SyntheticDeviceModel(),13,2)
                return dict(fold=fold,chosen_iterations=[0,1,2],prediction=np.full(int(np.count_nonzero(groups//2==fold)),3.6,dtype='<f4'),
                    preprocessing={'synthetic_fixture':True},seed_order=[2026,7,13],seed_offset=fold,
                    schedule='all_original_independent_inner_selections_then_fresh_outer_refits',full_representation_matrices_resident_at_once=1)
            with patch.object(study,'runtime',return_value=torch),patch.object(study.folds,'run',side_effect=fixture_fold),patch.object(torch.mps,'empty_cache'):
                result=study.fit('latent',lambda **kw:None)
                self.assertEqual(set(result['fold_proof_sha256']),{str(study.paths('latent')[1].relative_to(root))+'/nn/'+str(fold)+'.json' for fold in range(5)})
                run,report=study.paths('latent')
                for fold in range(5):
                    receipt=json.loads((report/'nn'/(str(fold)+'.json')).read_text())
                    self.assertEqual(receipt['chosen_iterations'],[0,1,2]);self.assertEqual(receipt['actual_saved_model_count'],2)
                    self.assertFalse((run/'nn'/str(fold)/'nn-2026.pt').exists())
                    path=run/'nn'/str(fold)/'nn-7.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
                    self.assertEqual(saved['seed'],7+fold);self.assertEqual(saved['epochs'],1)
                    self.assertFalse(saved['training_resume_supported']);self.assertEqual(saved['feature_width'],864)
                    for name,tensor in SyntheticDeviceModel().state_dict().items():self.assertTrue(torch.equal(saved['state'][name],tensor))
                with self.assertRaises(ValueError):study.fit('latent',lambda **kw:None)


if __name__=='__main__':unittest.main()
