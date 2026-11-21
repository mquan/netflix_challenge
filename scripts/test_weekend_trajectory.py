"""New full586/813-input stage invariants; no production fits or recovered outcomes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import upstream_weekend_trajectory_ops as ops
import upstream_weekend_trajectory_data as data
import verify_upstream_weekend_trajectory_data as independent
import upstream_weekend_trajectory_mps as neural
import upstream_complete_trajectory_mps as previous
import upstream_exact_weekend_scaler as scaler
import verify_upstream_weekend_trajectory_mps as replay
import upstream_weekend_trajectory_fold_bag as bag
import upstream_own16_weekend_trajectory_calibration as cal
import verify_upstream_own16_weekend_trajectory_calibration as audit


class WeekendTrajectoryTests(unittest.TestCase):
    def test_all195_candidates_original_order_and_hash_bound_native_projection(self):
        import verify_upstream_complete_trajectory_data as old
        original=[dict(id='old:%03d'%i,paired_files={}) for i in range(128)]
        with mock.patch.object(old,'independent_candidates',return_value=original):rows=independent.independent_candidates()
        self.assertEqual(rows[:128],original);self.assertEqual(len(rows),195)
        self.assertEqual([r['id'] for r in rows[128:144]],['weekend:'+c for c in data.u.read(data.u.ROOT/'experiments/upstream-remote-weekend/plan.json')['jobs'][0]['published_columns']])
        self.assertTrue(all(r['known_row_map']=='artifacts/upstream-split3-pilot/probex-row-map.npy' for r in rows[128:]))

    def test_projected_original_query_bits_and_independent_take(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=np.ones(1760494,dtype='<f4');raw[:8]=[-1,-0.,0.,1.1,4.9,6.,7.,3.25]
            mapping=np.zeros(1128164,dtype='<i8');mapping[:8]=[7,6,5,4,3,2,1,0]
            np.save(root/'native.npy',raw);np.save(root/'map.npy',mapping)
            row=dict(paired_files={'probex':'native.npy'},known_row_map='map.npy')
            actual=ops.scalar(root,row,'probex',0,8)
            with mock.patch.object(data.u,'ROOT',root):expected=independent.scalar(row,'probex',0,8)
            self.assertTrue(np.array_equal(actual.view('u4'),expected.view('u4')))
            self.assertTrue(np.array_equal(actual.view('u4'),np.clip(raw[mapping[:8]],0,6).view('u4')))
            mapping[0]=1760494;np.save(root/'map.npy',mapping)
            with self.assertRaises(ValueError):ops.scalar(root,row,'probex',0,8)

    def test_all195_features_preserved_in_complete_813_layout(self):
        z=np.arange(3*467,dtype=np.float64).reshape(3,467);f=np.arange(3*617,dtype='<f4').reshape(3,617)
        columns=[np.full(3,float(i)/100,dtype='<f4') for i in range(195)]
        design,features=ops.extend(z,f,columns)
        self.assertEqual(design.shape,(3,662));self.assertEqual(features.shape,(3,812))
        self.assertTrue(np.array_equal(design[:,:466],z[:,:466]));self.assertTrue(np.array_equal(design[:,-1],z[:,-1]))
        self.assertTrue(np.array_equal(features[:,:617],f))
        self.assertTrue(np.array_equal(features[:,617:],np.column_stack(columns)-np.float32(3.6)))

    def test_every_eligible_width_retains_full_context_signal_partition(self):
        for width in range(618,814):
            self.assertEqual(sorted(neural.signal_indices(width)+neural.CONTEXT),list(range(width)))
        for width in [617,814]:
            with self.assertRaises(ValueError):neural.signal_indices(width)
        self.assertEqual(neural.CONFIGS['h128']['seeds'],[2026,7,13])
        for key in ['nn_hidden','context_rank','nn_epochs','nn_patience','nn_batch']:
            self.assertEqual(neural.CONFIGS['h128'][key],previous.CONFIGS['h128'][key])

    def test_zero_addition_model_initialization_matches_original_complete_recipe(self):
        import torch
        torch.manual_seed(12);old=previous.model_factory(618,previous.CONFIGS['h128'])
        torch.manual_seed(12);new=neural.model_factory(618,neural.CONFIGS['h128'])
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for name,value in old.state_dict().items():self.assertTrue(torch.equal(value,new.state_dict()[name]),name)

    def test_complete813_scaler_matches_independent_full_columns(self):
        rng=np.random.default_rng(3);features=rng.normal(size=(41,813)).astype('<f4');features[:,15]=3.6
        eligible=np.arange(41)%3!=1;expected=replay.independent_scaler(features,eligible)
        for size in [1,7,8192]:
            actual=scaler.fit_scaler(features,eligible,block_rows=size)
            for key in expected:self.assertTrue(np.array_equal(actual[key].view('u8'),expected[key].view('u8')),key)

    def test_full65535_supports_accept_nonunique_actual16_coefficients(self):
        old=json.loads((data.u.ROOT/'experiments/upstream-own14-matched391-calibration/calibration.json').read_text())
        gram=np.zeros((16,16));gram[:14,:14]=old['simplex_gram'];rhs=np.r_[old['simplex_rhs'],0.,0.]
        weights=np.r_[old['simplex_weights'],0.,0.];weights[13:]=weights[13]/3
        independent_weights,count=audit.support_solution(gram,rhs)
        self.assertEqual(count,65535);audit.verify_simplex_weights(gram,rhs,weights,independent_weights)
        broken=weights.copy();broken[0]+=0.01;broken[1]-=0.01
        with self.assertRaises(ValueError):audit.verify_simplex_weights(gram,rhs,broken,independent_weights)

    def test_inherited_expected_metadata_hash_required_for_new_known_alias(self):
        self.assertEqual(len(cal.BASE_IDS),15)
        self.assertEqual(cal.CANDIDATES,['own16-simplex','own16-ridge10','weekend-trajectory-fixed-half'])
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(data.u,'ROOT',Path(tmp)):
            with mock.patch.object(cal,'KNOWN',Path(tmp)/'new-neural'):
                cal.KNOWN.mkdir();path=cal.KNOWN/'labels.npy';np.save(path,np.array([4,5],dtype=np.int16));sha=ops.sha(path)
                bindings={'artifacts/upstream-weekend-trajectory-data/labels.npy':sha}
                cal.bind_known_metadata(bindings,'labels.npy')
                self.assertEqual(bindings['new-neural/labels.npy'],sha)
                with self.assertRaises(ValueError):cal.bind_known_metadata({},'labels.npy')

    def test_full_bag_arithmetic_keeps_zero_slot_and_float32_seed_division(self):
        baseline=np.array([3.1,3.2],dtype=np.float64)
        values=[np.array([.1,-.2],dtype='<f4'),None,np.array([.3,.4],dtype='<f4')]
        expected=np.asarray(np.clip(baseline+values[0]/3+values[2]/3,1.05,4.95),dtype='<f4')
        self.assertTrue(np.array_equal(bag.fold_values(baseline,values).view('u4'),expected.view('u4')))

    def test_preceding_local_reference_keeps_original_model_and_prediction_hashes(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(data.u,'ROOT',Path(tmp)):
            root=Path(tmp);report=root/'experiments/upstream-complete-trajectory-h128-mps'
            trained=root/'artifacts/upstream-complete-trajectory-h128-mps';report.mkdir(parents=True);trained.mkdir(parents=True)
            (trained/'nn.f32').write_bytes(b'known nn bits');(trained/'fixed-half-tree154.f32').write_bytes(b'known fixed half bits')
            write=lambda p,v:p.write_text(json.dumps(v))
            manifest=dict(source_sha256={},code_sha256={},qualifying_labels_read=False)
            mp=report/'manifest.json';write(mp,manifest);records=[];model_paths=[]
            for fold in range(5):
                folder=trained/'nn'/str(fold);folder.mkdir(parents=True)
                model=folder/'nn-2026.pt';model.write_bytes(('original model'+str(fold)).encode());model_paths.append(model)
                outputs={model.name:ops.sha(model)}
                path=report/('fold%s.json'%fold)
                write(path,dict(outputs=outputs,chosen_iterations=[1,0,0],manifest_sha256=ops.sha(mp)))
                records.append(dict(fold=fold,receipt_path=str(path.relative_to(root)),receipt_sha256=ops.sha(path),outputs=outputs,
                    actual_models_reopened=1,both_complete_eligible_scalers_independently_recomputed=True,
                    all_normalized_input_bits_equal=True,actual_saved_pipeline_bitwise_equal=True))
            vp=report/'independent-verification.json'
            actual=dict(state='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions',
                manifest_sha256=ops.sha(mp),qualifying_labels_read=False,complete_original_statistics_bitwise_equal=True,
                all_original_models_preserved_without_refitting=True,matched_B_known_and_qualifying=True,records=records,
                nn_sha256=ops.sha(trained/'nn.f32'),fixed_half_sha256=ops.sha(trained/'fixed-half-tree154.f32'),
                nn_rmse=.87,fixed_half_tree154_rmse=.86)
            write(vp,actual)
            canonical=dict(state='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores',
                manifest_sha256=ops.sha(mp),complete_actual_model_replay_sha256=ops.sha(vp),source_sha256={},qualifying_labels_read=False,
                all_five_original_folds_verified=True,every_nn_and_fixed_half_float32_bit_equal=True,neural_inputs_matched_B_on_both_queries=True,
                nn_rmse=.87,fixed_half_tree154_rmse=.86,prediction_sha256=actual['fixed_half_sha256'],actual_models_rehashed=5)
            write(report/'independent-canonical-verification.json',canonical)
            bindings={};reference=neural.bind_local_reference(bindings)
            self.assertEqual(reference['actual_models'],5);self.assertEqual(reference['fixed_half_rmse'],.86)
            for name,sha in bindings.items():self.assertEqual(ops.sha(root/name),sha)
            original=bindings[str(model_paths[0].relative_to(root))];model_paths[0].write_bytes(b'altered model')
            self.assertNotEqual(ops.sha(model_paths[0]),original)
            rebound={};neural.bind_local_reference(rebound)
            self.assertEqual(rebound[str(model_paths[0].relative_to(root))],original)


if __name__=='__main__':unittest.main()
