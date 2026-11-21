"""Private metadata/file fixtures; fake proofs here are never numerical admission."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import complete_trajectory_publication as gate

SOURCE=Path(__file__).resolve().parents[1]


class Publication(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        def copy_source(name):
            dest=self.root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
        queue='experiments/upstream-complete-trajectory-neural-queue/plan.json';copy_source(queue)
        for name in gate.read(self.root/queue)['source_sha256']:copy_source(name)
        ref='experiments/upstream-own391-paired-h128-mps/'
        for name in ('manifest.json','independent-canonical-verification.json'):copy_source(ref+name)
        self.m=copy.deepcopy(gate.read(SOURCE/gate.REPORT/'manifest.json'))
        for name in self.m['code_sha256']:copy_source(name)
        self.data_sources={name:self.m['code_sha256'][name] for name in
            ('scripts/upstream_complete_trajectory_data.py','scripts/upstream_complete_trajectory_ops.py')}
        self.im=dict(retained=list(range(95)),qualifying_labels_read=False,source_sha256=self.data_sources)
        self.iv=dict(state='complete_independent_all519_scalar_pairs_and_both_operators',retained_additions=95,
            exact_paired_aliases=33,original391_retained=True,all128_candidates_replayed=True,
            every_complete_parent_operator_bit_equal=True,qualifying_labels_read=False,source_sha256=self.data_sources)
        dims={k:self.m[k] for k in ('raw_predictions','design_columns','nonlinear_columns','nn_inputs')}
        self.dm=dict(dims,retained=self.im['retained'],source_sha256=self.data_sources)
        self.dv=dict(dims,state='complete_independent_all_trajectory_operators_statistics_ridge_and_stored_predictions',
            source_sha256=self.data_sources,all_original_paired_B391_bits_preserved=True,
            every_full_operator_statistic_solve_and_prediction_bit_equal=True,matched_B_known_and_qualifying=True,
            qualifying_labels_read=False,neural_models_fitted=0)
        self.m['source_sha256']=dict(self.m['code_sha256'])
        self.tree='artifacts/private-tree154.f32';self.put_bytes(self.tree,b'fixture-tree')
        self.m['controls']=[dict(id='tree154',prediction=self.tree)]
        for name in (self.tree,ref+'independent-canonical-verification.json'):
            self.m['source_sha256'][name]=gate.digest(self.root/name)
        self.records=[];self.receipts=[]
        for fold in range(5):
            rows=gate.read(SOURCE/gate.REPORT/'nn'/('%d.json'%fold))['canonical_rows']
            files={}
            for name in ('ridge.npz','scalers.npz','prediction.f32','nn-2026.pt','nn-7.pt','nn-13.pt'):
                path=gate.RUN+'/nn/'+str(fold)+'/'+name;self.put_bytes(path,('fixture '+path).encode());files[name]=gate.digest(self.root/path)
            receipt=dict(fold=fold,canonical_rows=rows,chosen_iterations=[1,2,3],outputs=files)
            self.receipts.append(receipt)
            self.records.append(dict(fold=fold,rows=rows,actual_models_reopened=3,
                receipt_path=gate.REPORT+'/nn/'+str(fold)+'.json',outputs=files,
                both_complete_eligible_scalers_independently_recomputed=True,all_normalized_input_bits_equal=True,
                actual_saved_pipeline_bitwise_equal=True))
        for name in ('nn.f32','fixed-half-tree154.f32'):self.put_bytes(gate.RUN+'/'+name,('fixture '+name).encode())
        self.model=dict(state='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions',
            variant='h128',records=self.records,verifier_sha256=gate.digest(self.root/'scripts/verify_upstream_complete_trajectory_mps.py'),
            complete_original_statistics_bitwise_equal=True,all_original_models_preserved_without_refitting=True,
            matched_B_known_and_qualifying=True,nn_rmse=.8589,fixed_half_tree154_rmse=.8588,
            nn_sha256=gate.digest(self.root/gate.RUN/'nn.f32'),fixed_half_sha256=gate.digest(self.root/gate.RUN/'fixed-half-tree154.f32'),
            declared_global_reference_rmse=self.m['declared_global_reference_rmse'],fixed_half_below_declared_global_reference=True,
            qualifying_labels_read=False,production_started=False,goal_complete=False)
        self.current=dict(state='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores',variant='h128',
            verifier_sha256=gate.digest(self.root/'scripts/verify_upstream_complete_trajectory_canonical.py'),
            actual_models_rehashed=15,all_five_original_folds_verified=True,every_nn_and_fixed_half_float32_bit_equal=True,
            nn_rmse=.8589,fixed_half_tree154_rmse=.8588,prediction_sha256=self.model['fixed_half_sha256'],
            declared_global_reference_rmse=self.m['declared_global_reference_rmse'],improves_previous=True,
            qualifying_labels_read=False,goal_complete=False,tree_control_retains_historical_A_B_realization_caveat=True,
            neural_inputs_matched_B_on_both_queries=True)
        self.write(gate.REPORT+'/fit-completion.json',dict(state='complete_all_five_original_budget_complete_trajectory_neural_fits'))
        self.refresh()

    def put_bytes(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)

    def write(self,name,value):self.put_bytes(name,(json.dumps(value,indent=2)+'\n').encode())

    def refresh(self):
        self.write(gate.DATA+'/input-manifest.json',self.im)
        self.iv['input_manifest_sha256']=gate.digest(self.root/gate.DATA/'input-manifest.json')
        self.write(gate.DATA+'/input-verification.json',self.iv)
        self.dm.update(input_manifest_sha256=self.iv['input_manifest_sha256'],
            input_verification_sha256=gate.digest(self.root/gate.DATA/'input-verification.json'))
        self.write(gate.DATA+'/manifest.json',self.dm)
        self.dv.update(manifest_sha256=gate.digest(self.root/gate.DATA/'manifest.json'),input_verification_sha256=self.dm['input_verification_sha256'])
        self.write(gate.DATA+'/independent-verification.json',self.dv)
        for name in ('manifest.json','independent-verification.json'):
            p=gate.DATA+'/'+name;self.m['source_sha256'][p]=gate.digest(self.root/p)
        self.m['input_manifest_sha256']=self.dv['manifest_sha256'];self.write(gate.REPORT+'/manifest.json',self.m)
        manifest_sha=gate.digest(self.root/gate.REPORT/'manifest.json')
        for receipt,row in zip(self.receipts,self.records):
            receipt['manifest_sha256']=manifest_sha;self.write(row['receipt_path'],receipt)
            row['receipt_sha256']=gate.digest(self.root/row['receipt_path'])
        self.model['manifest_sha256']=manifest_sha;self.write(gate.REPORT+'/independent-verification.json',self.model)
        self.current.update(manifest_sha256=manifest_sha,source_sha256=self.m['source_sha256'],
            complete_actual_model_replay_sha256=gate.digest(self.root/gate.REPORT/'independent-verification.json'))
        self.write(gate.REPORT+'/independent-canonical-verification.json',self.current)
        digest=gate.digest(self.root/gate.REPORT/'independent-canonical-verification.json')
        self.record=dict(independent_verification_sha256=digest,verified_development_report_sha256=digest,best_development_rmse=self.current['fixed_half_tree154_rmse'])

    def validate(self):
        gate.validate(self.root,self.record,self.root/gate.REPORT/'independent-canonical-verification.json',self.current,
            self.root/gate.REPORT/'independent-verification.json',self.model)

    def test_complete_private_schema_fixture(self):self.validate()

    def test_coherently_rehashed_partial_models_or_false_replay_flags(self):
        for row,key,value in ((self.current,'actual_models_rehashed',14),(self.records[0],'actual_models_reopened',2),
            (self.records[1],'all_normalized_input_bits_equal',False),(self.current,'every_nn_and_fixed_half_float32_bit_equal',False)):
            before=row[key];row[key]=value;self.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
            row[key]=before

    def test_coherently_rehashed_shorter_recipe_or_missing_candidate(self):
        self.m['config']['nn_epochs']=20;self.refresh()
        with self.assertRaises(ValueError):self.validate()
        self.m['config']['nn_epochs']=40;self.iv['exact_paired_aliases']=32;self.refresh()
        with self.assertRaises(ValueError):self.validate()

    def test_real_file_tampering_in_code_models_scalers_and_vectors(self):
        names=['scripts/upstream_complete_trajectory_mps.py','scripts/upstream_mps_context.py',
            gate.RUN+'/nn/0/nn-2026.pt',gate.RUN+'/nn/0/scalers.npz',gate.RUN+'/nn.f32',gate.RUN+'/fixed-half-tree154.f32']
        for name in names:
            p=self.root/name;before=p.read_bytes();p.write_bytes(before+b'changed')
            try:
                with self.subTest(name=name),self.assertRaises(ValueError):self.validate()
            finally:p.write_bytes(before)

    def test_pending_or_wrong_score_cannot_be_promoted(self):
        self.current['state']='pending_full_replay';self.refresh()
        with self.assertRaises(ValueError):self.validate()
        self.current['state']='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores';self.refresh()
        self.record['best_development_rmse']=.85
        with self.assertRaises(ValueError):self.validate()

    def test_coherently_rehashed_external_source_binding(self):
        self.m['source_sha256']['../external.py']='0'*64;self.refresh()
        with self.assertRaises(ValueError):self.validate()

    def test_coherently_rehashed_context_signal_exchange_is_rejected(self):
        self.m['context_indices'][0],self.m['signal_indices'][0]=self.m['signal_indices'][0],self.m['context_indices'][0]
        self.refresh()
        with self.assertRaises(ValueError):self.validate()


if __name__=='__main__':unittest.main()
