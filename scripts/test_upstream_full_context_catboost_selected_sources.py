"""Both-arm admission fixtures; known-study/source/owner replay is synthetic."""
from contextlib import contextmanager,ExitStack
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import upstream_full_context_catboost_selected_sources as module
import test_upstream_full_context_catboost_consumer as fixtures


class SelectedTests(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(module.study.wire.encoded(value)+b'\n');return module.study.sha(path)

    @contextmanager
    def fixture(self):
        with fixtures.ConsumerTests().fixture() as f,ExitStack() as stack:
            root=f['root'];study=module.study;e=module.owner;c=module.consumer
            for obj,name,value in [(study,'ROOT',root),(study,'ROWS',20),(study,'QUALIFYING_ROWS',23),
                    (e,'REPORT',root/'controller'),(e,'RESULTS',root/'controller-results')]:stack.enter_context(patch.object(obj,name,value))
            code=root/'fixture-code.py';code.write_text('# synthetic proof/code fixture')
            code_bindings={'fixture-code.py':study.sha(code)}
            stack.enter_context(patch.object(e,'check_sources'));stack.enter_context(patch.object(e,'validate_result'))
            stack.enter_context(patch.object(module.canonical,'code_bindings',return_value=code_bindings))
            stack.enter_context(patch.object(module.replay,'code_bindings',return_value=code_bindings))
            plan={'source_sha256':code_bindings};self.write(e.REPORT/'plan.json',plan)
            (e.REPORT/'first-scientific-baseline.json').write_bytes(study.BASELINE.read_bytes())
            labels=np.load(root/c.ARRAYS/'labels.npy');paths={a:(root/'art'/a,root/'reports'/a) for a in study.ARMS}
            stack.enter_context(patch.object(study,'paths',side_effect=lambda a:paths[a]))
            cpu_path=root/c.DATA/'manifest.json';cpu=json.loads(cpu_path.read_text())
            qoperators={'scalar':['2'*64,'3'*64],'latent':['4'*64,'5'*64]}
            cpu['virtual_c_order_sha256']['qual']={'arm_operator_sha256':qoperators};cpu_sha=self.write(cpu_path,cpu)
            query_code=root/'scripts/upstream_full_context_catboost_query_producer.py';query_code.parent.mkdir();query_code.write_text('# synthetic query source')
            arms={};canon_sha={};scores={}
            for arm in study.ARMS:
                run,report=paths[arm];run.mkdir(parents=True);report.mkdir(parents=True)
                prediction=np.full(20,3. if arm=='scalar' else 3.1,dtype='<f4');np.save(run/'prediction.npy',prediction)
                metadata=copy.deepcopy(f['metadata']);metadata['arm']=arm;metadata['dimensions']=c.operators.dimensions(False,arm=='latent')
                metadata['parent_CPU_manifest_sha256']=cpu_sha;metadata['source_sha256'][str(cpu_path.relative_to(root))]=cpu_sha
                metadata_sha=hashlib.sha256(study.wire.encoded(metadata)).hexdigest();contract={'synthetic_contract':arm}
                m=dict(dimensions=metadata['dimensions'],source_metadata_sha256=metadata_sha,outputs={},authorized_prepare_bindings={})
                for name in ('source','contract','weights'):
                    path=report/(name+'.json');digest=self.write(path,{'synthetic':name})
                    m['outputs'][name]=dict(path=str(path),sha256=digest)
                manifest_sha=self.write(report/'manifest.json',m);fit_sha=self.write(report/'fit-completion.json',{'synthetic_completed_fit':True})
                outputs={str((run/'prediction.npy').relative_to(root)):study.sha(run/'prediction.npy')}
                p=dict(state=module.replay.STATE,arm=arm,rows=20,qualifying_rows=23,manifest_sha256=manifest_sha,fit_completion_sha256=fit_sha,
                    original_seed_slots=15,ridge_equations=10,qualifying_labels_read=False,
                    every_original_raw_population_and_held_prediction_bit_verified=True,every_full_OOF_join_bit_verified=True,
                    every_eligible_ridge_equation_and_coefficient_bit_verified=True,verifier_code_sha256=code_bindings,output_sha256=outputs)
                replay_sha=self.write(report/'independent-verification.json',p)
                score=module.canonical.score(prediction,labels);scores[arm]=score
                canonical=dict(state=module.canonical.STATE,arm=arm,rows=20,qualifying_labels_read=False,score_is_complete_five_user_fold_CV=True,
                    canonical_code_sha256=code_bindings,manifest_sha256=manifest_sha,fit_completion_sha256=fit_sha,independent_replay_sha256=replay_sha,
                    prediction_sha256=outputs[str((run/'prediction.npy').relative_to(root))],tree_only_CV_rmse=score)
                canon_sha[arm]=self.write(report/'independent-canonical-verification.json',canonical)
                for kind in ('fit','replay'):
                    exit_proof=dict(state='complete_owned_original_tree_producer_zero_exit',exit_code=0,observed_by_actual_parent_waitpid=True,
                        source_admission_sha256=metadata_sha,contract_sha256=hashlib.sha256(study.wire.encoded(contract)).hexdigest(),qualifying_labels_read=False)
                    exit_sha=self.write(report/(kind+'-producer-exit.json'),exit_proof)
                    admission_sha=self.write(report/(kind+'-producer-admission.json'),{'synthetic_admission':True})
                    external=report/(kind+'-external.json');external_sha=self.write(external,{'synthetic_external_owner':True})
                    pipe=dict(state='complete_actual_isolated_full_context_tree_pipeline_component',mode=kind,arm=arm,manifest_sha256=manifest_sha,
                        scientific_proof_sha256=study.sha(report/('fit-completion.json' if kind=='fit' else 'independent-verification.json')),
                        producer_reaped_zero_before_any_tree_fit_or_saved_model_replay=True,qualifying_labels_read=False,
                        actual_producer_exit_sha256=exit_sha,producer_admission_sha256=admission_sha,external_authorization=dict(path=str(external),sha256=external_sha))
                    self.write(report/(kind+'-pipeline-completion.json'),pipe)
                arms[arm]=(m,metadata,contract,{'synthetic_fit':True},[],outputs)
            stack.enter_context(patch.object(module.replay,'bindings',side_effect=lambda arm,*a:arms[arm]))
            selected=root/'experiments/upstream-full-context-catboost-selection/selection.json'
            selection=dict(state=module.canonical.SELECTION_STATE,canonical_sha256=canon_sha,tree_only_known_CV_scores=scores,selected_arm='scalar',
                selected_known_prediction_sha256=arms['scalar'][5][str((paths['scalar'][0]/'prediction.npy').relative_to(root))],
                selection_uses_only_complete_known_CV=True,qualifying_labels_read=False,tie_tolerance=1e-12,tie_rule='scalar',production_study_admitted=False)
            selected_sha=self.write(selected,selection)
            completed={}
            for phase in e.PHASES:
                path=e.RESULTS/(phase+'.json');completed[str(path.relative_to(root))]=self.write(path,{'synthetic_phase_dispatch':phase})
                path=e.REPORT/(phase+'-resource-peaks.json')
                completed[str(path.relative_to(root))]=self.write(path,dict(state='complete_observed_native_original_family_resource_samples',phase=phase,
                    continuous_peak_claimed=False,original_baseline_sha256=study.BASELINE_SHA))
            done=dict(state=e.COMPLETE,verification_sha256=selected_sha,qualifying_labels_read=False,completed_phase_and_observed_resource_proof_sha256=completed)
            self.write(e.REPORT/'completion.json',done)
            yield dict(root=root,paths=paths,arms=arms,selection=selection,selected=selected,selected_sha=selected_sha,done=done,qoperators=qoperators,query_code=query_code)

    def reseal(self,f):
        for arm,(run,report) in f['paths'].items():
            p=report/'independent-verification.json';c=report/'independent-canonical-verification.json'
            canonical=json.loads(c.read_text());canonical['independent_replay_sha256']=module.study.sha(p)
            f['selection']['canonical_sha256'][arm]=self.write(c,canonical)
            pipe=report/'replay-pipeline-completion.json';value=json.loads(pipe.read_text());value['scientific_proof_sha256']=module.study.sha(p);self.write(pipe,value)
        digest=self.write(f['selected'],f['selection']);f['done']['verification_sha256']=digest
        self.write(module.owner.REPORT/'completion.json',f['done']);return digest

    def query(self,f,known):
        source=known['arms'][known['arm']];original=source['metadata'];code=f['query_code']
        bindings=dict(original['source_sha256']);bindings[str(code.relative_to(f['root']))]=module.study.sha(code)
        q=dict(format='own_full_context_tree_original_qualifying_source_v1',query='qual',arm=known['arm'],rows=23,known_rows=20,
            dimensions=source['manifest']['dimensions'],retain_scalar=False,original_known_source_metadata=original,
            original_known_source_metadata_sha256=source['manifest']['source_metadata_sha256'],weights_sha256=original['weights_sha256'],
            query_code_sha256={str(code.relative_to(f['root'])):module.study.sha(code)},source_sha256=bindings,
            models_fitted=0,qualifying_labels_read=False,selection_authorized=False,production_data_admitted=False,operator_sha256=f['qoperators'][known['arm']])
        return q

    def test_both_complete_known_scores_rechecked_and_matching_source_contract(self):
        with self.fixture() as f:
            known=module.load(f['selected_sha'],lambda:None);self.assertEqual(known['arm'],'scalar')
            self.assertFalse(known['production_data_admitted']);self.assertEqual(set(known['arms']),{'scalar','latent'})
            q=self.query(f,known);path=f['root']/'query.json';digest=self.write(path,q)
            actual,contract=module.qualifying_metadata(known,path,digest,lambda:None)
            self.assertEqual(actual,q);self.assertEqual(contract['rows'],23);self.assertEqual(contract['width'],713)
            self.assertEqual(contract['admission_sha256'],hashlib.sha256(module.study.wire.encoded(q)).hexdigest())
            self.assertNotEqual(contract['admission_sha256'],digest)

    def test_partial_second_arm_rejected_even_with_rehashed_parent_receipts(self):
        with self.fixture() as f:
            path=f['paths']['latent'][1]/'independent-verification.json';p=json.loads(path.read_text());p['every_full_OOF_join_bit_verified']=False;self.write(path,p)
            digest=self.reseal(f)
            with self.assertRaisesRegex(ValueError,'replay closure'):module.load(digest,lambda:None)

    def test_changed_complete_known_score_is_not_accepted_as_selection_metadata(self):
        with self.fixture() as f:
            path=f['paths']['scalar'][1]/'independent-canonical-verification.json';c=json.loads(path.read_text());c['tree_only_CV_rmse']+=.2;self.write(path,c)
            f['selection']['tree_only_known_CV_scores']['scalar']=c['tree_only_CV_rmse'];digest=self.reseal(f)
            with self.assertRaisesRegex(ValueError,'known scores'):module.load(digest,lambda:None)

    def test_unmatched_query_arm_weights_operator_or_unapproved_source_rejected(self):
        with self.fixture() as f:
            known=module.load(f['selected_sha'],lambda:None);original=self.query(f,known);path=f['root']/'query.json'
            cases=[('arm','latent'),('weights_sha256','f'*64),('operator_sha256',['8'*64,'9'*64])]
            for key,value in cases:
                q=copy.deepcopy(original);q[key]=value;digest=self.write(path,q)
                with self.assertRaises(ValueError):module.qualifying_metadata(known,path,digest,lambda:None)
            q=copy.deepcopy(original);q['source_sha256']['unapproved-qualifying-outcomes']='c'*64;digest=self.write(path,q)
            with self.assertRaises(ValueError):module.qualifying_metadata(known,path,digest,lambda:None)


if __name__=='__main__':unittest.main()
