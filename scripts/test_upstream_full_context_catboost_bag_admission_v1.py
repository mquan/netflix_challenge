"""Actual small-model pair/replay; owner authorization/resource proofs synthetic."""
from pathlib import Path
import subprocess
import sys
import unittest


class AdmissionTests(unittest.TestCase):
    def child(self,body):
        code="import sys\nsys.path.insert(0,sys.argv[1])\n"+self.FIXTURE+body
        result=subprocess.run([sys.executable,'-c',code,str(Path(__file__).resolve().parent)],
            start_new_session=True,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    FIXTURE=r'''
from contextlib import ExitStack
from unittest import mock
import hashlib,json,os
import upstream_full_context_catboost_bag_execution_v1 as e
import test_upstream_full_context_catboost_bag_v1 as fixtures
import test_verify_upstream_full_context_catboost_bag_v1 as replay_fixtures
with fixtures.BagTests().fixture() as f,ExitStack() as stack:
    root=f['root'];guard=f['guard'];control=root/'control';results=root/'results';source_report=root/'source-report'
    for name,value in [('ROOT',root),('REPORT',control),('RESULTS',results),('SOURCE_REPORT',source_report)]:
        stack.enter_context(mock.patch.object(e,name,value))
    code=root/'fixture-code.py';worker=root/'upstream_fixture_worker.py';worker.write_text('# synthetic worker authority')
    stack.enter_context(mock.patch.object(e,'WORKER',worker))
    stack.enter_context(mock.patch.object(e.pipeline,'code_paths',return_value=[code]))
    stack.enter_context(mock.patch.object(e.replay,'code_bindings',return_value={code.name:e.study.sha(code)}))
    stack.enter_context(mock.patch.object(e,'check_sources'))
    known=dict(f['known'],selection_sha256='a'*64)
    stack.enter_context(mock.patch.object(e,'query_inputs',return_value=(known,f['query'],f['file_sha'],f['metadata'],f['contract'])))
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(e.study.wire.encoded(value)+b'\n');return e.study.sha(path)
    plan={'source_sha256':{code.name:e.study.sha(code)}};write(control/'plan.json',plan);plan_sha=e.study.sha(control/'plan.json')
    owner=e.identity.process(os.getpid());assert owner['pid']==owner['pgid']
    for path in (control/'monitor.json',control/'prerequisites.json',control/'admit-gate.json',control/'admit-ownership.json'):
        write(path,{'synthetic_owner_authority_only':True})
    (control/'first-scientific-baseline.json').write_bytes(e.study.BASELINE.read_bytes())
    write(source_report/'preparation.json',dict(state=e.PREPARED,models_fitted=0,qualifying_labels_read=False))
    def finish_phase(phase):
        path,state=e.scientific_path(phase)
        write(results/(phase+'.json'),dict(state='complete_owned_original_tree_pair_'+phase,
            execution_plan_sha256=plan_sha,scientific_proof_sha256=e.study.sha(path),qualifying_labels_read=False))
        write(control/(phase+'-resource-peaks.json'),dict(state='complete_observed_native_original_family_resource_samples',phase=phase,
            continuous_peak_claimed=False,original_baseline_sha256=e.study.BASELINE_SHA,observed_peaks={'synthetic_sample_only':True}))
    finish_phase('prepare')
    gates={}
    def gate_for(phase,original):
        def gate(wire):
            receipt=original(wire);producer=receipt['producer']
            # Numerical pipe and actual parent exit are real; controller role
            # authorization and resource receipts in this fixture are synthetic.
            write(control/(phase+'-ownership.json'),owner)
            write(control/(phase+'-roles.json'),dict(scientific_root_pid=owner['pid'],roles={'producer':producer}))
            bindings=dict(known['source_sha256']);e.selected.merge(bindings,f['metadata']['source_sha256'])
            e.selected.merge(bindings,{str(f['query'].relative_to(root)):f['file_sha'],code.name:e.study.sha(code),worker.name:e.study.sha(worker)})
            completion=e.study.sha(e.bag.paths()[1]/'completion.json') if phase=='replay' else None
            if phase=='replay':e.selected.merge(bindings,{str((e.bag.paths()[1]/'completion.json').relative_to(root)):completion})
            request=dict(mode=phase,arm='scalar',command=e.producer_command(phase,owner['pid']),scientific_root=owner,producer=producer,
                selection_sha256='a'*64,completion_sha256=completion,query_metadata_path=str(f['query']),query_metadata_sha256=f['file_sha'],
                source_metadata_sha256=f['logical_sha'],contract_sha256=hashlib.sha256(e.study.wire.encoded(f['contract'])).hexdigest(),
                source_sha256=bindings,qualifying_labels_read=False)
            auth_path=control/(phase+'-source-authorization.json')
            auth=dict(path=str(auth_path),sha256=write(auth_path,dict(state='authorized_original_full_context_tree_query_producer_role',request=request)))
            admission=e.pipeline.paths()/(phase+'-producer-admission.json')
            admission_sha=write(admission,dict(request=request,external_authorization=auth,production_data_admitted=False,qualifying_labels_read=False))
            closure=dict(bindings);e.selected.merge(closure,{str(auth_path.relative_to(root)):auth['sha256'],str(admission.relative_to(root)):admission_sha})
            exit_path=e.pipeline.paths()/(phase+'-producer-exit.json')
            exit_proof=dict(state='complete_owned_original_tree_producer_zero_exit',exit_code=0,producer=producer,owner=owner,
                scientific_group=owner['pid'],contract_sha256=request['contract_sha256'],source_admission_sha256=f['logical_sha'],
                source_sha256=closure,observed_by_actual_parent_waitpid=True,qualifying_labels_read=False,production_study_admitted=False)
            digest=write(exit_path,exit_proof)
            gates[phase]=dict(request=request,closure=closure,auth=auth,admission_sha=admission_sha,exit_sha=digest,producer=producer,completion=completion)
            return dict(path=exit_path,sha256=digest,producer=producer)
        return gate
    def pipeline_completion(phase):
        g=gates[phase];path,state=e.scientific_path(phase)
        write(e.pipeline.paths()/(phase+'-pipeline-completion.json'),dict(state='complete_actual_isolated_selected_tree_query_pipeline_component',
            mode=phase,arm='scalar',selection_sha256='a'*64,completion_sha256=g['completion'],scientific_proof_sha256=e.study.sha(path),
            query_metadata_sha256=f['file_sha'],source_metadata_sha256=f['logical_sha'],actual_producer_exit_sha256=g['exit_sha'],
            producer_admission_sha256=g['admission_sha'],external_authorization=g['auth'],scientific_root=owner,producer=g['producer'],
            source_sha256=g['closure'],producer_reaped_zero_before_any_pair_publication=True,production_data_admitted=False,qualifying_labels_read=False))
        finish_phase(phase)
    e.bag.infer('a'*64,f['query'],f['file_sha'],f['source'],guard=guard,producer_exit_gate=gate_for('infer',f['gate']))
    pipeline_completion('infer')
    with replay_fixtures.ReplayTests().fresh(f) as (stream,gate):
        e.replay.verify(e.study.sha(e.bag.paths()[1]/'completion.json'),stream,guard=guard,producer_exit_gate=gate_for('replay',gate))
    pipeline_completion('replay')
'''

    def test_real_saved_models_and_separate_replay_pass_full_pair_checks(self):
        self.child('''
    value=e.admit_pair(guard)
    assert value['state']==e.ADMITTED and value['known_rows']==20 and value['qualifying_rows']==23
    assert value['every_original_raw_ridge_model_fold_bag_and_matched_known_bit_independently_verified']
    assert value['both_actual_owned_source_exit_and_authorization_closures_verified']
    assert value['production_data_admitted'] is False and value['models_fitted']==0
    assert set(value['outputs'])=={'known','qualifying'} and 'torch' not in sys.modules
''')

    def test_changed_actual_vector_cannot_pass_final_pair_checks(self):
        self.child('''
    path=e.bag.paths()[0]/'prediction.f32';raw=bytearray(path.read_bytes());raw[0]^=1;path.write_bytes(raw)
    try:e.admit_pair(guard)
    except ValueError as error:assert 'hash mismatch' in str(error)
    else:raise AssertionError('Changed actual prediction admitted')
    assert not (source_report/'pair-admission.json').exists()
''')

    def test_rehashed_pipeline_with_wrong_original_parent_is_rejected(self):
        self.child('''
    path=e.pipeline.paths()/'replay-pipeline-completion.json';value=e.read(path)
    value['producer']['ppid']+=1;write(path,value)
    try:e.admit_pair(guard)
    except ValueError as error:assert 'lifetimes' in str(error)
    else:raise AssertionError('Wrong original source parent admitted')
    assert not (source_report/'pair-admission.json').exists()
''')

    def test_rehashed_replay_without_independent_arithmetic_is_rejected(self):
        self.child('''
    path=e.bag.paths()[1]/'independent-verification.json';value=e.read(path)
    value['independent_arithmetic_does_not_call_production_predict']=False;write(path,value)
    pipe=e.pipeline.paths()/'replay-pipeline-completion.json';value=e.read(pipe);value['scientific_proof_sha256']=e.study.sha(path);write(pipe,value)
    wrapper=results/'replay.json';value=e.read(wrapper);value['scientific_proof_sha256']=e.study.sha(path);write(wrapper,value)
    try:e.admit_pair(guard)
    except ValueError as error:assert 'independently replayed' in str(error)
    else:raise AssertionError('Production arithmetic substituted for independent replay')
''')


if __name__=='__main__':unittest.main()
