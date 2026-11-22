"""Actual dedicated-root/producer pipe tests; source/ownership are synthetic."""
from pathlib import Path
import subprocess
import sys
import unittest


class PipelineTests(unittest.TestCase):
    def child(self,body):
        directory=str(Path(__file__).resolve().parent)
        code="import sys\nsys.path.insert(0,sys.argv[1])\nimport upstream_full_context_catboost_pipeline as p\n"+body
        value=subprocess.run([sys.executable,'-c',code,directory],start_new_session=True,capture_output=True,text=True,timeout=90)
        self.assertEqual(value.returncode,0,value.stdout+value.stderr)

    FIXTURE='''
from contextlib import ExitStack
from unittest import mock
import hashlib,io,json
import test_upstream_full_context_catboost_study as fixtures
with fixtures.StudyTests().fixture() as (f,prepared,ownership),ExitStack() as stack:
    root=f['root'];guard=lambda **kw:None
    stack.enter_context(mock.patch.object(p,'code_paths',return_value=[root/'fixture-study-code.py']))
    receipt=p.study.prepare_arm('scalar',prepared,ownership,guard)
    manifest_sha=receipt['sha256'];run,report=p.study.paths('scalar')
    contract=prepared['contract'];wire=p.consumer.wire
    packet=io.BytesIO();wire.write_all(packet,wire.MAGIC,guard);wire.write_metadata(packet,contract,guard)
    wire.write_all(packet,f['x'],guard);wire.write_all(packet,f['b'],guard)
    footer=dict(raw_sha256=hashlib.sha256(f['x'].tobytes()).hexdigest(),
        baseline_sha256=[hashlib.sha256(f['b'][fold,pop].tobytes()).hexdigest() for fold in range(5) for pop in range(2)],
        operator_sha256=contract['operator_sha256'],admission_sha256=contract['admission_sha256'])
    wire.write_metadata(packet,footer,guard);payload=root/'payload.bin';payload.write_bytes(packet.getvalue())
    producer=root/'upstream_fixture_producer.py'
    producer.write_text("import sys,time,os\\nassert sys.stdin.buffer.read(1)==b'G'\\nimport torch\\nwith open(sys.argv[1],'rb') as f:sys.stdout.buffer.write(f.read())\\nsys.stdout.buffer.flush()\\nos.close(1)\\ntime.sleep(.2)\\nsys.exit(int(sys.argv[2]))\\n")
    events=[]
    def admit(request):
        actual=p.consumer.identity.process(request['producer']['pid'])
        assert actual and actual['ppid']==request['scientific_root']['pid'] and actual['pgid']==request['scientific_root']['pid']
        events.append('actual-live-role-admitted')
        path=root/(request['mode']+'-external-authorization.json')
        p.study.save_json(path,dict(state='authorized_original_full_context_tree_producer_role',request=request),guard)
        return dict(path=str(path),sha256=p.study.sha(path))
    def preserve(root,roles,error):raise AssertionError('Successful fixture unexpectedly preserved a live failed role: '+repr(error))
    command=[sys.executable,str(producer),str(payload),'0']
'''

    def test_actual_dedicated_group_source_pipe_zero_exit_then_complete_small_fit(self):
        self.child(self.FIXTURE+'''
    result=p.run('fit','scalar',manifest_sha,command=command,guard=guard,admit_role=admit,preserve_family=preserve)
    assert result['all_five_folds_complete'] and result['original_seed_slots']==15
    assert result['production_study_admitted'] is False
    assert events==['actual-live-role-admitted'] and 'torch' not in sys.modules
    exit_proof=json.loads((report/'fit-producer-exit.json').read_text())
    assert exit_proof['observed_by_actual_parent_waitpid'] and exit_proof['exit_code']==0
    assert exit_proof['producer']['pgid']==exit_proof['owner']['pid']
    complete=json.loads((report/'fit-pipeline-completion.json').read_text())
    assert complete['scientific_proof_sha256']==p.study.sha(report/'fit-completion.json')
    assert complete['actual_producer_exit_sha256']==p.study.sha(report/'fit-producer-exit.json')
    assert 'fit-external-authorization.json' in exit_proof['source_sha256']
    assert complete['production_study_admitted'] is False
    assert not p.PRESERVED
    def replay_body(arm,manifest_sha,fit_sha,source,*,guard,producer_exit_gate):
        actual_x,actual_b,proof=wire.receive(source,contract,guard=guard)
        gate=producer_exit_gate(proof)
        p.consumer.producer_exit(gate['path'],gate['sha256'],gate['producer'],contract,contract['admission_sha256'],root,guard)
        value=dict(state='synthetic_replay_orchestration_body_only',qualifying_labels_read=False,production_study_admitted=False)
        p.study.save_json(report/'independent-verification.json',value,guard)
        return value
    fit_sha=p.study.sha(report/'fit-completion.json')
    with mock.patch.object(p.replay,'verify',side_effect=replay_body):
        replay_result=p.run('replay','scalar',manifest_sha,fit_sha256=fit_sha,command=command,guard=guard,admit_role=admit,preserve_family=preserve)
    assert replay_result['state']=='synthetic_replay_orchestration_body_only'
    assert (report/'replay-producer-exit.json').exists() and (report/'replay-pipeline-completion.json').exists()
    assert 'torch' not in sys.modules and events==['actual-live-role-admitted']*2
    try:p.run('fit','scalar',manifest_sha,command=command,guard=guard,admit_role=admit,preserve_family=preserve)
    except ValueError:pass
    else:raise AssertionError('Actual learned source namespace was retried')
''')

    def test_actual_nonzero_producer_exit_prevents_any_tree_fit_or_success_receipt(self):
        self.child(self.FIXTURE+'''
    command[-1]='7'
    with mock.patch.object(p.study.consumer,'fit',side_effect=AssertionError('failed producer started tree fitting')):
        try:p.run('fit','scalar',manifest_sha,command=command,guard=guard,admit_role=admit,preserve_family=preserve)
        except ValueError as error:assert 'failed' in str(error)
        else:raise AssertionError('Actual failed producer admitted')
    assert not (report/'fit-producer-exit.json').exists() and not (run/'folds').exists()
    assert (report/'fit-started.json').exists() and not p.PRESERVED
''')

    def test_mutated_authorization_after_release_prevents_any_fit_or_success_receipt(self):
        self.child(self.FIXTURE+r'''
    body=producer.read_text().replace('import torch\n',"import torch\nopen(sys.argv[3],'w').write('{}')\n")
    producer.write_text(body);command.append(str(root/'fit-external-authorization.json'))
    with mock.patch.object(p.study.consumer,'fit',side_effect=AssertionError('changed gate started fitting')):
        try:p.run('fit','scalar',manifest_sha,command=command,guard=guard,admit_role=admit,preserve_family=preserve)
        except ValueError as error:assert 'hash mismatch' in str(error)
        else:raise AssertionError('Changed immutable original authorization admitted')
    assert not (report/'fit-producer-exit.json').exists() and not (run/'folds').exists()
''')

    def test_failed_external_authorization_preserves_actual_live_role_and_channels(self):
        self.child(self.FIXTURE+'''
    def reject(request):raise ValueError('original authorization missing')
    def preserve(root,roles,error):
        events.append('live-role-preserved')
        assert roles['producer'] and p.consumer.identity.process(roles['producer']['pid'])
        assert not p.PRESERVED[-1]['handle'].stdout.closed
    try:p.run('fit','scalar',manifest_sha,command=command,guard=guard,admit_role=reject,preserve_family=preserve)
    except ValueError as error:assert 'authorization missing' in str(error)
    else:raise AssertionError('No source authorization required')
    assert events==['live-role-preserved'] and len(p.PRESERVED)==1
    # Only synthetic helpers are released for test cleanup. The component
    # provides no production kill/restart/release path.
    child=p.PRESERVED[0]['handle'];assert child.poll() is None
    child.stdin.write(b'G');child.stdin.flush();child.stdin.close()
    child.stdout.read();assert child.wait(timeout=30)==0;child.stdout.close();p.PRESERVED.clear()
    assert not (report/'fit-producer-admission.json').exists()
    assert not (report/'fit-started.json').exists()
''')


if __name__=='__main__':unittest.main()
