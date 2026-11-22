"""Actual isolated query-source lifecycle; scientific bodies are fixture stubs.

Full native model/ridge/vector inference and replay have separate component
tests. These checks exercise ownership, byte-EOF versus process exit, immutable
authorization, logical metadata hashes, and preservation in a dedicated group.
"""
from pathlib import Path
import subprocess
import sys
import unittest


class PipelineTests(unittest.TestCase):
    def child(self,body):
        code="import sys\nsys.path.insert(0,sys.argv[1])\nimport upstream_full_context_catboost_bag_pipeline_v1 as p\n"+body
        result=subprocess.run([sys.executable,'-c',code,str(Path(__file__).resolve().parent)],
            start_new_session=True,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    FIXTURE=r'''
from contextlib import ExitStack
from unittest import mock
from pathlib import Path
import hashlib,io,json,tempfile
import numpy as np
with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
    root=Path(temporary).resolve();guard=lambda **kw:None
    stack.enter_context(mock.patch.object(p.study,'ROOT',root))
    art=root/'artifacts/bag';report=root/'experiments/bag';control=p.paths()
    stack.enter_context(mock.patch.object(p.bag,'paths',return_value=(art,report)))
    code=root/'fixture-code.py';code.write_text('# isolated orchestration fixture')
    stack.enter_context(mock.patch.object(p,'code_paths',return_value=[code]))
    metadata=dict(source_sha256={code.name:p.study.sha(code)},synthetic=True)
    query=root/'query.json';query.write_bytes(p.study.wire.encoded(metadata)+b'\n')
    file_sha=p.study.sha(query);logical_sha=hashlib.sha256(p.study.wire.encoded(metadata)).hexdigest()
    assert file_sha!=logical_sha
    contract=p.consumer.wire.contract(23,retain_scalar=False,latent=False,
        weights_sha256='b'*64,admission_sha256=logical_sha,
        operator_sha256=['c'*64,hashlib.sha256(np.zeros((23,712),dtype='<f4').tobytes()).hexdigest()])
    known=dict(arm='scalar',source_sha256={code.name:p.study.sha(code)})
    stack.enter_context(mock.patch.object(p.bag.selected,'load',return_value=known))
    stack.enter_context(mock.patch.object(p.bag.selected,'qualifying_metadata',return_value=(metadata,contract)))
    packet=io.BytesIO();wire=p.consumer.wire;x=np.zeros((23,713),dtype='<f4');b=np.full((5,2,23),3.,dtype='<f8')
    wire.write_all(packet,wire.MAGIC,guard);wire.write_metadata(packet,contract,guard)
    wire.write_all(packet,x,guard);wire.write_all(packet,b,guard)
    wire.write_metadata(packet,dict(raw_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        baseline_sha256=[hashlib.sha256(b[f,pop].tobytes()).hexdigest() for f in range(5) for pop in range(2)],
        operator_sha256=contract['operator_sha256'],admission_sha256=logical_sha),guard)
    payload=root/'payload.bin';payload.write_bytes(packet.getvalue())
    producer=root/'upstream_fixture_query.py'
    producer.write_text("import sys,time,os\nassert sys.stdin.buffer.read(1)==b'G'\nimport torch\nwith open(sys.argv[1],'rb') as f:sys.stdout.buffer.write(f.read())\nsys.stdout.buffer.flush()\nos.close(1)\ntime.sleep(.2)\nsys.exit(int(sys.argv[2]))\n")
    command=[sys.executable,str(producer),str(payload),'0'];events=[]
    def admit(request):
        actual=p.consumer.identity.process(request['producer']['pid'])
        assert actual and actual['ppid']==request['scientific_root']['pid'] and actual['pgid']==request['scientific_root']['pid']
        assert request['query_metadata_sha256']==file_sha and request['source_metadata_sha256']==logical_sha
        events.append('actual-live-source-authorized')
        path=root/(request['mode']+'-authorization.json')
        p.study.save_json(path,dict(state='authorized_original_full_context_tree_query_producer_role',request=request),guard)
        return dict(path=str(path),sha256=p.study.sha(path))
    def preserve(root,roles,error):raise AssertionError('Unexpected live family failure: '+repr(error))
    def scientific(mode,source,gate):
        actual_x,actual_b,proof=wire.receive(source,contract,guard=guard)
        receipt=gate(proof)
        assert Path(receipt['path']).exists()
        assert p.consumer.identity.process(receipt['producer']['pid']) is None
        events.append('source-reaped-before-publication')
        p.study.durable_directory(report,root,guard)
        value=dict(state='synthetic_'+mode,selection_sha256='a'*64,query_metadata_path=str(query),query_metadata_sha256=file_sha)
        p.study.save_json(report/('completion.json' if mode=='infer' else 'independent-verification.json'),value,guard)
        return value
    def infer(selection,q,sha,source,*,guard,producer_exit_gate):return scientific('infer',source,producer_exit_gate)
    def replay(completion,source,*,guard,producer_exit_gate):return scientific('replay',source,producer_exit_gate)
    stack.enter_context(mock.patch.object(p.bag,'infer',side_effect=infer))
    stack.enter_context(mock.patch.object(p.replay,'verify',side_effect=replay))
    def run(mode='infer',**kwargs):
        return p.run(mode,'a'*64,query,file_sha,command=command,guard=guard,admit_role=admit,preserve_family=preserve,**kwargs)
'''

    def test_actual_fresh_inference_and_replay_sources_exit_before_publication(self):
        self.child(self.FIXTURE+'''
    assert run()['state']=='synthetic_infer'
    first=json.loads((control/'infer-producer-exit.json').read_text())
    assert first['observed_by_actual_parent_waitpid'] and first['exit_code']==0
    assert first['source_admission_sha256']==logical_sha
    assert 'infer-authorization.json' in first['source_sha256']
    complete=p.study.sha(report/'completion.json')
    assert run('replay',completion_sha256=complete)['state']=='synthetic_replay'
    second=json.loads((control/'replay-producer-exit.json').read_text())
    assert first['producer']['pid']!=second['producer']['pid']
    assert events==['actual-live-source-authorized','source-reaped-before-publication']*2
    for mode in ('infer','replay'):
        proof=json.loads((control/(mode+'-pipeline-completion.json')).read_text())
        assert proof['producer_reaped_zero_before_any_pair_publication']
        assert proof['production_data_admitted'] is False
    assert 'torch' not in sys.modules and not p.PRESERVED
    try:run()
    except ValueError:pass
    else:raise AssertionError('Original inference pipeline retried')
''')

    def test_actual_failed_source_after_full_EOF_never_publishes_pair(self):
        self.child(self.FIXTURE+'''
    command[-1]='7'
    try:run()
    except ValueError as error:assert 'failed' in str(error)
    else:raise AssertionError('Nonzero source admitted')
    assert not (report/'completion.json').exists()
    assert not (control/'infer-producer-exit.json').exists()
    assert not (control/'infer-pipeline-completion.json').exists()
    assert events==['actual-live-source-authorized'] and not p.PRESERVED
''')

    def test_source_mutation_of_authorization_rejected_before_publication(self):
        self.child(self.FIXTURE+r'''
    body=producer.read_text().replace('import torch\n',"import torch\nopen(sys.argv[3],'w').write('{}')\n")
    producer.write_text(body);command.append(str(root/'infer-authorization.json'))
    try:run()
    except ValueError as error:assert 'hash mismatch' in str(error)
    else:raise AssertionError('Changed source authorization admitted')
    assert not (report/'completion.json').exists() and not (control/'infer-producer-exit.json').exists()
''')

    def test_missing_authorization_preserves_actual_live_child_and_channels(self):
        self.child(self.FIXTURE+'''
    def admit(request):raise ValueError('Missing source authorization')
    def preserve(root,roles,error):
        assert p.consumer.identity.process(roles['producer']['pid'])
        assert not p.PRESERVED[-1]['handle'].stdout.closed
        events.append('live-role-preserved')
    try:run()
    except ValueError as error:assert 'Missing source authorization' in str(error)
    else:raise AssertionError('Unauthenticated source started')
    assert events==['live-role-preserved'] and len(p.PRESERVED)==1
    assert not (control/'infer-producer-admission.json').exists()
    # Release only this synthetic fixture helper. No scientific job is affected.
    child=p.PRESERVED[0]['handle'];child.stdin.write(b'G');child.stdin.flush();child.stdin.close()
    child.stdout.read();assert child.wait(timeout=30)==0;child.stdout.close();p.PRESERVED.clear()
''')


if __name__=='__main__':unittest.main()
