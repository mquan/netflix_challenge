"""Isolated original-source adapter fixtures; full-data proof is synthetic."""
from pathlib import Path
import subprocess
import sys
import unittest


class QueryProducerTests(unittest.TestCase):
    def child(self,body):
        directory=str(Path(__file__).resolve().parent)
        code="import sys\nsys.path.insert(0,sys.argv[1])\nimport upstream_full_context_catboost_query_producer as q\n"+body
        result=subprocess.run([sys.executable,'-c',code,directory],capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    FIXTURE='''
from contextlib import ExitStack,contextmanager
from unittest import mock
from pathlib import Path
import hashlib,io,json,copy
import numpy as np
import test_upstream_rbm400_neural_study_reference_v4 as fixtures
import upstream_full_context_catboost_query_stream as receiver
with fixtures.StudyTests().fixture(full_operators=True) as (root,manifest,tree),ExitStack() as stack:
    rng=np.random.default_rng(947)
    z=rng.normal(size=(103,563));z[:,-1]=1
    f=rng.normal(size=(103,863)).astype('<f4')
    manifest.update(rows=100,qualifying_rows=103)
    manifest['virtual_c_order_sha256']['qual']={'arm_operator_sha256':{
        arm:[hashlib.sha256(z.tobytes()).hexdigest(),hashlib.sha256(np.ascontiguousarray(f[:,:713 if arm=='scalar' else 863]).tobytes()).hexdigest()]
        for arm in ('scalar','latent')}}
    path=q.known.inputs.data.REPORT/'manifest.json'
    base_declaration,_,_=q.known.inputs.sources(lambda:None)
    (root/'query-adapter.py').write_text('# synthetic query-specific implementation')
    path.write_text(json.dumps({k:v for k,v in manifest.items() if k!='_fixture_operators'}))
    def sources(guard):
        declaration,value=copy.deepcopy(base_declaration),copy.deepcopy(manifest)
        declaration['source_sha256'][str(path.relative_to(root))]=q.known.sha(path)
        return declaration,value,q.consumer.verify_files(root,declaration['source_sha256'],guard)
    def verify(bindings,guard):return q.consumer.verify_files(root,bindings,guard)
    for module,name,value in [(q,'ROOT',root),(q,'QUALIFYING_ROWS',103),(q.known,'ROOT',root),(q.known,'ROWS',100),(q.known,'QUALIFYING_ROWS',103)]:stack.enter_context(mock.patch.object(module,name,value))
    stack.enter_context(mock.patch.object(q.known,'code_paths',return_value=[]))
    stack.enter_context(mock.patch.object(q,'code_paths',return_value=[root/'query-adapter.py']))
    stack.enter_context(mock.patch.object(q.known.inputs,'sources',side_effect=sources))
    stack.enter_context(mock.patch.object(q.known.inputs.cpu,'read',return_value={'state':'complete_both_RBM400_context_arms_data_and_independent_known_ridge_replay','qualifying_labels_read':False}))
    stack.enter_context(mock.patch.object(q.known.inputs.context.provider.original,'verify_bindings',side_effect=verify))
    calls=[]
    @contextmanager
    def open_query(query,declaration,records,guard,**kwargs):
        assert query=='qual' and set(declaration['source_sha256'])==set(records)
        assert kwargs['retain_scalar'] is True
        width=863 if kwargs['latent'] else 713
        class Reader:
            prediction=np.empty(103,dtype='<f4')
            def block(self,lo,hi):
                calls.append((lo,hi,width))
                return z[lo:hi],np.ascontiguousarray(f[lo:hi,:width])
        yield Reader()
    stack.enter_context(mock.patch.object(q.known.inputs.context,'open_query',side_effect=open_query))
'''

    def test_both_full_arms_stream_all_qualifying_rows_with_matching_ten_known_weights(self):
        self.child(self.FIXTURE+'''
    for arm in ('scalar','latent'):
        prepared=q.prepare(arm,lambda:None)
        metadata=prepared['metadata'];original=metadata['original_known_source_metadata']
        assert metadata['operator_sha256']!=original['operator_sha256']
        assert metadata['weights_sha256']==original['weights_sha256']
        assert 'query-adapter.py' not in original['source_sha256']
        assert 'query-adapter.py' in metadata['source_sha256']
        assert metadata['rows']==103 and metadata['known_rows']==100
        assert metadata['selection_authorized'] is False and metadata['production_data_admitted'] is False
        output=io.BytesIO();q.send(prepared,output,lambda:None)
        seen=[]
        def consume(lo,hi,x,b):
            width=863 if arm=='latent' else 713
            assert x[:,:-1].tobytes()==np.ascontiguousarray(f[lo:hi,:width]).tobytes()
            for fold in range(5):
                for pop in range(2):
                    expected=np.clip(z[lo:hi]@prepared['weights'][fold,pop],1.05,4.95)
                    assert b[fold,pop].tobytes()==expected.tobytes()
            seen.extend(range(lo,hi))
        proof=receiver.receive(io.BytesIO(output.getvalue()),prepared['contract'],consume,guard=lambda:None)
        assert seen==list(range(103)) and proof['exact_footer_and_EOF']
        assert not proof['prediction_publication_authorized']
    assert 'torch' in sys.modules and calls==[(0,103,713),(0,103,863)]
''')

    def test_changed_source_weights_or_query_contract_rejected(self):
        self.child(self.FIXTURE+'''
    prepared=q.prepare('scalar',lambda:None)
    original=prepared['weights'].copy()
    prepared['weights'][0,0,0]=np.nextafter(prepared['weights'][0,0,0],np.inf)
    try:q.send(prepared,io.BytesIO(),lambda:None)
    except ValueError:pass
    else:raise AssertionError('Changed original known weights admitted')
    prepared['weights'][:]=original
    changed=copy.deepcopy(prepared);changed['contract']['rows']=100
    try:q.send(changed,io.BytesIO(),lambda:None)
    except ValueError:pass
    else:raise AssertionError('Known row count substituted for qualifying rows')
    (root/'query-adapter.py').write_text('# changed query adapter')
    try:q.send(prepared,io.BytesIO(),lambda:None)
    except ValueError:pass
    else:raise AssertionError('Changed frozen qualifying code admitted')
    assert calls==[]
''')

    def test_pending_original_CPU_proof_prevents_qualifying_preparation(self):
        self.child('''
from unittest import mock
with mock.patch.object(q.known.inputs.cpu,'read',return_value={'state':'pending','qualifying_labels_read':False}),mock.patch.object(q.known.inputs,'sources',side_effect=AssertionError('source scan started')):
    try:q.prepare('scalar',lambda:None)
    except ValueError:pass
    else:raise AssertionError('Pending original data admitted')
''')


if __name__=='__main__':unittest.main()
