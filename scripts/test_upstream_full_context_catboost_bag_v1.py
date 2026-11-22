"""Actual saved CPU models and source pipe; admission/row proofs synthetic."""
from contextlib import contextmanager,ExitStack
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_full_context_catboost_bag_v1 as module
import upstream_full_context_catboost_artifacts as artifacts


class BagTests(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(module.study.wire.encoded(value)+b'\n');return module.study.sha(path)

    @contextmanager
    def fixture(self,*,exit_code=0,truncate=False):
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary).resolve();study=module.study;guard=lambda **kw:None
            stack.enter_context(patch.object(study,'ROOT',root));stack.enter_context(patch.object(study,'ROWS',20));stack.enter_context(patch.object(study,'QUALIFYING_ROWS',23))
            code=root/'fixture-code.py';code.write_text('# synthetic admission/code fixture')
            stack.enter_context(patch.object(module,'code_bindings',return_value={code.name:study.sha(code)}))
            rng=np.random.default_rng(498);train=rng.normal(size=(64,713)).astype('<f4');x=rng.normal(size=(23,713)).astype('<f4');x[:,-1]=0
            baselines=np.empty((5,2,23),dtype='<f8');folds=[];bindings={code.name:study.sha(code)}
            z=rng.normal(0,.1,(20,562));z[:,-1]=1;zq=rng.normal(0,.1,(23,562));zq[:,-1]=1
            groups=np.arange(20)%10;known_y=(np.arange(20)%5+1).astype('<f8')
            gram=np.stack([z[groups==g].T@z[groups==g] for g in range(10)])
            rhs=np.stack([z[groups==g].T@known_y[groups==g] for g in range(10)])
            weights=np.empty((5,2,562),dtype='<f8');regularizer=np.eye(562)*1000.;regularizer[-1,-1]=0.
            for fold in range(5):
                for population in range(2):
                    mask=np.asarray([g//2!=fold and (population==1 or g%2==0) for g in range(10)])
                    weights[fold,population]=np.linalg.solve(gram[mask].sum(0)+regularizer,rhs[mask].sum(0))
                    baselines[fold,population]=np.clip(zq@weights[fold,population],1.05,4.95)
            for name,values in [('gram.npy',gram),('rhs.npy',rhs)]:
                path=root/module.consumer.ARRAYS/name;path.parent.mkdir(parents=True,exist_ok=True);np.save(path,values)
                bindings[str(path.relative_to(root))]=study.sha(path)
            for fold in range(5):
                train[:,-1]=2.1+fold*.2;labels=(.2*train[:,0]+.1*train[:,2]).astype('<f8')
                model=module.prediction.backend.train(train,labels,2026+fold,3,guard)
                record=artifacts.save_model(root/('tree-%s.cbm'%fold),model,seed=2026+fold,iterations=3,width=713,root=root,guard=guard)
                bindings[str(Path(record['path']).relative_to(root))]=record['sha256']
                folds.append(dict(fold=fold,seed_order=[2026,7,13],seed_offset=fold,chosen_iterations=[3,0,0],models={'2026':record},qualifying_labels_read=False))
            run=root/'known-art';run.mkdir();np.save(run/'prediction.npy',np.full(20,3.,dtype='<f4'));bindings['known-art/prediction.npy']=study.sha(run/'prediction.npy')
            stack.enter_context(patch.object(study,'paths',return_value=(run,root/'known-report')))
            np.save(run/'weights.npy',weights);bindings['known-art/weights.npy']=study.sha(run/'weights.npy')
            known=dict(arm='scalar',arms={'scalar':dict(folds=folds,manifest={'dimensions':{'nn_inputs':713,'design_columns':562},
                'outputs':{'weights':dict(path=str(run/'weights.npy'),sha256=study.sha(run/'weights.npy'))}})},source_sha256=bindings)
            stack.enter_context(patch.object(module.selected,'load',return_value=known))
            q=dict(synthetic_full_stream_query=True,weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),source_sha256=dict(bindings))
            qp=root/'query-metadata.json';file_sha=self.write(qp,q);logical_sha=hashlib.sha256(study.wire.encoded(q)).hexdigest()
            contract=study.wire.contract(23,retain_scalar=False,latent=False,weights_sha256=q['weights_sha256'],admission_sha256=logical_sha,
                operator_sha256=[hashlib.sha256(zq.tobytes()).hexdigest(),hashlib.sha256(x[:,:-1].tobytes()).hexdigest()])
            stack.enter_context(patch.object(module.selected,'qualifying_metadata',return_value=(q,contract)))
            packet=io.BytesIO();wire=study.wire;wire.write_all(packet,wire.MAGIC,guard);wire.write_metadata(packet,contract,guard)
            wire.write_all(packet,x,guard);wire.write_all(packet,baselines,guard)
            wire.write_metadata(packet,dict(raw_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                baseline_sha256=[hashlib.sha256(baselines[f,p].tobytes()).hexdigest() for f in range(5) for p in range(2)],
                operator_sha256=contract['operator_sha256'],admission_sha256=logical_sha),guard)
            payload=root/'payload.bin';payload.write_bytes(packet.getvalue()[:-1] if truncate else packet.getvalue())
            body="import sys\nassert sys.stdin.buffer.read(1)==b'G'\nimport torch\nwith open(sys.argv[1],'rb') as f:sys.stdout.buffer.write(f.read())\nsys.stdout.buffer.flush()\nsys.exit(int(sys.argv[2]))\n"
            child=subprocess.Popen([sys.executable,'-c',body,str(payload),str(exit_code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            captured=module.consumer.identity.process(child.pid);assert captured
            child.stdin.write(b'G');child.stdin.flush();child.stdin.close()
            def gate(proof):
                assert proof['admission_sha256']==logical_sha and logical_sha!=file_sha
                result=child.wait(timeout=20)
                if result!=0:raise ValueError('Actual original fixture source producer failed')
                path=root/'source-exit.json';digest=self.write(path,dict(state='complete_owned_original_tree_producer_zero_exit',exit_code=0,
                    producer=captured,contract_sha256=hashlib.sha256(wire.encoded(contract)).hexdigest(),
                    source_admission_sha256=logical_sha,qualifying_labels_read=False,observed_by_actual_parent_waitpid=True))
                return dict(path=path,sha256=digest,producer=captured)
            try:
                yield dict(root=root,x=x,baselines=baselines,folds=folds,known=known,query=qp,file_sha=file_sha,logical_sha=logical_sha,
                    source=child.stdout,gate=gate,guard=guard,contract=contract,metadata=q)
            finally:
                if child.poll() is None:
                    child.stdout.close();child.wait(timeout=20)
                child.stdout.close();child.stderr.close()

    def infer(self,f):
        return module.infer('a'*64,f['query'],f['file_sha'],f['source'],guard=f['guard'],producer_exit_gate=f['gate'])

    def test_real_five_model_pipe_fold_bag_and_matched_known_bits_after_actual_exit(self):
        with self.fixture() as f:
            result=self.infer(f);self.assertEqual(result['state'],module.STATE);self.assertEqual(result['original_seed_slots'],15)
            self.assertEqual(result['qualifying_rows'],23);self.assertEqual(result['known_rows'],20)
            self.assertFalse(result['production_data_admitted']);self.assertFalse(result['qualifying_labels_read'])
            self.assertEqual(result['query_metadata_sha256'],f['file_sha']);self.assertEqual(result['query_source_admission_sha256'],f['logical_sha'])
            total=np.zeros(23,dtype='<f8')
            for fold in range(5):
                record=f['folds'][fold]['models']['2026'];matrix=f['x'].copy();matrix[:,-1]=f['baselines'][fold,1].astype('<f4')
                with module.prediction.backend.actual_model(record['path'],record,record['signature'],713,f['guard']) as model:
                    values=model.predict(matrix,thread_count=2)
                expected=np.clip(f['baselines'][fold,1]+values/3,1.05,4.95).astype('<f4');total+=expected.astype('<f8')
                self.assertEqual(Path(result['outputs']['fold-'+str(fold)]['path']).read_bytes(),expected.tobytes())
            self.assertEqual(Path(result['outputs']['qualifying']['path']).read_bytes(),np.clip(total/5,1.05,4.95).astype('<f4').tobytes())
            self.assertEqual(Path(result['outputs']['known']['path']).read_bytes(),np.full(20,3.,dtype='<f4').tobytes())
            with self.assertRaisesRegex(ValueError,'retry'):self.infer(f)
            self.assertNotIn('torch',sys.modules)

    def test_actual_failed_producer_never_publishes_any_prediction_or_completion(self):
        with self.fixture(exit_code=7) as f:
            with self.assertRaisesRegex(ValueError,'producer failed'):self.infer(f)
            art,report=module.paths();self.assertEqual(list(art.iterdir()),[])
            self.assertFalse((report/'completion.json').exists());self.assertTrue((report/'started.json').exists())

    def test_truncated_complete_row_stream_never_publishes_predictions(self):
        with self.fixture(truncate=True) as f:
            with self.assertRaises(ValueError):self.infer(f)
            art,report=module.paths();self.assertEqual(list(art.iterdir()),[]);self.assertFalse((report/'completion.json').exists())


if __name__=='__main__':unittest.main()
