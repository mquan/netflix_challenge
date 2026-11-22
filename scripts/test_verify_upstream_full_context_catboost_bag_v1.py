"""Real saved models/pipe/ridge replay; full owner/data admission is mocked."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import verify_upstream_full_context_catboost_bag_v1 as module
import test_upstream_full_context_catboost_bag_v1 as fixtures


class ReplayTests(unittest.TestCase):
    def write(self,path,value):
        path.write_bytes(module.study.wire.encoded(value)+b'\n');return module.study.sha(path)

    @contextmanager
    def fresh(self,f):
        code="import sys\nassert sys.stdin.buffer.read(1)==b'G'\nimport torch\nwith open(sys.argv[1],'rb') as f:sys.stdout.buffer.write(f.read())\nsys.stdout.buffer.flush()\n"
        child=subprocess.Popen([sys.executable,'-c',code,str(f['root']/'payload.bin')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        captured=module.consumer.identity.process(child.pid);assert captured
        child.stdin.write(b'G');child.stdin.flush();child.stdin.close()
        def gate(proof):
            self.assertEqual(proof['admission_sha256'],f['logical_sha']);self.assertEqual(child.wait(timeout=20),0)
            path=f['root']/'replay-source-exit.json'
            digest=self.write(path,dict(state='complete_owned_original_tree_producer_zero_exit',exit_code=0,producer=captured,
                contract_sha256=hashlib.sha256(module.study.wire.encoded(f['contract'])).hexdigest(),source_admission_sha256=f['logical_sha'],
                qualifying_labels_read=False,observed_by_actual_parent_waitpid=True))
            return dict(path=path,sha256=digest,producer=captured)
        try:yield child.stdout,gate
        finally:
            if child.poll() is None:child.stdout.close();child.wait(timeout=20)
            child.stdout.close();child.stderr.close()

    def infer(self,f):
        return module.bag.infer('a'*64,f['query'],f['file_sha'],f['source'],guard=f['guard'],producer_exit_gate=f['gate'])

    def verify(self,f):
        code={'fixture-code.py':module.study.sha(f['root']/'fixture-code.py')}
        path=module.bag.paths()[1]/'completion.json'
        with patch.object(module,'code_bindings',return_value=code),self.fresh(f) as (source,gate):
            return module.verify(module.study.sha(path),source,guard=f['guard'],producer_exit_gate=gate)

    def test_every_actual_model_ridge_system_known_fold_and_bag_bit_replays(self):
        with fixtures.BagTests().fixture() as f:
            self.infer(f);value=self.verify(f)
            self.assertEqual(value['state'],module.STATE);self.assertEqual(value['ridge_equations'],10)
            self.assertEqual(len(value['ridge_residuals']),10);self.assertEqual(value['actual_saved_models_reopened'],5)
            self.assertTrue(value['every_known_and_all_five_qualifying_fold_and_bag_bit_verified'])
            self.assertTrue(value['every_original_full_raw_stream_and_baseline_bit_replayed'])
            self.assertTrue(value['independent_arithmetic_does_not_call_production_predict'])
            self.assertFalse(value['production_data_admitted']);self.assertFalse(value['qualifying_labels_read'])
            with self.assertRaisesRegex(ValueError,'retry'):self.verify(f)

    def mutate(self,f,name):
        report=module.bag.paths()[1];path=report/'completion.json';value=json.loads(path.read_text());row=value['outputs'][name]
        payload=Path(row['path']);data=np.fromfile(payload,dtype='<f4');data[0]=np.nextafter(data[0],np.float32(np.inf));payload.write_bytes(data.tobytes())
        row['sha256']=module.study.sha(payload);self.write(path,value)

    def test_rehashed_one_ULP_fold_prediction_change_rejected(self):
        with fixtures.BagTests().fixture() as f:
            self.infer(f);self.mutate(f,'fold-2')
            with self.assertRaisesRegex(ValueError,'fold prediction bit'):self.verify(f)
            self.assertFalse((module.bag.paths()[1]/'independent-verification.json').exists())

    def test_rehashed_one_ULP_bag_prediction_change_rejected(self):
        with fixtures.BagTests().fixture() as f:
            self.infer(f);self.mutate(f,'qualifying')
            with self.assertRaisesRegex(ValueError,'bag prediction bit'):self.verify(f)

    def test_rehashed_one_ULP_known_pair_change_rejected(self):
        with fixtures.BagTests().fixture() as f:
            self.infer(f);self.mutate(f,'known')
            with self.assertRaisesRegex(ValueError,'known tree prediction bit'):self.verify(f)


if __name__=='__main__':unittest.main()
