import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gbdt_blend import HEADER,PROBE,blend_fold
from p2_candidate_gate import SOURCES,score
from selection_objective_gate import compare
from train_holdout import digest


class CandidateGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);rng=np.random.default_rng(17)
        self.users=np.repeat(np.arange(200,dtype=np.uint32),4)
        self.y=rng.integers(1,6,len(self.users)).astype(float)
        self.bank=np.column_stack([self.y+rng.normal(0,1.1,len(self.y)) for _ in range(2)])
        self.control=self.y+rng.normal(0,.8,len(self.y))
        self.treatment=self.y+rng.normal(0,.3,len(self.y))
        self.data=self.root/'data.bin';self.write_data(self.y)
        reference=self.root/'protocol.txt';reference.write_text('Fixed fixture protocol before scoring; gain >=0.0003.')
        self.paths=[]
        for k in range(2):
            p=self.root/f'bank{k}.f32';self.bank[:,k].astype('<f4').tofile(p);self.paths.append(str(p))
        sources=Path(__file__).resolve().parents[1]/'scripts'
        frozen=[self.data,reference,*map(Path,self.paths)]+[sources/name for name in SOURCES]
        self.plan=dict(experiment='fixture',protocol='SO1-P2-fold0-matched-pair-v1',minimum_gain=.0003,
            protocol_reference=str(reference),data=str(self.data),bank=self.paths,
            frozen={str(p):digest(p) for p in frozen},arms={})
        for name,values in [('control',self.control),('treatment',self.treatment)]:
            p=self.root/f'{name}.f32';values.astype('<f4').tofile(p)
            r=self.root/f'{name}.json'
            r.write_text(json.dumps(dict(state='complete',smoke_only=False,experiment='fixture',arm=name,
                data_sha256=digest(self.data),prediction=str(p),prediction_sha256=digest(p))))
            self.plan['arms'][name]=dict(prediction=str(p),receipt=str(r))
        self.planfile=self.root/'plan.json';self.save_plan()

    def save_plan(self):
        self.planfile.write_text(json.dumps(self.plan))

    def write_data(self,y):
        h=np.zeros(1,dtype=HEADER);h['magic']=b'NFPDAT2';h['users']=200;h['items']=4
        h['mean']=3;h['probe_count']=len(y);h['max_day']=4
        p=np.zeros(len(y),dtype=PROBE);p['user']=self.users;p['item']=1;p['day']=3;p['rating']=y
        with self.data.open('wb') as stream:
            h.tofile(stream);np.zeros(201,dtype='<u8').tofile(stream);p.tofile(stream)
            np.zeros(201,dtype='<u8').tofile(stream)

    def test_matches_existing_comparator_and_rejects_overwrite(self):
        result,pred=score(self.planfile,self.root/'score.json')
        bank=np.column_stack([np.fromfile(p,dtype='<f4').astype(float) for p in self.paths])
        arms={k:np.fromfile(v['prediction'],dtype='<f4').astype(float) for k,v in self.plan['arms'].items()}
        expected,raw=compare(bank,arms['control'],arms['treatment'],self.y,self.users)
        for key in ('primary','replacement_diagnostic','selected_penalties'):
            self.assertEqual(result[key],expected[key])
        for key in pred:np.testing.assert_array_equal(pred[key],raw[key])
        self.assertTrue(result['passes_p2_gate'])
        self.assertFalse(result['primary']['all_folds_improve']) # Scope is only fold 0.
        self.assertIn('P2 user fold 0 only',result['scope'])
        with self.assertRaises(FileExistsError):score(self.planfile,self.root/'score.json')

    def test_held_labels_cannot_change_fit_or_selected_penalties(self):
        first,pred=score(self.planfile,self.root/'first.json')
        changed=self.y.copy();held=blend_fold(self.users)==0;changed[held]=(changed[held]%5)+1
        self.write_data(changed);self.plan['frozen'][str(self.data)]=digest(self.data)
        for arm in self.plan['arms'].values():
            p=Path(arm['receipt']);r=json.loads(p.read_text());r['data_sha256']=digest(self.data)
            p.write_text(json.dumps(r))
        self.save_plan();second,other=score(self.planfile,self.root/'second.json')
        self.assertEqual(first['selected_penalties'],second['selected_penalties'])
        for key in pred:np.testing.assert_array_equal(pred[key],other[key])
        self.assertNotEqual(first['primary']['control_rmse'],second['primary']['control_rmse'])

    def test_bad_completion_and_changed_payload_fail_before_output(self):
        arm=self.plan['arms']['treatment'];p=Path(arm['receipt']);r=json.loads(p.read_text())
        for key,value in [('state','running'),('smoke_only',True),('experiment','other'),('arm','control')]:
            bad=dict(r);bad[key]=value;p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'receipt'):
                score(self.planfile,self.root/'bad.json')
        p.write_text(json.dumps(r))
        np.zeros(len(self.y),dtype='<f4').tofile(arm['prediction'])
        with self.assertRaisesRegex(ValueError,'prediction differs'):
            score(self.planfile,self.root/'bad.json')
        self.assertFalse((self.root/'bad.json').exists())
        Path(self.paths[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'frozen scoring input changed'):
            score(self.planfile,self.root/'bad.json')


if __name__ == '__main__':
    unittest.main()
