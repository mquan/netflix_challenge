import contextlib
import dataclasses
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from e2_code_cache import CacheWriter, CodeCache
from e2_corater_data import Coraters,build
from e2_joint_context import JointContextModel,peer_code_batch,frozen_peer_vectors
from e2_query_codes import parameter_hash
import e2_train as train
from gbdt_blend import HEADER,PROBE
import masked_sequence as ms
from test_selection_objective import fixture
from train_holdout import digest


class E2TrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.d=fixture()
        self.c=ms.Config(rank=4,hidden=8,d_model=8,heads=2,ff=16,layers=1,
            context='session',time_bins=4,dropout=.1,batch_users=3,batch_tokens=96,
            seq_len=24,max_targets=8,pad_width=4,pad_rows=2,device='cpu')
        torch.manual_seed(7)
        self.base=ms.SequenceModel(self.d['items'],self.c,ms.weeks(self.d)).eval()
        data=self.root/'data.bin';header=np.zeros(1,dtype=HEADER);header['magic']=b'NFPDAT2'
        for key in ('users','items','mean','min_day','max_day'):header[key]=self.d[key]
        for family in ('train','probe','implicit'):
            header[family+'_count']=len(self.d['probe'] if family=='probe' else self.d[family+'_items'])
        with data.open('wb') as stream:
            header.tofile(stream)
            for key,dtype in [('train_offsets','<u8'),('train_items','<u2'),('train_days','<u2'),
                    ('train_ratings','u1'),('probe',PROBE),('implicit_offsets','<u8'),
                    ('implicit_items','<u2'),('implicit_days','<u2')]:
                self.d[key].astype(dtype).tofile(stream)
        checkpoint=self.root/'warm.pt';report=self.root/'warm.json'
        torch.save(dict(epoch=1,config=dataclasses.asdict(self.c),items=self.d['items'],
            weeks=ms.weeks(self.d),state=self.base.state_dict()),checkpoint)
        report.write_text(json.dumps(dict(selected_epoch=1,config=dataclasses.asdict(self.c),
            selection_sha256=digest(data))))
        cache_plan=self.root/'cache-plan.json'
        cache_plan.write_text(json.dumps(dict(data=str(data),checkpoint=str(checkpoint),
            frozen={str(data):digest(data),str(checkpoint):digest(checkpoint)})))
        binding=dict(plan_sha256=digest(cache_plan),model_sha256=parameter_hash(self.base),
            torch_version=str(torch.__version__))
        cache_path=self.root/'cache'
        writer=CacheWriter(cache_path,binding,len(self.d['train_items']),self.c.rank,100)
        values,_=frozen_peer_vectors(self.base,peer_code_batch(self.d,np.arange(30),self.c))
        writer.append(values.numpy());writer.seal()
        self.lookup=CodeCache(cache_path,binding=binding)
        index_path=self.root/'index';build(data,index_path)
        self.index=Coraters(data,index_path)
        sources=Path(__file__).resolve().parents[1]/'scripts'
        frozen=[data,checkpoint,report,cache_plan,index_path/'manifest.json',cache_path/'manifest.json']
        frozen += [sources/name for name in ('e2_train.py','e2_code_cache.py','e2_corater_data.py',
            'e2_joint_context.py','e2_query_codes.py','masked_history.py','masked_sequence.py',
            'gbdt_blend.py','train_holdout.py')]
        self.plan=dict(data=str(data),warm_checkpoint=str(checkpoint),warm_report=str(report),
            cache=str(cache_path),cache_plan=str(cache_plan),cache_binding=binding,index=str(index_path),
            frozen={str(p):digest(p) for p in frozen},device='cpu',threads=1,epochs=2,lr=.003,
            seed=19,peer_seed=31,peer_cap=4,max_batches=0)
        self.planfile=self.root/'plan.json';self.planfile.write_text(json.dumps(self.plan))

    def test_rating_inputs_and_rng_match_original_preparation(self):
        users=np.array([4,0,3])
        a=np.random.default_rng(13);b=np.random.default_rng(13)
        expected=ms.prepare_batch(self.d,users,self.c,rng=a)
        actual=train.rating_batch(self.d,users,self.c,b,self.index,self.lookup,31,4)
        for key,value in expected.items():
            if torch.is_tensor(value):self.assertTrue(torch.equal(value,actual[key]),key)
            else:self.assertEqual(value,actual[key],key)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        for k,(local,item) in enumerate(zip(actual['target_rows'],actual['target_items'])):
            user=users[int(local)];begin,end=self.d['train_offsets'][user:user+2]
            row=int(begin)+int(np.flatnonzero(self.d['train_items'][begin:end]==int(item)+1)[0])
            peers=self.index.sample(user,int(item)+1,int(self.d['train_days'][row]),seed=31,cap=4)
            np.testing.assert_array_equal(actual['peer_gaps'][k].numpy(),peers['gaps'])

    def test_matched_training_reproducibility_and_label_independent_predictions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            records=[];states=[];predictions=[]
            for arm in ('control','treatment','treatment'):
                output=self.root/f'run{len(records)}'
                records.append(train.run(self.planfile,output,arm))
                states.append(torch.get_rng_state())
                predictions.append(np.fromfile(output/'predictions.f32',dtype='<f4'))
            self.assertTrue(all(r['state']=='complete' for r in records))
            self.assertTrue(torch.equal(states[0],states[1]))
            self.assertFalse(np.array_equal(predictions[0],predictions[1]))
            np.testing.assert_array_equal(predictions[1],predictions[2])
            a=torch.load(self.root/'run1/model.pt',weights_only=True)
            b=torch.load(self.root/'run2/model.pt',weights_only=True)
            for key in a['state']:self.assertTrue(torch.equal(a['state'][key],b['state'][key]),key)
            model=JointContextModel(self.d['items'],self.c,ms.weeks(self.d)).eval()
            model.load_state_dict(a['state'])
            q=self.d['probe'].copy();q['rating']=5
            changed=train.predict(model,self.d,self.index,self.lookup,31,4,queries=q)
            np.testing.assert_array_equal(changed,predictions[1])
            order=np.random.default_rng(7).permutation(len(q))
            reordered=train.predict(model,self.d,self.index,self.lookup,31,4,queries=q[order])
            np.testing.assert_allclose(reordered,changed[order],rtol=0,atol=1e-6)
            with self.assertRaises(FileExistsError):
                train.run(self.planfile,self.root/'run1','treatment')
            self.assertNotIn('validation_rmse',records[1])

    def test_missing_binding_and_changed_cache_fail_before_learning(self):
        plan=dict(self.plan);plan['frozen']=dict(plan['frozen'])
        del plan['frozen'][str(Path(train.__file__).resolve())]
        self.planfile.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'bindings'):
            train.run(self.planfile,self.root/'bad','treatment')
        self.planfile.write_text(json.dumps(self.plan))
        values=np.load(self.root/'cache/vectors.npy',mmap_mode='r+')
        values[0,0]+=1;values.flush()
        with self.assertRaisesRegex(ValueError,'chunk changed'):
            train.run(self.planfile,self.root/'bad','treatment')
        self.assertFalse((self.root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
