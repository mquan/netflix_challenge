from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_joint_context as joint
from e2_corater_data import Coraters, build
from e2_query_codes import masked_query_batch, query_code
from gbdt_blend import HEADER, PROBE
import masked_history as mh
import masked_sequence as ms
from test_selection_objective import fixture


class JointContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.d = fixture()
        self.c = ms.Config(rank=4,hidden=8,d_model=8,heads=2,ff=16,layers=1,
            context='session',time_bins=4,dropout=.1,seq_len=24,max_targets=8,
            pad_width=4,pad_rows=1,device='cpu')
        self.base = ms.SequenceModel(self.d['items'],self.c,ms.weeks(self.d)).eval()

    def model(self, values=True):
        torch.manual_seed(81)
        model=joint.JointContextModel(self.d['items'],self.c,ms.weeks(self.d),use_peer_values=values)
        model.load_rating_state(self.base.state_dict())
        return model

    def test_batched_codes_equal_separate_forwards_with_duplicate_users(self):
        rows=np.array([0,1,5,11,22])
        batch=joint.peer_code_batch(self.d,rows,self.c)
        vectors,error=joint.frozen_peer_vectors(self.base,batch)
        self.assertLessEqual(error,1e-6)
        for k,row in enumerate(rows):
            user=int(np.searchsorted(self.d['train_offsets'],row,side='right')-1)
            item,day=int(self.d['train_items'][row]),int(self.d['train_days'][row])
            single=masked_query_batch(self.d,user,item,day,self.c)
            np.testing.assert_allclose(vectors[k].numpy(),query_code(self.base,single)[1],rtol=0,atol=1e-6)
        # Each duplicate user's row masks only its own target, not every target
        # for that global user elsewhere in the batch.
        changed=dict(self.d,train_ratings=self.d['train_ratings'].copy())
        changed['train_ratings'][0]=5
        other,_=joint.frozen_peer_vectors(self.base,joint.peer_code_batch(changed,rows,self.c))
        torch.testing.assert_close(vectors[0],other[0],rtol=0,atol=1e-6)
        self.assertFalse(torch.equal(vectors[1],other[1]))

    def batch(self):
        batch=ms.prepare_batch(self.d,np.arange(6),self.c,rng=np.random.default_rng(13))
        n=len(batch['target_items']); k=4
        rng=np.random.default_rng(10)
        batch.update(peer_codes=torch.tensor(rng.normal(size=(n,k,self.c.rank)),dtype=torch.float32),
            peer_ratings=torch.tensor(rng.integers(1,6,size=(n,k))),
            peer_gaps=torch.tensor(rng.normal(size=(n,k)),dtype=torch.float32),
            peer_mask=torch.ones(n,k,dtype=torch.bool))
        return batch

    def test_warm_prediction_and_empty_peer_context_preserve_base(self):
        batch=self.batch()
        model=self.model().eval()
        with torch.no_grad():
            expected=self.base.forward_batch(batch)
            self.assertTrue(torch.equal(expected,model.forward_batch(batch)))
            model.peer_output.weight.fill_(.3)
            batch['peer_mask'].zero_()
            value=model.forward_batch(batch)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.equal(expected,value))

    def test_matched_training_learns_peer_values_without_cache_gradients(self):
        batch=self.batch(); batch['peer_codes'].requires_grad_(True)
        models=[self.model(False),self.model(True)]
        states=[]; outputs=[]
        torch.manual_seed(91); start=torch.get_rng_state()
        for model in models:
            torch.set_rng_state(start)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            model.train()
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss=mh.loss_value(model,model.forward_batch(batch),batch['target_ratings'])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(float(model.peer_output.weight.grad.norm()),0)
                optimizer.step()
            states.append(torch.get_rng_state())
            model.eval()
            with torch.no_grad(): outputs.append(model.expected(model.forward_batch(batch)))
        self.assertTrue(torch.equal(states[0],states[1]))
        self.assertIsNone(batch['peer_codes'].grad)
        self.assertGreater(float(models[1].peer_input.weight.grad.norm()),0)
        self.assertFalse(torch.equal(outputs[0],outputs[1]))
        changed=dict(batch,target_ratings=torch.zeros_like(batch['target_ratings']))
        with torch.no_grad():
            self.assertTrue(torch.equal(models[1].forward_batch(batch),models[1].forward_batch(changed)))

    def test_sampler_code_lookup_and_model_integrate_without_target_user(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path=root/'data.bin'
            header=np.zeros(1,dtype=HEADER)
            header['magic']=b'NFPDAT2'
            for key in ('users','items','mean','min_day','max_day'): header[key]=self.d[key]
            header['train_count']=len(self.d['train_items'])
            header['probe_count']=len(self.d['probe'])
            header['implicit_count']=len(self.d['implicit_items'])
            with path.open('wb') as stream:
                header.tofile(stream)
                for key,dtype in [('train_offsets','<u8'),('train_items','<u2'),('train_days','<u2'),
                        ('train_ratings','u1'),('probe',PROBE),('implicit_offsets','<u8'),
                        ('implicit_items','<u2'),('implicit_days','<u2')]:
                    self.d[key].astype(dtype).tofile(stream)
            build(path,root/'index'); index=Coraters(path,root/'index')
            producer_batch=joint.peer_code_batch(self.d,np.arange(len(self.d['train_items'])),self.c)
            vectors,_=joint.frozen_peer_vectors(self.base,producer_batch)
            table=vectors.numpy()
            users=np.arange(6)
            batch=ms.prepare_batch(self.d,users,self.c,rng=np.random.default_rng(13))
            days=[]
            for local,item in zip(batch['target_rows'].numpy(),batch['target_items'].numpy()):
                begin,end=map(int,self.d['train_offsets'][local:local+2])
                source=begin+np.flatnonzero(self.d['train_items'][begin:end]==item+1)[0]
                days.append(self.d['train_days'][source])
            attached=joint.attach_peers(batch,users,days,index,lambda rows:table[rows],seed=31,cap=4)
            self.assertTrue(attached['peer_mask'].any())
            self.assertTrue((~attached['peer_mask']).any())
            model=self.model().eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model.forward_batch(attached),self.base.forward_batch(batch)))
            def missing(rows):
                return np.full((len(rows),self.c.rank),np.nan,dtype=np.float32)
            with self.assertRaisesRegex(ValueError,'missing or invalid'):
                joint.attach_peers(batch,users,days,index,missing,seed=31,cap=4)
            class LeakyIndex:
                def queries(inner,queries,**kwargs):
                    record=index.queries(queries,**kwargs)
                    a,b=np.argwhere(record['mask'])[0]
                    record['users'][a,b]=queries['user'][a]
                    return record
            with self.assertRaisesRegex(ValueError,'query user'):
                joint.attach_peers(batch,users,days,LeakyIndex(),lambda rows:table[rows],seed=31,cap=4)
            with self.assertRaisesRegex(ValueError,'target date'):
                joint.attach_peers(batch,users,np.full(len(days),-1),index,lambda rows:table[rows],seed=31,cap=4)


if __name__ == '__main__':
    unittest.main()
