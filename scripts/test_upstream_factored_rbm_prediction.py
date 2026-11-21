"""Prediction replay against separate scalar user/item/factor equations, CPU only."""
import unittest
import numpy as np
from test_upstream_factored_rbm_reference import fixture
from upstream_factored_rbm_prediction import predict


def scalar(model, support, query):
    m,b = model,support; hidden=[]; scores=[]; probabilities=[]
    nf,nh=m['q'].shape
    for user in range(len(m['bu'])):
        a=m['bh'].copy();rows=np.flatnonzero(b['users']==user)
        conditional=b['conditional_items'][b['conditional_users']==user]
        for j in range(nh):
            for f in range(nf):
                total=sum(m['p'][b['items'][r],b['ratings'][r],f] for r in rows)
                a[j]+=total*m['q'][f,j]
            for item in conditional:a[j]+=m['d'][item,j]
        hidden.append(2/(1+np.exp(-a))-1)
    for item,user,day in zip(query['items'],query['users'],query['days']):
        logits=m['bv'][item]+m['bu'][user]
        if day>=0:logits=logits+m['but'][day]
        for k in range(5):
            for f in range(nf):
                logits[k]+=m['p'][item,k,f]*sum(m['q'][f,j]*hidden[user][j] for j in range(nh))
        ex=np.exp(logits-max(logits));prob=ex/sum(ex)
        probabilities.append(prob);scores.append(sum((k+1)*prob[k] for k in range(5)))
    return dict(prediction=np.array(scores),probabilities=np.asarray(probabilities).reshape(-1,5),hidden_expected=np.array(hidden))


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def inputs(self,dtype):
        t=self.torch;m,b,_,_=fixture()
        m['bu']=np.vstack((m['bu'],np.array([[.1,-.1,.2,-.2,.05]])))
        b=dict(b);b['conditional_items']=np.r_[b['conditional_items'],2]
        b['conditional_users']=np.r_[b['conditional_users'],3]
        q=dict(items=np.array([3,0,2,1,1]),users=np.array([0,1,2,3,3]),
               days=np.array([1,-1,4,-1,-1]),day_owner=b['day_owner'])
        model={k:t.tensor(v,dtype=dtype) for k,v in m.items()}
        support={k:t.tensor(v,dtype=t.int64) for k,v in b.items()}
        query={k:t.tensor(v,dtype=t.int64) for k,v in q.items()}
        return model,support,query,b,q

    def test_complete_prediction_probabilities_and_cold_user_scalar_replay(self):
        t=self.torch
        for dtype,tolerance in ((t.float64,3e-13),(t.float32,2e-6)):
            m,s,q,b,rawq=self.inputs(dtype)
            rm={k:v.numpy().astype(np.float64) for k,v in m.items()}
            wanted=scalar(rm,b,rawq)
            for tile in (1,2,16384):
                got=predict(m,s,q,tile)
                for k,value in wanted.items():np.testing.assert_allclose(got[k].numpy(),value,atol=tolerance,rtol=tolerance)
                np.testing.assert_allclose(got['probabilities'].sum(dim=1).numpy(),1,atol=tolerance,rtol=tolerance)
                self.assertTrue(bool(t.all((got['prediction']>=1)&(got['prediction']<=5))))
                self.assertFalse(np.allclose(got['hidden_expected'][3].numpy(),0))

    def test_missing_day_never_indexes_last_training_day(self):
        t=self.torch;m,s,q,_,_=self.inputs(t.float64)
        before=predict(m,s,q)['prediction'].numpy()
        m['but'][-1]=t.tensor([30.,0.,0.,0.,-30.],dtype=t.float64)
        after=predict(m,s,q)['prediction'].numpy()
        np.testing.assert_array_equal(before[[1,3,4]],after[[1,3,4]])
        self.assertGreater(abs(before[2]-after[2]),.1)

    def test_queries_retain_input_order_duplicates_and_empty_queries(self):
        t=self.torch;m,s,q,_,_=self.inputs(t.float64)
        base=predict(m,s,q)['prediction'];order=t.tensor([4,0,2,4,1,3])
        perm={k:v[order] if k!='day_owner' else v for k,v in q.items()}
        np.testing.assert_array_equal(predict(m,s,perm,tile_rows=1)['prediction'].numpy(),base[order].numpy())
        empty={k:v[:0] if k!='day_owner' else v for k,v in q.items()}
        out=predict(m,s,empty)
        self.assertEqual(tuple(out['prediction'].shape),(0,));self.assertEqual(tuple(out['probabilities'].shape),(0,5))

    def test_reject_cross_user_day_and_invalid_missing_day(self):
        t=self.torch;m,s,q,_,_=self.inputs(t.float64)
        wrong={k:v.clone() for k,v in q.items()};wrong['days'][0]=2
        with self.assertRaises(ValueError):predict(m,s,wrong)
        wrong['days'][0]=-2
        with self.assertRaises(ValueError):predict(m,s,wrong)

    def test_no_training_rows_and_no_day_slots_still_use_conditional_and_hidden_bias(self):
        t=self.torch;m,s,q,_,_=self.inputs(t.float64)
        for k in ('items','users','ratings'):s[k]=s[k][:0]
        m['but']=m['but'][:0];q['day_owner']=q['day_owner'][:0];q['days']=t.full_like(q['days'],-1)
        out=predict(m,s,q)
        expected=2*t.sigmoid(m['bh']+m['d'][2])-1
        np.testing.assert_allclose(out['hidden_expected'][3].numpy(),expected.numpy(),atol=1e-14,rtol=1e-14)


if __name__=='__main__':unittest.main()
