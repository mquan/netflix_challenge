"""Synthetic CPU training-state and saved-model boundary tests; no actual data."""
import copy
import io
import unittest
import numpy as np
from test_upstream_factored_rbm_rows import fixture
from upstream_factored_rbm_rows import UserRows
from upstream_factored_rbm_training import State,Config,SHARED,LOCAL,cd_steps,user_order

DIGEST='1'*64  # Synthetic fixture identifier, not production source admission.


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def setup_state(self):
        tr,q=fixture();reader=UserRows(tr,q,5)
        return reader,State(reader,DIGEST,Config(hidden=3,factors=2,epochs=2,batch_users=1,tile_rows=1))

    def assert_state_equal(self,a,b):
        for group in ('shared','velocity','bias','bias_velocity'):
            for key in getattr(a,group):
                self.assertTrue(self.torch.equal(getattr(a,group)[key],getattr(b,group)[key]),(group,key))
        self.assertTrue(self.torch.equal(a.generator.get_state(),b.generator.get_state()))
        self.assertEqual(a.completed_epoch,b.completed_epoch)

    def test_source_scope_initialization_counts_global_fallback_and_seed(self):
        r,a=self.setup_state();b=State(r,DIGEST,a.config)
        self.assert_state_equal(a,b)
        counts=r.visible_counts;global_counts=counts.sum(axis=0,dtype=np.int64).astype(np.float32)
        eps=np.float32(1e-6)
        expected=np.log((global_counts+eps)/(np.float32(global_counts.sum())+np.float32(5)*eps))
        np.testing.assert_array_equal(a.shared['bv'][4].numpy(),expected)
        self.assertTrue(bool(self.torch.isfinite(a.shared['bv']).all()))
        for v in list(a.bias.values())+list(a.velocity.values()):self.assertEqual(self.torch.count_nonzero(v),0)

    def test_epoch_visits_all_nonempty_users_and_preserves_cold_bias_slots(self):
        r,s=self.setup_state();visits=[];calls=[]
        s.fit_epoch(1,lambda:calls.append(True),lambda e,b,ids:visits.extend(ids.tolist()))
        np.testing.assert_array_equal(visits,user_order(r.nonempty_users,s.config.seed,1))
        self.assertEqual(set(visits),{0,2});self.assertEqual(s.completed_epoch,1)
        self.assertEqual(s.completed_batches,2);self.assertIsNone(s.active_epoch)
        self.assertGreaterEqual(len(calls),6)
        np.testing.assert_array_equal(s.bias['bu'][[1,3]].numpy(),0)

    def test_complete_epoch_serialization_restores_rng_momentum_and_next_epoch_exactly(self):
        t=self.torch;r,s=self.setup_state();s.fit_epoch(1,lambda:None)
        data=io.BytesIO();t.save(s.snapshot(),data);data.seek(0)
        saved=t.load(data,map_location='cpu',weights_only=True)
        restored=State.from_snapshot(saved,r,DIGEST)
        self.assert_state_equal(s,restored)
        s.fit_epoch(2,lambda:None);restored.fit_epoch(2,lambda:None)
        self.assert_state_equal(s,restored)
        np.testing.assert_array_equal(s.predict_queries(lambda:None),restored.predict_queries(lambda:None))

    def test_full_query_prediction_matches_independent_scalar_model_in_original_order(self):
        from test_upstream_factored_rbm_prediction import scalar
        r,s=self.setup_state();s.fit_epoch(1,lambda:None)
        b=r.batch(np.arange(r.n_users,dtype=np.int64))
        m={k:v.numpy().astype(np.float64) for k,v in dict(s.shared,**s.bias).items()}
        want=scalar(m,b['support'],b['query'])['prediction']
        ordered=np.empty(len(want));ordered[b['query_rows']]=want
        np.testing.assert_allclose(s.predict_queries(lambda:None),ordered,atol=2e-6,rtol=2e-6)

    def test_failed_partial_epoch_cannot_be_restarted_or_checkpointed(self):
        _,s=self.setup_state();count=0
        def guard():
            nonlocal count
            count+=1
            if count==3:raise RuntimeError('synthetic fatal guard')
        with self.assertRaises(RuntimeError):s.fit_epoch(1,guard)
        self.assertEqual(s.active_epoch,1);self.assertEqual(s.completed_batches,1);self.assertEqual(s.completed_epoch,0)
        with self.assertRaises(ValueError):s.fit_epoch(1,lambda:None)
        with self.assertRaises(ValueError):s.snapshot()
        with self.assertRaises(ValueError):s.predict_queries(lambda:None)

    def test_snapshot_rejects_wrong_input_partial_counters_missing_or_invalid_tensors(self):
        t=self.torch;r,s=self.setup_state();s.fit_epoch(1,lambda:None);good=s.snapshot()
        with self.assertRaises(ValueError):State.from_snapshot(good,r,'2'*64)
        for modify in (
            lambda x:x.update(active_epoch=1),lambda x:x.update(completed_batches=0),
            lambda x:x['shared'].pop('q'),lambda x:x['bias'].update(bu=x['bias']['bu'].double()),
            lambda x:x['shared']['p'].fill_(float('nan')),
        ):
            bad=copy.deepcopy(good);modify(bad)
            with self.assertRaises(ValueError):State.from_snapshot(bad,r,DIGEST)
        with self.assertRaises(ValueError):s.fit_epoch(1,lambda:None)
        with self.assertRaises(ValueError):s.fit_epoch(3,lambda:None)

    def test_full_default_recipe_cd_schedule_and_shuffle_are_explicit(self):
        c=Config();c.validate()
        self.assertEqual((c.hidden,c.factors,c.epochs,c.batch_users,c.seed),(400,50,15,500,128))
        self.assertEqual([cd_steps(e) for e in (1,5,6,10,11,15,16,21)],[1,1,2,2,3,3,4,5])
        users=np.arange(100);before=users.copy()
        np.testing.assert_array_equal(user_order(users,128,1),user_order(users,128,1))
        np.testing.assert_array_equal(np.sort(user_order(users,128,1)),users)
        self.assertFalse(np.array_equal(user_order(users,128,1),user_order(users,128,2)))
        np.testing.assert_array_equal(users,before)
        for config in (Config(epochs=0),Config(seed=2**64),Config(momentum=1),Config(lr=-.1)):
            with self.assertRaises(ValueError):config.validate()
        with self.assertRaises(ValueError):cd_steps(0)


if __name__=='__main__':unittest.main()
