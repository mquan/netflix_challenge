"""Synthetic native row/mask/day/query-identity checks; no production data read."""
import unittest
import numpy as np
from upstream_factored_rbm_rows import UserRows


def fixture():
    tr=dict(user_idxs=np.array([0,0,0,2,2,3],dtype=np.int32),
            item_idxs=np.array([1,2,3,0,3,1],dtype=np.int16),ratings=np.array([4,3,99,1,5,99],dtype=np.int8),
            dates=np.array([2,1,9,10,10,4],dtype=np.int16),is_test=np.array([0,0,1,0,0,1],dtype=np.int8),
            user_cnts=np.array([3,0,2,1],dtype=np.int32))
    q=dict(user_idxs=np.array([2,0,3,1,0,2],dtype=np.int32),item_idxs=np.array([1,3,0,2,1,3],dtype=np.int16),
           dates=np.array([10,9,4,6,1,20],dtype=np.int16),user_cnts=np.array([2,1,2,1],dtype=np.int32))
    return tr,q


class RowTests(unittest.TestCase):
    def test_preserves_positive_source_order_and_sorted_training_days(self):
        tr,q=fixture();reader=UserRows(tr,q,4,block_rows=2)
        b=reader.batch(np.array([2,0]),training=True)
        np.testing.assert_array_equal(b['training_rows'],[3,4,0,1])
        np.testing.assert_array_equal(b['support']['items'],[0,3,1,2])
        np.testing.assert_array_equal(b['support']['ratings'],[0,4,3,2])
        np.testing.assert_array_equal(b['support']['days'],[0,0,2,1])
        np.testing.assert_array_equal(b['global_days'],[2,0,1])
        np.testing.assert_array_equal(reader.day_values,[1,2,10])
        np.testing.assert_array_equal(reader.nonempty_users,[0,2])

    def test_conditional_union_includes_masked_and_query_addresses_not_labels(self):
        tr,q=fixture();r=UserRows(tr,q,4);b=r.batch(np.array([0,3,1]))
        expected={0:[1,2,3],1:[0,1],2:[2]}
        for u,items in expected.items():
            np.testing.assert_array_equal(b['support']['conditional_items'][b['support']['conditional_users']==u],items)
        self.assertEqual(len(b['training_rows']),2)
        self.assertNotIn(99,b['support']['ratings'])
        np.testing.assert_array_equal(b['query']['days'],[-1,0,-1,-1])

    def test_query_source_row_ids_reconstruct_exact_original_order(self):
        tr,q=fixture();r=UserRows(tr,q,4);b=r.batch(np.array([3,1,0,2]))
        np.testing.assert_array_equal(b['query_rows'],[2,3,1,4,0,5])
        np.testing.assert_array_equal(b['query']['items'],q['item_idxs'][b['query_rows']])
        reconstructed=np.empty(len(q['user_idxs']),dtype=np.int64)
        reconstructed[b['query_rows']]=b['global_users'][b['query']['users']]
        np.testing.assert_array_equal(reconstructed,q['user_idxs'])

    def test_visible_counts_exclude_masked_and_ignore_query_targets(self):
        tr,q=fixture()
        class NoLabels(dict):
            def __getitem__(self,key):
                if key=='ratings':raise AssertionError('Query outcomes must never be requested')
                return super().__getitem__(key)
        r=UserRows(tr,NoLabels(q),4)
        self.assertEqual(r.visible_counts.sum(),4)
        self.assertEqual(r.visible_counts[1,3],1);self.assertEqual(r.visible_counts[2,2],1)
        self.assertEqual(r.visible_counts[0,0],1);self.assertEqual(r.visible_counts[3,4],1)
        with self.assertRaises(ValueError):r.batch(np.array([3]),training=True)
        with self.assertRaises(ValueError):r.batch(np.array([0,0]))

    def test_reject_raw_row_group_count_mask_and_positive_label_errors(self):
        for key,value in [('user_idxs',np.array([0,2,0,2,2,3])),('is_test',np.array([0,0,2,0,0,1])),
                          ('ratings',np.array([0,3,99,1,5,99]))]:
            tr,q=fixture();tr[key]=value
            with self.assertRaises(ValueError):UserRows(tr,q,4,block_rows=2)
        tr,q=fixture();q['user_cnts']=np.array([1,2,2,1])
        with self.assertRaises(ValueError):UserRows(tr,q,4)

    def test_shuffled_batch_connects_gradients_updates_and_query_prediction(self):
        import torch
        import upstream_factored_rbm_reference as reference
        import upstream_factored_rbm_torch as kernel
        from upstream_factored_rbm_prediction import predict
        from test_upstream_factored_rbm_prediction import scalar
        torch.set_num_threads(1)
        tr,q=fixture();r=UserRows(tr,q,4);b=r.batch(np.array([2,0]),training=True)
        rng=np.random.default_rng(8)
        shapes=dict(p=(4,5,2),q=(2,3),d=(4,3),bv=(4,5),bh=(3,),bu=(2,5),but=(3,5))
        m={k:rng.normal(0,.1,size=s).astype(np.float32) for k,s in shapes.items()}
        hu=rng.random((3,2,3)).astype(np.float32);vu=rng.random((3,4)).astype(np.float32)
        tm={k:torch.tensor(v) for k,v in m.items()}
        support={k:torch.tensor(v,dtype=torch.int64) for k,v in b['support'].items()}
        query={k:torch.tensor(v,dtype=torch.int64) for k,v in b['query'].items()}
        got=kernel.batch_gradients(tm,support,torch.tensor(hu),torch.tensor(vu),tile_rows=1)
        wanted=reference.batch_gradients(m,b['support'],hu,vu,tile_rows=2)
        momentum={k:torch.zeros_like(v) for k,v in tm.items()}
        actual,_=kernel.update(tm,momentum,got)
        expected,_=reference.update(m,{k:np.zeros_like(v) for k,v in m.items()},wanted)
        for key in m:np.testing.assert_allclose(actual[key].numpy(),expected[key],atol=2e-6,rtol=2e-6)
        predictions=predict(actual,support,query,tile_rows=1)['prediction'].numpy()
        independent=scalar({k:v.numpy().astype(np.float64) for k,v in actual.items()},b['support'],b['query'])['prediction']
        np.testing.assert_allclose(predictions,independent,atol=2e-6,rtol=2e-6)
        output=np.full(len(q['user_idxs']),np.nan);output[b['query_rows']]=predictions
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(output)),[0,1,4,5])


if __name__=='__main__':unittest.main()
