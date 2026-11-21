"""Independent full-array semantic checks on tiny synthetic fixtures only."""
import unittest
import numpy as np
from test_upstream_factored_rbm_rows import fixture
from upstream_factored_rbm_rows import UserRows
from upstream_factored_rbm_data_audit import audit_reader,audit_known_projection,audit_zero_targets


class BrokenReader:
    def __init__(self,reader,change):self.reader,self.change=reader,change
    def __getattr__(self,key):return getattr(self.reader,key)
    def batch(self,ids,training=False):
        value=self.reader.batch(ids,training=training);self.change(value);return value


def projection_fixture():
    tr,q=fixture();q=dict(q);q['ratings']=np.array([5,4,2,1,3,3],dtype=np.int8)
    mapping=np.array([1,4,0],dtype=np.int64)
    known={k:q[k][mapping].copy() for k in ('user_idxs','item_idxs','dates','ratings')}
    return tr,q,known,mapping


class DataAuditTests(unittest.TestCase):
    def test_all_rows_conditionals_days_and_queries_replay_with_different_batch_sizes(self):
        tr,q=fixture();r=UserRows(tr,q,4);results=[]
        for size in (1,2,500):
            calls=[];p=audit_reader(r,tr,q,lambda:calls.append(True),4,users_per_batch=size,block_rows=2)
            self.assertEqual((p['raw_training_rows'],p['positive_ratings'],p['query_rows'],p['conditional_pairs'],p['training_day_slots'],p['nonempty_users']),(6,4,6,9,3,2))
            self.assertGreater(len(calls),5);results.append(p['stream_sha256'])
        self.assertEqual(results[0],results[1]);self.assertEqual(results[1],results[2])

    def test_rejects_corrupted_producer_positive_row_query_and_conditional_outputs(self):
        tr,q=fixture();r=UserRows(tr,q,4)
        def drop_masked_address(b):
            s=b['support'];keep=~((s['conditional_users']==3)&(s['conditional_items']==1))
            s['conditional_users']=s['conditional_users'][keep];s['conditional_items']=s['conditional_items'][keep]
        for change in (lambda b:b['training_rows'].__setitem__(0,1),lambda b:b['query_rows'].__setitem__(0,0),drop_masked_address):
            with self.assertRaises(ValueError):audit_reader(BrokenReader(r,change),tr,q,lambda:None,4)

    def test_rejects_wrong_global_day_and_visible_count_metadata(self):
        tr,q=fixture();r=UserRows(tr,q,4)
        with self.assertRaises(ValueError):audit_reader(r,tr,q,lambda:None,5)
        for field in ('day_values','visible_counts'):
            tr,q=fixture();r=UserRows(tr,q,4)
            getattr(r,field).flat[0]+=1
            with self.assertRaises(ValueError):audit_reader(r,tr,q,lambda:None,4)

    def test_known_map_labels_and_positive_disjointness_allow_masked_addresses(self):
        tr,q,k,m=projection_fixture();p=audit_known_projection(tr,q,k,m,4,4,lambda:None,block_rows=2)
        self.assertEqual((p['known_rows'],p['source_query_rows'],p['positive_training_rows']),(3,6,4))
        self.assertTrue(p['known_labels_equal']);self.assertTrue(p['known_identities_absent_from_positive_training'])
        self.assertTrue(p['native_holdout_identities_absent_from_positive_training'])

    def test_unmapped_native_holdout_leak_is_also_rejected(self):
        tr,q,k,m=projection_fixture();q['dates'][5]=10
        self.assertNotIn(5,m)
        with self.assertRaisesRegex(ValueError,'Native holdout.*positive training'):
            audit_known_projection(tr,q,k,m,4,4,lambda:None)

    def test_rejects_training_leak_wrong_map_label_and_duplicate_query(self):
        tr,q,k,m=projection_fixture();q['item_idxs'][0]=0;k={name:q[name][m].copy() for name in k}
        with self.assertRaisesRegex(ValueError,'positive training'):audit_known_projection(tr,q,k,m,4,4,lambda:None)
        tr,q,k,m=projection_fixture()
        for wrong in (np.array([0,4,1],dtype=np.int64),m.astype(np.int32)):
            with self.assertRaises(ValueError):audit_known_projection(tr,q,k,wrong,4,4,lambda:None)
        k['ratings'][0]=1
        with self.assertRaises(ValueError):audit_known_projection(tr,q,k,m,4,4,lambda:None)
        tr,q,k,m=projection_fixture()
        for name in ('user_idxs','item_idxs','dates'):q[name][5]=q[name][0]
        with self.assertRaisesRegex(ValueError,'Unique'):audit_known_projection(tr,q,k,m,4,4,lambda:None)

    def test_zero_qualifying_targets_guard_and_schema(self):
        zero=np.zeros(5,dtype=np.int8);calls=[]
        self.assertTrue(audit_zero_targets(zero,5,lambda:calls.append(True),block_rows=2)['zero_targets'])
        self.assertEqual(len(calls),4)
        for value in (np.ones(5,dtype=np.int8),zero.astype(np.float32),zero[:4]):
            with self.assertRaises(ValueError):audit_zero_targets(value,5,lambda:None)


if __name__=='__main__':unittest.main()
