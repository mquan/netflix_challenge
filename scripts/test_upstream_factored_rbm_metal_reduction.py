"""CPU/address parity only. These tests never qualify actual MPS operation."""
import unittest
import numpy as np
import upstream_factored_rbm_metal_reduction as reduction
import upstream_factored_rbm_preflight as old_preflight
import upstream_factored_rbm_torch as old_kernel
import upstream_factored_rbm_torch_metal as new_kernel
from upstream_factored_rbm_prediction import predict as old_predict
from upstream_factored_rbm_prediction_metal import predict as new_predict


class ReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(2)

    def test_stable_address_groups_preserve_input_order_and_empty_rows(self):
        ids=np.array([3,1,3,0,1,3],dtype=np.int64)
        order,offsets=reduction.addresses(ids,5)
        self.assertEqual(order.tolist(),[3,1,4,0,2,5])
        self.assertEqual(offsets.tolist(),[0,1,3,3,6,6])
        self.assertEqual(order.dtype,np.int32)

    def test_invalid_addresses_reject(self):
        for ids,rows in [(np.array([-1]),3),(np.array([3]),3),(np.array([1.]),3),(np.zeros((2,2),dtype=int),3)]:
            with self.assertRaises(ValueError):reduction.addresses(ids,rows)

    def test_empty_address_groups(self):
        order,offsets=reduction.addresses(np.array([],dtype=np.int64),3)
        self.assertEqual(order.tolist(),[]);self.assertEqual(offsets.tolist(),[0,0,0,0])

    def test_cpu_existing_destination_order_matches_sequential_float32_reference(self):
        t=self.torch
        source=np.array([[1e8,2.],[1.,3.],[-1e8,4.],[7.,5.]],dtype=np.float32)
        ids=np.array([0,0,0,1]);initial=np.array([[1.,-1.],[2.,3.],[4.,5.]],dtype=np.float32)
        expected=initial.copy()
        for i,value in zip(ids,source):expected[i]+=value
        got=t.tensor(initial);reduction.add_rows_(got,t.tensor(ids),t.tensor(source))
        np.testing.assert_array_equal(got.numpy(),expected)

    def test_cpu_all_gradient_and_update_bits_equal_original_kernel(self):
        t=self.torch;model,batch,hu,vu=old_preflight.fixture()
        m={k:t.tensor(v) for k,v in model.items()};b={k:t.tensor(v,dtype=t.int64) for k,v in batch.items()}
        old=old_kernel.batch_gradients(m,b,t.tensor(hu),t.tensor(vu),2)
        new=new_kernel.batch_gradients(m,b,t.tensor(hu),t.tensor(vu),2)
        for k in old:
            if k=='batch_users':self.assertEqual(new[k],old[k])
            else:self.assertTrue(t.equal(new[k],old[k]),k)
        velocities={k:t.zeros_like(v) for k,v in m.items()}
        a,aa=old_kernel.update(m,velocities,old);z,zz=new_kernel.update(m,velocities,new)
        for k in a:self.assertTrue(t.equal(a[k],z[k]) and t.equal(aa[k],zz[k]),k)

    def test_cpu_prediction_bits_equal_original_including_unknown_days(self):
        t=self.torch;model,batch,_,_=old_preflight.fixture()
        m={k:t.tensor(v) for k,v in model.items()};b={k:t.tensor(v,dtype=t.int64) for k,v in batch.items()}
        query=dict(items=t.tensor([0,1,2]),users=t.tensor([0,1,2]),days=t.tensor([-1,-1,4]),day_owner=b['day_owner'])
        old=old_predict(m,b,query,2);new=new_predict(m,b,query,2)
        for k in old:self.assertTrue(t.equal(old[k],new[k]),k)

    def test_wrong_shapes_reject_before_any_shader(self):
        t=self.torch
        with self.assertRaises(ValueError):reduction.add_rows_(t.zeros((3,2)),t.tensor([0,1]),t.zeros((2,3)))


if __name__=='__main__':unittest.main()
