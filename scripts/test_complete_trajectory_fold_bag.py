"""Preserve all seed slots and the original mixed-precision inference arithmetic."""
import unittest
import numpy as np
import upstream_complete_trajectory_fold_bag as production


class CompleteTrajectoryBagTests(unittest.TestCase):
    def test_zero_budget_slot_does_not_renormalize_other_seeds(self):
        baseline=np.array([3.,3.],dtype='f8')
        a=np.array([.3,.6],dtype='f4');b=np.array([.3,.3],dtype='f4')
        result=production.fold_values(baseline,[a,None,b])
        np.testing.assert_array_equal(result.view('u4'),np.array([3.2,3.3],dtype='f4').view('u4'))
        result=production.fold_values(np.array([5.,1.]),[None,None,None])
        np.testing.assert_array_equal(result,np.array([4.95,1.05],dtype='f4'))

    def test_float32_seed_division_then_ordered_float64_accumulation(self):
        baseline=np.array([3.6,2.25,4.94,1.06])
        slots=[np.array([1e7,1e-4,.1,-.1],dtype='f4'),
               np.array([-1e7,-1e-4,.1,-.1],dtype='f4'),
               np.array([.03,0.,-.1,.1],dtype='f4')]
        expected=np.zeros(len(baseline),dtype='f8')
        for values in slots:
            expected=np.add(expected,np.divide(values,np.float32(3),dtype=np.float32).astype(np.float64))
        expected=np.asarray(np.clip(baseline+expected,1.05,4.95),dtype='<f4')
        np.testing.assert_array_equal(production.fold_values(baseline,slots).view('u4'),expected.view('u4'))

    def test_missing_slot_wrong_dtype_and_nonfinite_seed_rejected(self):
        baseline=np.array([3.,3.])
        invalid=[[None,None],[np.ones(2,dtype='f8'),None,None],
                 [np.array([np.inf,0.],dtype='f4'),None,None],
                 [np.ones(1,dtype='f4'),None,None]]
        for slots in invalid:
            with self.assertRaises(ValueError):production.fold_values(baseline,slots)


if __name__=='__main__':unittest.main()
