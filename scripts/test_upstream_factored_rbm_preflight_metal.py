"""Preflight protocols on CPU/small synthetic cases; never acquire MPS here."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import upstream_factored_rbm_preflight_metal as preflight


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def test_cpu_small_check_never_claims_production_mps_and_restores_runtime_flag(self):
        previous=self.torch.are_deterministic_algorithms_enabled();calls=[]
        result=preflight.run(lambda:calls.append(True),'cpu',stress=False)
        self.assertFalse(result['actual_MPS_tested']);self.assertFalse(result['full_required_stress_tested'])
        self.assertFalse(result['passed_for_production_device']);self.assertTrue(result['cases']['small']['cold_user_checked'])
        self.assertTrue(result['cases']['small']['same_device_prediction_repeats_bitwise_equal'])
        self.assertEqual(self.torch.are_deterministic_algorithms_enabled(),previous);self.assertGreater(len(calls),3)

    def test_heavy_user_generator_preserves_all_rows_and_full_stress_spec(self):
        m,b,h,v,q,shape=preflight.stress_fixture(n_users=3,n_items=11,hidden=4,factors=3,ordinary_rows=3)
        self.assertEqual((shape['training_rows'],shape['conditional_rows'],shape['query_rows'],shape['max_user_rows']),(17,19,6,11))
        self.assertEqual(np.count_nonzero(b['users']==0),11);self.assertEqual(len(np.unique(b['items'][b['users']==0])),11)
        result=preflight.check_case(m,b,h,v,lambda:None,'cpu',query=q,tile_rows=4)
        self.assertTrue(result['same_device_prediction_repeats_bitwise_equal'])
        spec=preflight.REQUIRED_STRESS
        self.assertEqual((spec['users'],spec['hidden'],spec['factors'],spec['max_user_rows']),(500,400,50,17770))
        self.assertGreater(spec['max_user_rows'],16384)

    def test_mps_fallback_or_fast_math_is_rejected_before_any_gpu_operation(self):
        with patch.dict(os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'1','PYTORCH_MPS_FAST_MATH':'0'}):
            with self.assertRaises(ValueError):preflight.run(lambda:None,'mps',stress=False)
        with self.assertRaises(ValueError):preflight.run(lambda:None,'cuda',stress=False)

    def test_synthetic_guard_failure_restores_deterministic_runtime_flag(self):
        previous=self.torch.are_deterministic_algorithms_enabled()
        count=0
        def fail():
            nonlocal count
            count+=1
            if count==2:raise RuntimeError('synthetic fatal guard')
        with self.assertRaises(RuntimeError):preflight.run(fail,'cpu',stress=False)
        self.assertEqual(self.torch.are_deterministic_algorithms_enabled(),previous)

    def test_changed_environment_cannot_qualify_preimported_torch_as_fresh_mps(self):
        with patch.dict(os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}):
            with self.assertRaisesRegex(ValueError,'already imported'):
                preflight.run(lambda:None,'mps',stress=False)
