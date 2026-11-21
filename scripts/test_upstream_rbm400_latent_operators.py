"""Full-width inherited-bit and coordinate-boundary checks; no actual data."""
import unittest
import numpy as np
import upstream_rbm400_latent_operators as ops
import upstream_complete_trajectory_mps as existing


class OperatorTests(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(128)
        self.z=r.normal(size=(11,562));self.z[:,-1]=1.
        self.f=r.normal(size=(11,712)).astype('<f4')
        self.z[0,3]=-0.;self.f[0,12]=-0.
        self.p=np.linspace(-1,7,11,dtype=np.float32)
        self.context=r.normal(size=(11,150)).astype('<f4')

    def test_scalar_arm_preserves_all_old_bits_and_original_intercept(self):
        z,f=ops.extend(self.z,self.f,self.p,retain_scalar=True,latent=False)
        np.testing.assert_array_equal(z[:,:561].view('u8'),self.z[:,:-1].view('u8'))
        np.testing.assert_array_equal(z[:,-1:].view('u8'),self.z[:,-1:].view('u8'))
        np.testing.assert_array_equal(f[:,:712].view('u4'),self.f.view('u4'))
        np.testing.assert_array_equal(z[:,561],np.clip(self.p,0,6))
        np.testing.assert_array_equal(f[:,712],np.clip(self.p,0,6)-np.float32(3.6))
        self.assertEqual((z.shape,f.shape),((11,563),(11,713)))

    def test_latent_arm_adds_all150_only_to_nonlinear_operator(self):
        a,b=ops.extend(self.z,self.f,self.p,retain_scalar=True,latent=False)
        z,f=ops.extend(self.z,self.f,self.p,self.context,retain_scalar=True,latent=True)
        np.testing.assert_array_equal(z.view('u8'),a.view('u8'))
        np.testing.assert_array_equal(f[:,:713].view('u4'),b.view('u4'))
        np.testing.assert_array_equal(f[:,713:].view('u4'),self.context.view('u4'))
        self.assertEqual((z.shape,f.shape),((11,563),(11,863)))

    def test_declared_scalar_alias_still_preserves_every_latent_coordinate(self):
        z,f=ops.extend(self.z,self.f,self.p,self.context,retain_scalar=False,latent=True)
        np.testing.assert_array_equal(z.view('u8'),self.z.view('u8'))
        np.testing.assert_array_equal(f[:,:712].view('u4'),self.f.view('u4'))
        np.testing.assert_array_equal(f[:,712:].view('u4'),self.context.view('u4'))
        self.assertEqual(ops.dimensions(False,True)['nn_inputs'],863)

    def test_both_arms_and_alias_cases_partition_all_actual_old_and_new_coordinates(self):
        old_signals=existing.signal_indices(713);old_context=existing.CONTEXT
        for retain in (False,True):
            for latent in (False,True):
                with self.subTest(retain=retain,latent=latent):
                    result=ops.layout(old_signals,old_context,retain_scalar=retain,latent=latent)
                    self.assertEqual(result['signals'][:-1-int(retain)],old_signals[:-1])
                    self.assertEqual(result['context'][:225],old_context)
                    self.assertEqual(len(result['context']),375 if latent else 225)
                    self.assertEqual(sorted(result['signals']+result['context']),list(range(result['nn_inputs'])))
                    self.assertEqual(result['signals'][-1],result['nn_inputs']-1)

    def test_original_operators_and_new_feature_buffers_are_not_mutated(self):
        before=[v.tobytes() for v in (self.z,self.f,self.p,self.context)]
        ops.extend(self.z,self.f,self.p,self.context,retain_scalar=True,latent=True)
        self.assertEqual(before,[v.tobytes() for v in (self.z,self.f,self.p,self.context)])

    def test_nonfinite_prediction_cannot_be_hidden_by_clipping(self):
        p=self.p.copy();p[0]=np.inf
        with self.assertRaises(ValueError):ops.extend(self.z,self.f,p,retain_scalar=True,latent=False)

    def test_missing_or_extra_context_wrong_intercept_or_types_reject(self):
        with self.assertRaises(ValueError):ops.extend(self.z,self.f,self.p,retain_scalar=True,latent=True)
        with self.assertRaises(ValueError):ops.extend(self.z,self.f,self.p,self.context,retain_scalar=True,latent=False)
        z=self.z.copy();z[0,-1]=0
        with self.assertRaises(ValueError):ops.extend(z,self.f,self.p,retain_scalar=True,latent=False)
        with self.assertRaises(ValueError):ops.dimensions(1,True)


if __name__=='__main__':unittest.main()
