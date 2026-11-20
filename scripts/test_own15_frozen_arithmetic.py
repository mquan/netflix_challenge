"""Full simplex search, nonunique optima and frozen prediction arithmetic."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import upstream_own15_complete_trajectory_calibration as fitting
import verify_upstream_own15_complete_trajectory_calibration as independent


class FrozenOwn15(unittest.TestCase):
    def test_all32767_supports_match_uniform_analytic_optimum(self):
        gram,rhs=np.eye(15),np.zeros(15)
        weights,count=independent.support_solution(gram,rhs)
        self.assertEqual(count,32767)
        np.testing.assert_allclose(weights,np.full(15,1/15),atol=1e-14)
        independent.verify_simplex_weights(gram,rhs,np.full(15,1/15),weights)
        wrong=np.zeros(15);wrong[0]=1
        with self.assertRaises(ValueError):independent.verify_simplex_weights(gram,rhs,wrong,weights)

    def test_boundary_optimum_retains_full_support_search(self):
        rhs=np.zeros(15);rhs[3]=10
        weights,count=independent.support_solution(np.eye(15),rhs)
        expected=np.zeros(15);expected[3]=1
        self.assertEqual(count,32767)
        np.testing.assert_allclose(weights,expected,atol=1e-14)
        independent.verify_simplex_weights(np.eye(15),rhs,expected,weights)

    def test_duplicate_predictors_allow_equivalent_optimal_weights(self):
        gram,rhs=np.zeros((15,15)),np.zeros(15)
        representative,count=independent.support_solution(gram,rhs)
        uniform=np.full(15,1/15)
        self.assertEqual(count,32767)
        self.assertGreater(np.max(abs(representative-uniform)),.5)
        proof=independent.verify_simplex_weights(gram,rhs,uniform,representative)
        self.assertEqual(proof['absolute_gap'],0.)
        self.assertEqual(proof['complementarity_error'],0.)

    def test_every_frozen_bit_replays_and_single_bit_corruption_rejected(self):
        arrays=[np.array([1.,3.,5.],dtype='<f4'),np.array([1.1,4.,4.9],dtype='<f4')]
        expected=fitting.combine(arrays,[.5,.5])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.f32';expected.tofile(path)
            actual=independent.stored(path,arrays,[.5,.5])
            self.assertEqual(actual.tobytes(),expected.tobytes())
            wrong=expected.copy();wrong.view('u4')[1]^=1;wrong.tofile(path)
            with self.assertRaises(ValueError):independent.stored(path,arrays,[.5,.5])

    def test_fitting_closure_rejects_recovered_outcomes(self):
        with self.assertRaises(ValueError):fitting.verify_bindings({'missing/qual_ratings.csv.gz':'not-a-digest'})

    def test_linked_known_metadata_inherits_original_expected_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);known=root/'artifacts/new-neural'
            expected='a'*64
            bindings={'artifacts/upstream-complete-trajectory-data/labels.npy':expected}
            with mock.patch.object(fitting.u,'ROOT',root),mock.patch.object(fitting,'KNOWN',known):
                fitting.bind_known_metadata(bindings,'labels.npy')
                self.assertEqual(bindings['artifacts/new-neural/labels.npy'],expected)
                with self.assertRaises(ValueError):fitting.bind_known_metadata({},'labels.npy')


if __name__=='__main__':unittest.main()
