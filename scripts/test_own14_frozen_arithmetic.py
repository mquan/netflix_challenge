import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_own14_matched391_calibration as fitting
import verify_upstream_own14_matched391_calibration as independent


class FrozenOwn14(unittest.TestCase):
    def test_all16383_supports_match_uniform_analytic_optimum(self):
        weights,count=independent.support_solution(np.eye(14),np.zeros(14))
        self.assertEqual(count,16383);np.testing.assert_allclose(weights,np.full(14,1/14),atol=1e-14)

    def test_boundary_optimum_retains_all14_support_search(self):
        rhs=np.zeros(14);rhs[3]=10
        weights,count=independent.support_solution(np.eye(14),rhs)
        expected=np.zeros(14);expected[3]=1
        self.assertEqual(count,16383);np.testing.assert_allclose(weights,expected,atol=1e-14)

    def test_store_is_independent_and_one_bit_change_is_rejected(self):
        arrays=[np.array([1.,3.,5.],dtype='<f4'),np.array([1.1,4.,4.9],dtype='<f4')]
        expected=fitting.combine(arrays,[.5,.5])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.f32';expected.tofile(path)
            actual=independent.stored(path,arrays,[.5,.5]);self.assertEqual(actual.tobytes(),expected.tobytes())
            wrong=expected.copy();wrong.view('u4')[1]^=1;wrong.tofile(path)
            with self.assertRaises(ValueError):independent.stored(path,arrays,[.5,.5])

    def test_descriptor_local_hash_preserves_fixture(self):
        payload=bytes(range(256))*8192
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'private.bin';path.write_bytes(payload);before=path.stat()
            self.assertEqual(fitting.sha(path),hashlib.sha256(payload).hexdigest())
            self.assertEqual(path.read_bytes(),payload);self.assertEqual(path.stat().st_mtime_ns,before.st_mtime_ns)

    def test_fitting_closure_cannot_include_recovered_payload(self):
        with self.assertRaises(ValueError):fitting.verify_bindings({'missing/qual_ratings.csv.gz':'not-a-digest'})


if __name__=='__main__':unittest.main()
