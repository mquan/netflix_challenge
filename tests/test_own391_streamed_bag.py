import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_own391_fold_bag import fold_values


class OriginalBagArithmetic(unittest.TestCase):
    def test_original_zero_slots_keep_denominator_three(self):
        baseline=np.array([2.],dtype=np.float64)
        actual=fold_values(baseline,[np.array([1.],dtype=np.float32),None,None])
        expected=np.array([2.+float(np.float32(1)/np.float32(3))],dtype=np.float32)
        np.testing.assert_array_equal(actual.view('u4'),expected.view('u4'))
        self.assertNotEqual(float(actual[0]),3.)

    def test_float32_division_changes_rounding_at_a_fold_boundary(self):
        third=float(np.float32(1)/np.float32(3))
        baseline=np.array([3.-third+2.**-23+2.e-9])
        actual=fold_values(baseline,[np.ones(1,dtype=np.float32),None,None])
        wrong=(baseline+1./3).astype(np.float32)
        self.assertEqual(float(actual[0]),float(np.nextafter(np.float32(3),np.float32(4))))
        self.assertEqual(float(wrong[0]),3.)

    def test_uneven_stream_matches_original_full_fold_ensemble(self):
        rng=np.random.default_rng(124);n=23
        baselines=[rng.uniform(.5,5.5,n) for _ in range(5)]
        slots=[[rng.normal(0,.3,n).astype(np.float32),None,rng.normal(0,.3,n).astype(np.float32)] for _ in range(5)]
        full=[]
        for baseline,three in zip(baselines,slots):
            correction=np.zeros(n)
            for raw in three:
                if raw is not None:correction+=np.divide(raw,np.float32(3),dtype=np.float32).astype(np.float64)
            full.append(np.asarray(np.clip(baseline+correction,1.05,4.95),dtype=np.float32))
        total=np.zeros(n)
        for values in full:total+=values.astype(np.float64)
        expected=(total/5).astype(np.float32)
        chunks=[]
        for lo in range(0,n,7):
            hi=min(n,lo+7);part=np.zeros(hi-lo)
            for baseline,three in zip(baselines,slots):
                part+=fold_values(baseline[lo:hi],[r[lo:hi] if r is not None else None for r in three]).astype(np.float64)
            chunks.append((part/5).astype(np.float32))
        np.testing.assert_array_equal(np.concatenate(chunks).view('u4'),expected.view('u4'))


if __name__=='__main__':unittest.main()
