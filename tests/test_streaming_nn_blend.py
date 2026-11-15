import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from streaming_nn_blend import standardized
from round8_nn_blend import standardize


class NormalizationTests(unittest.TestCase):
    def test_memory_and_disk_scaling_match_original(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(1000,37)); x[:,0]=1
        fit=np.arange(1000)%5!=0
        expected=standardize(x,fit)
        np.testing.assert_array_equal(standardized(x,fit,block=7),expected)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'norm.npy'
            actual=standardized(np.asfortranarray(x),fit,block=7,output=p)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(np.load(p),expected)


if __name__=='__main__': unittest.main()
