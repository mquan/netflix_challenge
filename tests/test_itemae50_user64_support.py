"""Scientific projection contract and incomplete-source rejection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_itemae50_user64_support as source

class UserContextTests(unittest.TestCase):
    def test_full_width_projection_has_correct_spectrum_and_user_coordinates(self):
        rng=np.random.default_rng(23);a=rng.normal(size=(1027,128))
        table,p=source.project(a)
        cov=np.cov(a,rowvar=False,bias=True)
        np.testing.assert_allclose(p['eigenvalues'],np.linalg.eigvalsh(cov)[::-1][:32],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(cov@p['vectors'],p['vectors']*p['eigenvalues'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(table.mean(0),0,atol=1e-7)
        np.testing.assert_allclose(table.var(0),1,atol=2e-6)
        self.assertTrue((p['vectors'][np.abs(p['vectors']).argmax(0),np.arange(32)]>0).all())
        ids=np.array([1026,0,302,0,18])
        expected=((a[ids]-p['center'])@p['vectors']/p['scales']).astype(np.float32)
        np.testing.assert_allclose(table[ids],expected,rtol=0,atol=1e-6)
        np.testing.assert_array_equal(table[ids][1],table[ids][3])
    def test_degenerate_factor_spectrum_rejected(self):
        with self.assertRaisesRegex(ValueError,'Positive'):
            source.project(np.ones((129,128)))
    def test_partial_source_cannot_start_projection(self):
        with patch.object(source.raw,'verify_plan',return_value={}),patch.object(source.u,'read',side_effect=[{'state':'incomplete'},{},{}]):
            with self.assertRaisesRegex(ValueError,'Independently complete'):
                source.inputs()

if __name__=='__main__':unittest.main()
