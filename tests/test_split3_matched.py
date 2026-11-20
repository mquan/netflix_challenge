import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_split3_matched_data as data
import upstream_production_bank as production

class MatchedStudyTests(unittest.TestCase):
    def test_canonical_score_reclips_float32_boundaries(self):
        import math
        from verify_upstream_split3_matched_mps import score
        predictions=np.array([1.05,np.nextafter(np.float32(4.95),np.float32(np.inf)),3.25],dtype=np.float32)
        labels=np.array([1,5,3],dtype=np.uint8)
        expected=math.sqrt(math.fsum([(.05)**2,(-.05)**2,(.25)**2])/3)
        self.assertAlmostEqual(score(predictions,labels),expected,places=14)
        self.assertNotEqual(score(predictions,labels),math.sqrt(float(np.mean((predictions.astype(np.float64)-labels)**2))))

    def test_all_additional_rows_of_held_user_excluded(self):
        users=np.array([0,1,0,2,1,3,2,4,4,5])
        mapping=np.array([0,1,3,5,7,9]);original=np.array([0,3,4,7,8,1],dtype=np.int8)
        groups,canonical=data.partition(users,mapping,users[mapping],original)
        for fold in range(5):
            for arm in ['control','expanded']:
                train,fit,valid=data.masks(groups,canonical,arm,fold)
                self.assertFalse(np.isin(users[train],users[mapping][original//2==fold]).any())
                np.testing.assert_array_equal(fit|valid,train)
                self.assertFalse((fit&valid).any())
                if arm=='control':self.assertFalse((train&~canonical).any())
        self.assertTrue((data.masks(groups,canonical,'expanded',4)[0]&~canonical).any())

    def test_disagreeing_original_user_groups_rejected(self):
        with self.assertRaisesRegex(ValueError,'split a user'):
            data.partition(np.array([1,1]),np.array([0,1]),np.array([1,1]),np.array([0,2]))

    def test_generalized_layout_matches_existing_original_blocks(self):
        rng=np.random.default_rng(891);raw=rng.normal(3.6,2,(257,85)).astype(np.float32)
        context=rng.normal(size=(257,5));voting=rng.normal(size=(257,130)).astype(np.float32)
        z,f=data.blocks(raw[:,:73],context,voting)
        old_z,old_f=production.blocks(raw,np.ones(85,dtype=bool),context,voting)
        # All shared old feature positions and every generalized design column
        # have identical bits, including first62 mean's float32 reduction order.
        np.testing.assert_array_equal(f[:,:79].view('u4'),old_f[:,:79].view('u4'))
        np.testing.assert_array_equal(f[:,79:].view('u4'),old_f[:,91:].view('u4'))
        np.testing.assert_array_equal(z[:,:148].view('u8'),old_z[:,:148].view('u8'))
        np.testing.assert_array_equal(z[:,-1].view('u8'),old_z[:,-1].view('u8'))

if __name__=='__main__':unittest.main()
