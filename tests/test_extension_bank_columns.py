import unittest
import numpy as np
import verify_upstream_extension_bank as v


class BankColumnTest(unittest.TestCase):
    def fixture(self,seed):
        rng=np.random.default_rng(seed)
        raw=rng.normal(3.5,1,size=(4,91)).astype(np.float32)
        clip=np.ones(91,dtype=bool);clip[4]=False;raw[:,4]=-2
        metadata=rng.normal(size=(4,5));votes=rng.normal(size=(4,130)).astype(np.float32)
        z,f=v.bank.parent.core.blocks(raw[:,:85],clip[:85],metadata,votes)
        z,f=v.bank.blocks(z,f,raw[:,85:]);return raw,clip,z,f,votes

    def test_declared_pair_math_and_corruption_detection(self):
        raw,clip,z,f,votes=self.fixture(7);v.check_block(raw,clip,z,f,votes)
        for column in [0,70,137,160,166]:
            changed=z.copy();changed[0,column]+=1
            with self.subTest(column=column),self.assertRaises(ValueError):v.check_block(raw,clip,changed,f,votes)
        for column in [0,2,65,70,100,225]:
            changed=f.copy();changed[0,column]+=1
            with self.subTest(feature=column),self.assertRaises(ValueError):v.check_block(raw,clip,z,changed,votes)

    def test_retrained_vectors_are_diagnostic_but_row_identities_match(self):
        a,clip,z,f,votes=self.fixture(9)
        other=a.copy();other[:,2]+=.03
        zz,ff=v.bank.parent.core.blocks(other[:,:85],clip[:85],z[:,62:67],votes)
        zz,ff=v.bank.blocks(zz,ff,other[:,85:])
        v.check_block(other,clip,zz,ff,votes)
        differences=v.compare_rows(zz,z)
        self.assertFalse(differences['all_values_equal']);self.assertGreater(differences['unequal_values'],0)
        labels=np.array([1,2,3,5],dtype=np.int8)
        self.assertTrue(v.compare_rows(labels,labels.astype(np.float64))['all_values_equal'])


if __name__=='__main__':unittest.main()
