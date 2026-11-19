import unittest
import numpy as np
import upstream_split3_pilot as pilot


class Split3ProjectionTest(unittest.TestCase):
    def arrays(self,rows):
        return [np.asarray([row[k] for row in rows],dtype=np.int64) for k in range(4)]

    def test_exact_permuted_projection_and_separate_training(self):
        source=self.arrays([(3,2,8,5),(1,2,8,4),(1,3,8,2),(2,4,9,1)])
        known=self.arrays([(1,3,8,2),(3,2,8,5)])
        mapping=pilot.projection(source,known)
        np.testing.assert_array_equal(mapping,[2,0])
        np.testing.assert_array_equal(np.asarray([.1,.2,.3,.4],np.float32)[mapping],np.asarray([.3,.1],np.float32))
        held=np.sort(pilot.keys(*known[:3]))
        other=self.arrays([(3,2,9,1),(1,4,8,4),(2,3,8,5)])
        pilot.assert_disjoint(held,other[:3])
        with self.assertRaisesRegex(ValueError,'appears in training'):
            pilot.assert_disjoint(held,source[:3])

    def test_duplicate_missing_and_changed_label_are_rejected(self):
        rows=[(1,2,8,4),(2,3,9,5)]
        source=self.arrays(rows)
        for bad in [self.arrays([rows[0],rows[0]]),self.arrays([(9,9,9,1)]),self.arrays([(1,2,8,1)])]:
            with self.assertRaises(ValueError):pilot.projection(source,bad)
        with self.assertRaises(ValueError):
            pilot.projection(self.arrays([rows[0],rows[0]]),self.arrays([rows[0]]))

    def test_identity_bounds_are_rejected(self):
        for row in [(-1,2,0,1),(480189,2,0,1),(1,17770,0,1),(1,2,32768,1)]:
            with self.assertRaises(ValueError):pilot.keys(*self.arrays([row])[:3])

    def test_memory_gate_requires_stable_allocated_temporal_or_primary_absent(self):
        before=dict(compressor_bytes=2*2**30,swap_used_bytes=100)
        after=dict(before)
        self.assertTrue(pilot.memory_admitted(before,after,18*2**30,True,True))
        self.assertFalse(pilot.memory_admitted(before,after,18*2**30,True,False))
        self.assertFalse(pilot.memory_admitted(before,after,21*2**30,True,True))
        self.assertTrue(pilot.memory_admitted(before,after,11*2**30,False,False))
        self.assertFalse(pilot.memory_admitted(before,after,13*2**30,False,False))
        self.assertFalse(pilot.memory_admitted(before,dict(after,swap_used_bytes=100+33*2**20),18*2**30,True,True))


if __name__=='__main__':unittest.main()
