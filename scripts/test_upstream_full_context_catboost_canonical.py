"""Known-only canonical arithmetic and fixed both-arm choice fixtures."""
import math
import unittest
import numpy as np
import upstream_full_context_catboost_canonical as module


class CanonicalTests(unittest.TestCase):
    def test_every_error_is_reclipped_in_float64_and_compensated(self):
        p=np.asarray([1.05,4.95,3.1234567],dtype='<f4');y=np.asarray([1,5,2],dtype='<f8')
        deltas=[max(1.05,min(4.95,float(v)))-float(t) for v,t in zip(p,y)]
        expected=math.sqrt(math.fsum(d*d for d in deltas)/3)
        self.assertEqual(module.score(p,y),expected)
        raw=float(np.sqrt(np.mean((p.astype(np.float64)-y)**2)))
        self.assertNotEqual(module.score(p,y),raw)
        with self.assertRaises(ValueError):module.score(p.astype('<f8'),y)
        with self.assertRaises(ValueError):module.score(p,np.asarray([1,6,2],dtype='<f8'))

    def test_fixed_complete_both_arm_choice_and_scalar_tie(self):
        self.assertEqual(module.choose({'scalar':.858,'latent':.857}),'latent')
        self.assertEqual(module.choose({'scalar':.857,'latent':.858}),'scalar')
        self.assertEqual(module.choose({'scalar':.857+5e-13,'latent':.857}),'scalar')
        with self.assertRaises(ValueError):module.choose({'scalar':.857})
        with self.assertRaises(ValueError):module.choose({'scalar':float('nan'),'latent':.857})


if __name__=='__main__':unittest.main()
