"""Check complete eight-column geometry, partition and unchanged training budget."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_split3eight_virtual_features as data
import upstream_split3eight_compact_mps_context as model

class EightColumnTests(unittest.TestCase):
    def test_every_new_column_and_retained_context(self):
        rng=np.random.default_rng(17)
        z=rng.normal(size=(37,183));f=rng.normal(size=(37,397)).astype(np.float32)
        raw=[rng.uniform(-2,8,37).astype(np.float32) for _ in range(8)]
        actual_z,actual_f=data.block((z,f,raw),3,31)
        expected=np.column_stack([np.clip(a[3:31],0,6) for a in raw])
        self.assertEqual(actual_z.shape,(28,191));self.assertEqual(actual_f.shape,(28,405))
        np.testing.assert_array_equal(actual_z,np.column_stack([z[3:31,:182],expected,z[3:31,-1]]))
        np.testing.assert_array_equal(actual_f[:,:243],f[3:31,:243])
        np.testing.assert_array_equal(actual_f[:,243:251],expected-np.float32(3.6))
        np.testing.assert_array_equal(actual_f[:,251:],f[3:31,243:])
    def test_all_eight_unique_and_full_training_recipe(self):
        self.assertEqual(len(data.NEW),8);self.assertEqual(len(set(data.NEW)),8)
        self.assertNotIn('s3-attn16',data.NEW)
        self.assertEqual(model.recipe.SIGNALS,list(range(63))+list(range(68,91))+list(range(221,251))+[405])
        self.assertEqual(model.recipe.CONTEXT[-154:],list(range(251,405)))
        self.assertFalse(set(model.recipe.SIGNALS)&set(model.recipe.CONTEXT))
        self.assertEqual(set(model.recipe.SIGNALS)|set(model.recipe.CONTEXT),set(range(406)))
        for name,value in [('nn_hidden',128),('context_rank',32),('nn_epochs',40),('nn_batch',8192),('seeds',[2026,7,13])]:
            self.assertEqual(model.CONFIG[name],value)

if __name__=='__main__':unittest.main()
