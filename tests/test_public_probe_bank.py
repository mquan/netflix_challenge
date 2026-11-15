import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from public_probe_bank import eligible, eligible_three, remap


class ImportTests(unittest.TestCase):
    def test_only_original_base_columns(self):
        self.assertTrue(eligible('preds_old/tsvdx5-400__nlpp__knn3x.probe.npy'))
        for p in ('preds_new/tsvdx5-400.probe.npy', 'preds_old/rbm-32.qual.npy',
                  'preds_old/fwls/A1.probe.npy', 'preds_old/gbr1-s1.probe.npy',
                  'preds_old/tsvdx4_fwls_10xblended_lambda200.probe.npy',
                  'preds_old/../rbm-32.probe.npy'):
            self.assertFalse(eligible(p), p)

    def test_mapping_direction_and_shape(self):
        np.testing.assert_array_equal(remap(np.array([1., 3., 5.]), np.array([2, 0, 1])), [5, 1, 3])
        with self.assertRaises(ValueError): remap(np.ones((3, 1)), np.arange(3))
        with self.assertRaises(ValueError): remap(np.array([1., np.nan]), np.arange(2))

    def test_safe_training_subset_projection(self):
        self.assertTrue(eligible_three('preds_3/s3-tx300lm.probe_3.npy'))
        self.assertFalse(eligible_three('preds_3/fwls15k4-s1.probe_3.npy'))
        self.assertFalse(eligible_three('preds_3/s3-tx300lm.qual.npy'))
        self.assertFalse(eligible_three('preds_new/s3-tx300lm.probe_3.npy'))
        np.testing.assert_array_equal(remap(np.array([1., 2., 3., 4.]), np.array([3, 1]), 4), [4, 2])


if __name__ == '__main__': unittest.main()
