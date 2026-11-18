import unittest
import upstream_voting_blend as v


class VotingChecks(unittest.TestCase):
    def receipt(self):
        return dict(outputs={f'preds_new/vf/vf{i:03}_feature.{split}.npy':'sha'
            for i in range(130) for split in ['probex','qual']})

    def test_complete_feature_pairing_and_canonical_order(self):
        values=v.feature_inventory(self.receipt())
        self.assertEqual(len(values),130)
        self.assertTrue(values[0].endswith('vf000_feature.probex.npy'))
        self.assertTrue(values[-1].endswith('vf129_feature.probex.npy'))

    def test_missing_qual_counterpart_is_rejected(self):
        receipt=self.receipt();del receipt['outputs']['preds_new/vf/vf015_feature.qual.npy']
        with self.assertRaisesRegex(ValueError,'counterpart'):v.feature_inventory(receipt)

    def test_duplicate_feature_index_does_not_satisfy_inventory(self):
        receipt=self.receipt();del receipt['outputs']['preds_new/vf/vf015_feature.probex.npy']
        receipt['outputs']['preds_new/vf/vf014_other.probex.npy']='sha'
        with self.assertRaisesRegex(ValueError,'indices'):v.feature_inventory(receipt)


if __name__=='__main__':unittest.main()
