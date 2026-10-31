import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from structural_scoping import arm_indices, write_bank
from train_holdout import digest


class ScopingTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(columns=[{}]*6, families=dict(a=[0, 1, 2], b=[3, 4, 5]),
            diversity=dict(counts=[1, 3, 6], permutation_seeds=[42, 43]))

    def test_nested_subsets_and_quality_ties_are_fixed(self):
        scores = [.8, .9, .9, .7, .7, 1.0]
        arms = arm_indices(self.plan, scores)
        self.assertEqual(arms, arm_indices(self.plan, scores))
        self.assertEqual(len(arms), 7)
        for seed in [42, 43]:
            subsets = [a['indices'] for a in arms if a.get('seed') == seed]
            self.assertTrue(set(subsets[0]) <= set(subsets[1]))
        quality = {a['family']: a for a in arms if a['kind'] == 'quality'}
        self.assertEqual(quality['a']['indices'], [1, 1, 2, 3, 4, 5])
        self.assertEqual(quality['b']['indices'], [0, 1, 2, 5, 4, 5])

    def test_manifest_round_trip_preserves_duplicate_columns_and_support_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for i in range(3):
                path = root/f'original-{i}.f32'
                np.array([i+1, i+2], '<f4').tofile(path)
                records.append(dict(path=str(path), sha256=digest(path)))
            bank = root/'bank'
            frozen = write_bank(bank, records, [2, 1, 2], records[::-1])
            union = json.loads((bank/'experiments/UNION-round-manifest.json').read_text())['reports']['union']['inputs']
            support = json.loads((bank/'experiments/BOOST-support-inputs.json').read_text())
            self.assertEqual(len(union), 3)
            loaded = np.column_stack([np.fromfile(p, '<f4') for p in union])
            np.testing.assert_array_equal(loaded, [[3, 2, 3], [4, 3, 4]])
            np.testing.assert_array_equal(np.column_stack([np.fromfile(p, '<f4') for p in support]),
                                          [[3, 2, 1], [4, 3, 2]])
            for path, sha in {**frozen, **union, **support}.items():
                self.assertEqual(digest(path), sha)

    def test_bad_score_or_family_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            arm_indices(self.plan, [float('nan')]*6)
        self.plan['families']['b'] = [2, 4, 5]
        with self.assertRaisesRegex(ValueError, 'partition'):
            arm_indices(self.plan, [1.0]*6)


if __name__ == '__main__':
    unittest.main()
