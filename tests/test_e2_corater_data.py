import json
import struct
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_corater_data import Coraters, build
from gbdt_blend import HEADER, PROBE
from test_joint import dataset


class CoraterDataTests(unittest.TestCase):
    def test_labels_never_change_context_and_query_owner_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for tag, fold in [('a', None), ('b', 0)]:
                dataset(root/(tag+'.bin'), change_fold=fold)
                build(root/(tag+'.bin'), root/tag)
            for name in ('rows.npy', 'users.npy', 'offsets.npy'):
                self.assertEqual((root/'a'/name).read_bytes(), (root/'b'/name).read_bytes())
            a = Coraters(root/'a.bin', root/'a')
            b = Coraters(root/'b.bin', root/'b')
            query = np.zeros(1, dtype=PROBE)
            query['user'], query['item'], query['day'], query['rating'] = 5, 2, 3, 1
            first = a.queries(query, seed=31, cap=64)
            query['rating'] = 5
            second = b.queries(query, seed=31, cap=64)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(first['mask'].sum(), 64)
            self.assertFalse(np.any(first['users'][first['mask']] == 5))
            self.assertEqual(len(np.unique(first['users'][first['mask']])), 64)
            self.assertTrue(np.all(a.d['train_items'][first['rows'][first['mask']]] == 2))

    def test_empty_context_and_user_zero_have_explicit_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset(root/'data.bin')
            build(root/'data.bin', root/'index')
            index = Coraters(root/'data.bin', root/'index')
            empty = index.sample(5, 4, 3, seed=1)
            self.assertFalse(empty['mask'].any())
            self.assertTrue(np.all(empty['rows'] == -1))
            all_users = index.sample(5, 2, 3, seed=1, cap=100)
            self.assertEqual(all_users['mask'].sum(), 99)
            self.assertIn(0, all_users['users'][all_users['mask']])
            self.assertFalse(all_users['mask'][-1])

    def test_uniform_sampling_over_eligible_users(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset(root/'data.bin')
            build(root/'data.bin', root/'index')
            index = Coraters(root/'data.bin', root/'index')
            counts = np.zeros(100, dtype=int)
            for seed in range(2000):
                sample = index.sample(5, 2, 3, seed=seed, cap=50)
                counts[sample['users'][sample['mask']]] += 1
            self.assertEqual(counts[5], 0)
            eligible = np.delete(counts, 5)
            self.assertLess(np.max(np.abs(eligible-2000*50/99)), 100)

    def test_changed_inputs_and_overwrites_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset(root/'data.bin')
            build(root/'data.bin', root/'index')
            with self.assertRaises(FileExistsError):
                build(root/'data.bin', root/'index')
            dataset(root/'changed.bin', change_fold=0)
            with self.assertRaises(ValueError):
                Coraters(root/'changed.bin', root/'index')
            payload = root/'index/users.npy'
            raw = bytearray(payload.read_bytes()); raw[-1] ^= 1; payload.write_bytes(raw)
            with self.assertRaises(ValueError):
                Coraters(root/'data.bin', root/'index')

    def test_duplicate_user_movie_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset(root/'data.bin')
            raw = bytearray((root/'data.bin').read_bytes())
            # User zero's second training item becomes a second rating of item 1.
            struct.pack_into('<H', raw, HEADER.itemsize + 101*8 + 2, 1)
            (root/'data.bin').write_bytes(raw)
            with self.assertRaises(ValueError):
                build(root/'data.bin', root/'index')
            self.assertFalse((root/'index/manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
