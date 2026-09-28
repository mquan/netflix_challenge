"""A new holdout must hide labels while retaining every observed interaction."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

from test_joint import dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import load  # noqa: E402
from train_holdout import create  # noqa: E402


class TrainHoldoutTests(unittest.TestCase):
    def test_split_preserves_implicit_pairs_and_hides_original_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dataset(root / 'source.bin')
            report = create(root / 'source.bin', root / 'split.bin', .35, 19, 'random')
            source, split = load(root / 'source.bin'), load(root / 'split.bin')
            self.assertGreater(report['holdout_count'], 0)
            self.assertEqual(report['source_probe_count'], 200)
            self.assertEqual(len(split['probe']) + len(split['train_ratings']),
                             len(source['train_ratings']))
            for key in ('implicit_offsets', 'implicit_items', 'implicit_days'):
                np.testing.assert_array_equal(split[key], source[key])
            self.assertEqual(set(split['probe']['item']), {1, 2, 3})
            self.assertTrue(np.all(np.diff(split['train_offsets']) > 0))
            self.assertAlmostEqual(split['mean'], np.mean(split['train_ratings']))
            with self.assertRaises(FileExistsError):
                create(root / 'source.bin', root / 'split.bin', .35, 19, 'random')

    def test_split_does_not_use_rating_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dataset(root / 'a.bin')
            dataset(root / 'b.bin', correlated_train=True)
            create(root / 'a.bin', root / 'a-split.bin', .35, 7, 'random')
            create(root / 'b.bin', root / 'b-split.bin', .35, 7, 'random')
            a, b = load(root / 'a-split.bin'), load(root / 'b-split.bin')
            for key in ('train_offsets', 'train_items', 'train_days'):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a['probe']['user'], b['probe']['user'])
            np.testing.assert_array_equal(a['probe']['item'], b['probe']['item'])

    def test_matched_follows_probe_users_without_reading_their_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dataset(root / 'a.bin')
            dataset(root / 'b.bin', change_fold=0)
            create(root / 'a.bin', root / 'a-split.bin', .014, 7)
            create(root / 'b.bin', root / 'b-split.bin', .014, 7)
            a, b = load(root / 'a-split.bin'), load(root / 'b-split.bin')
            np.testing.assert_array_equal(a['train_offsets'], b['train_offsets'])
            np.testing.assert_array_equal(a['probe']['user'], b['probe']['user'])
            np.testing.assert_array_equal(a['probe']['item'], b['probe']['item'])
            self.assertEqual(len(a['probe']), 200)


if __name__ == '__main__':
    unittest.main()
