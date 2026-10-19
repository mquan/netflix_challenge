import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import HEADER, PROBE, load
from production_dataset import create


class ProductionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training = self.root / 'raw'
        self.training.mkdir()
        for movie, content in enumerate([
            '40,5,1999-01-01\n7,3,1999-01-02\n',
            '99,2,1999-01-03\n40,4,1999-01-04\n',
            '7,1,1999-01-05\n99,5,1999-01-06\n',
        ], 1):
            (self.training / f'mv_{movie:07d}.txt').write_text(f'{movie}:\n' + content)
        self.qualifying = self.root / 'qualifying.txt'
        # Deliberately neither dense-user nor movie order.
        self.qualifying.write_text('1:\n99,1999-01-07\n3:\n40,1999-01-08\n2:\n7,1999-01-09\n')
        self.source = self.root / 'dev.bin'
        header = np.zeros(1, dtype=HEADER)
        header[0] = (b'NFPDAT2\0', 3, 3, 5, 1, 9, 0, 5, 0, 3.2)
        probe = np.zeros(1, dtype=PROBE)
        probe[0] = (0, 2, 3, 4, b'\0\0\0')
        with self.source.open('wb') as stream:
            for part in (header, np.array([0, 1, 3, 5], dtype='<u8'),
                         np.array([1, 1, 3, 2, 3], dtype='<u2'), np.array([0, 1, 4, 2, 5], dtype='<u2'),
                         np.array([5, 3, 1, 2, 5], dtype='u1'), probe,
                         np.array([0, 3, 6, 9], dtype='<u8'),
                         np.array([1, 2, 3, 1, 3, 2, 2, 3, 1], dtype='<u2'),
                         np.array([0, 3, 7, 1, 4, 8, 2, 5, 6], dtype='<u2')):
                part.tofile(stream)

    def test_merge_mapping_order_and_unknown_labels(self):
        original_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        destination = self.root / 'production'
        report = create(self.source, self.training, self.qualifying, destination)
        d = load(destination / 'train-probe-qualifying.bin')
        np.testing.assert_array_equal(d['train_offsets'], [0, 2, 4, 6])
        np.testing.assert_array_equal(d['train_items'], [1, 2, 1, 3, 2, 3])
        np.testing.assert_array_equal(d['train_days'], [0, 3, 1, 4, 2, 5])
        np.testing.assert_array_equal(d['train_ratings'], [5, 4, 3, 1, 2, 5])
        np.testing.assert_array_equal(d['probe']['user'], [2, 0, 1])
        np.testing.assert_array_equal(d['probe']['item'], [1, 3, 2])
        np.testing.assert_array_equal(d['probe']['rating'], [0, 0, 0])
        np.testing.assert_array_equal(np.load(destination / 'dense-to-raw-user.npy'), [40, 7, 99])
        order = np.load(destination / 'qualifying-order.npy')
        np.testing.assert_array_equal(order['raw_user'], [99, 40, 7])
        self.assertAlmostEqual(d['mean'], 20 / 6)
        self.assertEqual(report['train_count'], 6)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), original_sha)
        for name, expected in report['outputs'].items():
            self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), expected)
        with self.assertRaises(FileExistsError):
            create(self.source, self.training, self.qualifying, destination)
        with self.assertRaisesRegex(ValueError, 'labeled development'):
            create(destination / 'train-probe-qualifying.bin', self.training, self.qualifying, self.root / 'invalid-input')

    def test_reject_query_date_disagreement(self):
        self.qualifying.write_text(self.qualifying.read_text().replace('1999-01-07', '1999-01-10'))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            create(self.source, self.training, self.qualifying, self.root / 'bad-date')

    def test_reject_missing_or_duplicate_queries(self):
        text = self.qualifying.read_text()
        for name, changed in [('missing', '1:\n99,1999-01-07\n'), ('duplicate', text + '7,1999-01-09\n')]:
            self.qualifying.write_text(changed)
            with self.assertRaisesRegex(ValueError, 'partition|duplicate'):
                create(self.source, self.training, self.qualifying, self.root / name)


if __name__ == '__main__':
    unittest.main()
