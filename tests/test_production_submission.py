"""Submission serialization preserves movie headers and raw qualifying identities."""
import datetime
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from production_rbm_audit import digest
from production_submission import ORDER, QUERY, verify_order, write_submission
from test_native_fixed_epochs import query_labels
from test_user_knn import write_dataset


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / 'train-probe-qualifying.bin'
        _, _, train = write_dataset(self.data)
        query_labels(self.data, 0)
        self.offset = 56 + 8 * 81 + 5 * train
        q = np.memmap(self.data, dtype=QUERY, mode='r', offset=self.offset, shape=(80,))
        inverse = np.arange(1001, 1081, dtype='<u4')
        np.save(self.root / 'dense-to-raw-user.npy', inverse)
        order = np.zeros(80, ORDER)
        order['raw_user'], order['item'], order['day'] = inverse[q['user']], q['item'], q['day']
        np.save(self.root / 'qualifying-order.npy', order)
        self.raw = self.root / 'qualifying.txt'
        text, movie = [], None
        for row in order:
            if movie != row['item']:
                movie = int(row['item'])
                text.append(f'{movie}:')
            date = datetime.date(1999, 1, 1) + datetime.timedelta(days=int(row['day']))
            text.append(f'{row["raw_user"]},{date.isoformat()}')
        self.raw.write_text('\n'.join(text) + '\n')
        self.manifest = dict(state='complete', purpose='production_only_unlabeled_queries', users=80,
                             items=8, qualifying_count=80, qualifying=str(self.raw),
                             qualifying_sha256=digest(self.raw), outputs={})
        self.path = self.root / 'manifest.json'
        self.refresh()
        self.pred = self.root / 'predictions.f32'
        np.linspace(1, 5, 80, dtype='<f4').tofile(self.pred)
        self.output = self.root / 'submission.txt'

    def refresh(self):
        for name in ('train-probe-qualifying.bin', 'dense-to-raw-user.npy', 'qualifying-order.npy'):
            self.manifest['outputs'][name] = digest(self.root / name)
        self.manifest['qualifying_sha256'] = digest(self.raw)
        self.path.write_text(json.dumps(self.manifest))

    def test_raw_order_and_decimal_float32_roundtrip(self):
        verified = verify_order(self.path)
        self.assertEqual(verified['rows'], 80)
        receipt = write_submission(self.path, self.pred, self.output)
        self.assertEqual(receipt['rows'], 80)
        self.assertIsNone(receipt['metric'])
        raw, emitted = self.raw.read_text().splitlines(), self.output.read_text().splitlines()
        self.assertEqual(len(raw), len(emitted))
        ratings = []
        for before, after in zip(raw, emitted):
            if before.endswith(':'):
                self.assertEqual(before, after)
            else:
                ratings.append(float(after))
        np.testing.assert_array_equal(np.array(ratings, '<f4'), np.fromfile(self.pred, '<f4'))
        self.assertFalse(self.output.with_name('submission.txt.partial').exists())
        with self.assertRaises(FileExistsError):
            write_submission(self.path, self.pred, self.output)

    def test_raw_row_reorder_is_rejected_even_with_updated_digest(self):
        text = self.raw.read_text().splitlines()
        # Alter a user's raw identity without changing native rows or row count.
        index = next(i for i, line in enumerate(text) if ',' in line)
        text[index] = '99999,' + text[index].split(',')[1]
        self.raw.write_text('\n'.join(text) + '\n')
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'row order'):
            verify_order(self.path)

    def test_native_unknown_labels_and_dense_raw_mapping_are_required(self):
        body = bytearray(self.data.read_bytes())
        body[self.offset + 8] = 5
        self.data.write_bytes(body)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'unknown labels'):
            verify_order(self.path)
        body[self.offset + 8] = 0
        self.data.write_bytes(body)
        inverse = np.load(self.root / 'dense-to-raw-user.npy')
        inverse += 1
        np.save(self.root / 'dense-to-raw-user.npy', inverse)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'identity'):
            verify_order(self.path)

    def test_invalid_prediction_count_bounds_or_finiteness_never_publish(self):
        original = self.pred.read_bytes()
        for changed in (original[:-4], struct.pack('<f', float('nan')) + original[4:],
                        struct.pack('<f', .9) + original[4:], struct.pack('<f', 5.1) + original[4:]):
            self.pred.write_bytes(changed)
            with self.assertRaises(ValueError):
                write_submission(self.path, self.pred, self.output)
            self.assertFalse(self.output.exists())
            self.assertFalse(self.output.with_name('submission.txt.partial').exists())

    def test_changed_bound_input_is_rejected(self):
        self.raw.write_text(self.raw.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_order(self.path)


if __name__ == '__main__':
    unittest.main()
