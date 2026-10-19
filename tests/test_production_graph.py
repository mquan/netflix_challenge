import pathlib
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from production_graph import MAGIC, prefix_graph


class ProductionGraphTests(unittest.TestCase):
    def graph(self):
        ids = np.array([[3, 2, 1], [3, 2, 0], [3, 1, 0], [2, 1, 0]], dtype='<u2')
        scores = np.tile(np.array([.9, .5, .5], dtype='<f4'), (4, 1))
        return struct.pack('<QQII', MAGIC, 12345, 4, 3) + ids.tobytes() + scores.tobytes()

    def test_prefix_preserves_native_layout_fingerprint_and_tie_order(self):
        body, info = prefix_graph(self.graph(), 2)
        self.assertEqual(struct.unpack('<QQII', body[:24]), (MAGIC, 12345, 4, 2))
        self.assertEqual(info['dataset_fingerprint_hex'], '0000000000003039')
        np.testing.assert_array_equal(np.frombuffer(body, dtype='<u2', count=8, offset=24),
                                      [3, 2, 3, 2, 3, 1, 2, 1])
        np.testing.assert_array_equal(np.frombuffer(body, dtype='<f4', offset=40),
                                      np.tile(np.array([.9, .5], dtype='<f4'), 4))
        self.assertEqual(prefix_graph(self.graph(), 3)[0], self.graph())

    def test_rejects_corrupt_or_unsorted_graph_and_invalid_count(self):
        for body, count in [(self.graph()[:-1], 2), (b'x', 2), (self.graph(), 0), (self.graph(), 4)]:
            with self.assertRaises(ValueError):
                prefix_graph(body, count)
        for offset, replacement in [(0, struct.pack('<Q', 0)), (24, struct.pack('<H', 0)),
                                    (26, struct.pack('<H', 3)), (48, struct.pack('<f', float('nan'))),
                                    (48, struct.pack('<f', .1))]:
            body = bytearray(self.graph())
            body[offset:offset + len(replacement)] = replacement
            with self.assertRaises(ValueError):
                prefix_graph(body, 2)


if __name__ == '__main__':
    unittest.main()
