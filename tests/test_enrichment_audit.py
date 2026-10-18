"""Verify metadata parsing, binary row alignment and rating independence."""
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import enrichment_audit as audit


class AuditTests(unittest.TestCase):
    def test_counts_and_rating_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            titles, data = pathlib.Path(tmp) / 'titles', pathlib.Path(tmp) / 'data'
            titles.write_bytes('1,2000,Hello, World\n2,NULL,Café: Season 1\n3,2000,Hello World\n'.encode('latin-1'))
            self.assertEqual(audit.read_titles(titles)[1], (2000, 'Hello, World'))
            self.assertEqual(audit.read_titles(titles)[2], (None, 'Café: Season 1'))
            def binary(rating):
                return (audit.HEADER.pack(b'NFPDAT2\0', 1, 3, 2, 1, 3, 1, 3, 3.0)
                        + struct.pack('<2Q', 0, 2) + struct.pack('<2H', 1, 1)
                        + struct.pack('<2H', 1, 2) + bytes([rating, rating])
                        + struct.pack('<IHHB3x', 0, 2, 3, rating)
                        + struct.pack('<2Q', 0, 3) + struct.pack('<3H', 1, 1, 2)
                        + struct.pack('<3H', 1, 2, 3))
            data.write_bytes(binary(1))
            first = audit.audit(titles, data)
            self.assertEqual(first['counted_train_rows'], 2)
            self.assertEqual(first['counted_probe_rows'], 1)
            self.assertEqual(first['missing_year']['probe_fraction'], 1)
            self.assertEqual(first['package_marker']['items'], 1)
            self.assertEqual(first['duplicate_normalized_title_year_groups'], [[1, 3]])
            self.assertEqual(first['item_support']['1']['items'], 2)
            data.write_bytes(binary(5))
            second = audit.audit(titles, data)
            first.pop('inputs'); second.pop('inputs')
            self.assertEqual(first, second)
            data.write_bytes(binary(1)[:-1])
            with self.assertRaises(ValueError):
                audit.audit(titles, data)


if __name__ == '__main__':
    unittest.main()
