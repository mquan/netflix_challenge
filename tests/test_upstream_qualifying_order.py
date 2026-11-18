from pathlib import Path
import tempfile
import unittest
import numpy as np
from verify_upstream_qualifying_order import verify_order


class QualifyingOrderChecks(unittest.TestCase):
    def test_rejects_reordered_rows_and_each_changed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'qualifying.txt'
            path.write_text('3:\n99,2000-02-29\n7,2005-01-01\n1:\n40,1999-11-11\n')
            raw = np.array([7, 40, 99])
            users = np.array([2, 0, 1])
            items = np.array([2, 2, 0])
            dates = np.array([110, 1878, 0])
            self.assertEqual(verify_order(path, raw, users, items, dates), 3)
            for column in range(3):
                arrays = [users.copy(), items.copy(), dates.copy()]
                arrays[column][0] -= 1
                with self.assertRaisesRegex(ValueError, 'identity/order mismatch'):
                    verify_order(path, raw, *arrays)
            with self.assertRaisesRegex(ValueError, 'identity/order mismatch'):
                verify_order(path, raw, users[::-1], items[::-1], dates[::-1])
            with self.assertRaisesRegex(ValueError, 'counts differ'):
                verify_order(path, raw, np.r_[users, 0], np.r_[items, 0], np.r_[dates, 0])


if __name__ == '__main__':
    unittest.main()
