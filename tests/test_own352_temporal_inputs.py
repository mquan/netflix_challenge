from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upstream_own352_temporal_data as study


class Temporal352Inputs(unittest.TestCase):
    def test_all_original_context_and_intercept_positions_survive(self):
        z = np.arange(5*406, dtype=np.float64).reshape(5, 406)
        f = np.arange(5*556, dtype=np.float32).reshape(5, 556)
        class Original:
            def __init__(self, rows): pass
            def block(self, lo, hi): return z[lo:hi], f[lo:hi]
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = []; expected = []
            for j in range(22):
                value = np.array([-1, 2, j/10, 8, 4], dtype=np.float32)
                name = '%s.npy' % j; np.save(root / name, value)
                rows.append(dict(paired_files=dict(probex=name)))
                expected.append(np.clip(value, 0, 6))
            with patch.object(study, 'ROWS', 5), patch.object(study.u, 'ROOT', root), patch.object(study.base, 'Inputs', Original):
                reader = study.Inputs([], rows)
                try:
                    actual_z, actual_f = reader.block(1, 5)
                    extra = np.column_stack(expected)[1:5]
                    self.assertEqual(actual_z.shape, (4, 428))
                    self.assertEqual(actual_f.shape, (4, 578))
                    np.testing.assert_array_equal(actual_z[:, :405], z[1:5, :405])
                    np.testing.assert_array_equal(actual_z[:, 405:427], extra)
                    np.testing.assert_array_equal(actual_z[:, -1], z[1:5, -1])
                    np.testing.assert_array_equal(actual_f[:, :556], f[1:5])
                    np.testing.assert_array_equal(actual_f[:, 556:], extra-np.float32(3.6))
                finally:
                    reader.close()
                with self.assertRaises(ValueError):
                    study.Inputs([], rows[:-1])


if __name__ == '__main__':
    unittest.main()
