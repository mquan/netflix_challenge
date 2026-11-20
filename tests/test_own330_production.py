import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upstream_own330_production_inputs as data
import upstream_own330_refit_budget_gate as budget


class Own330Production(unittest.TestCase):
    def test_complete_transport_keeps_context_and_all_prediction_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oldz = np.arange(5*183, dtype=np.float64).reshape(5, 183)
            oldf = np.arange(5*333, dtype=np.float32).reshape(5, 333)
            np.save(root / 'probex-design.npy', oldz)
            np.save(root / 'probex-features.npy', oldf)
            rows = []
            values = []
            for j in range(223):
                a = np.asarray([-1, 2, 8, j/100, 5], dtype=np.float32)
                name = 'raw-%s.npy' % j
                np.save(root / name, a)
                values.append(np.clip(a, 0, 6))
                rows.append(dict(paired_files=dict(probex=name)))
            with patch.object(data.u, 'ROOT', root), patch.object(data.parent, 'RUN', root), patch.dict(data.COUNTS, probex=5):
                reader = data.Inputs('probex', rows)
                try:
                    z, f = reader.block(1, 5)
                    extra = np.column_stack(values)[1:5]
                    np.testing.assert_array_equal(z[:, :182], oldz[1:5, :182])
                    np.testing.assert_array_equal(z[:, 182:405], extra)
                    np.testing.assert_array_equal(z[:, -1], oldz[1:5, -1])
                    np.testing.assert_array_equal(f[:, :243], oldf[1:5, :243])
                    np.testing.assert_array_equal(f[:, 243:302], extra[:, :59]-np.float32(3.6))
                    np.testing.assert_array_equal(f[:, 302:392], oldf[1:5, 243:])
                    np.testing.assert_array_equal(f[:, 392:], extra[:, 59:]-np.float32(3.6))
                finally:
                    reader.close()

    def original_budget(self):
        records = [dict(fold=i, receipt_sha256=str(i), outputs={'model':str(i)}) for i in range(5)]
        rows = [(r, dict(fold=r['fold'], manifest_sha256='manifest', outputs=r['outputs'],
                        chosen_iterations=[5+r['fold'], 7, 8]), r['receipt_sha256']) for r in records]
        return records, rows

    def test_original_selected_medians(self):
        records, rows = self.original_budget()
        self.assertEqual(budget.validate(records, rows, 'manifest'), [7, 7, 8])

    def test_receipt_changed_since_original_replay_rejected(self):
        records, rows = self.original_budget()
        rows[0] = (rows[0][0], rows[0][1], 'new-current-hash')
        with self.assertRaises(ValueError):
            budget.validate(records, rows, 'manifest')

    def test_invalid_budget_and_duplicate_fold_rejected(self):
        records, rows = self.original_budget()
        invalid = copy.deepcopy(rows); invalid[0][1]['chosen_iterations'][0] = True
        with self.assertRaises(ValueError):
            budget.validate(records, invalid, 'manifest')
        records[1]['fold'] = 0
        with self.assertRaises(ValueError):
            budget.validate(records, rows, 'manifest')


if __name__ == '__main__':
    unittest.main()
