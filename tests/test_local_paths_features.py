"""Check full-column export order, parallel determinism and label isolation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gbdt_blend import PROBE
from local_paths_features import POOLED_NAMES, PATH_NAMES, export_features
from local_paths_pilot import TrainingIndex, summarize
from test_local_paths_pilot import fixture


class ExportTests(unittest.TestCase):
    def test_order_parallelism_and_label_isolation(self):
        d = fixture()
        d['probe'] = np.array([(0, 3, 6, 4, b'\0\0\0'), (2, 2, 7, 1, b'\0\0\0'),
                               (3, 1, 3, 5, b'\0\0\0')], dtype=PROBE)
        settings = dict(history=32, neighbors=32, candidates=256, seed=20261013)
        with tempfile.TemporaryDirectory() as tmp:
            first = export_features(d, Path(tmp) / 'serial', settings=settings, processes=1, chunk_size=1)
            d['probe']['rating'] = 6 - d['probe']['rating']
            second = export_features(d, Path(tmp) / 'parallel', settings=settings, processes=2, chunk_size=1)
            x, z = [np.load(Path(tmp) / name / 'features.npy') for name in ('serial', 'parallel')]
            np.testing.assert_array_equal(x, z)
            self.assertEqual(x.shape, (3, 160))
            index = TrainingIndex(d)
            for k, row in enumerate(d['probe']):
                graph = index.retrieve(int(row['user']), int(row['item']), int(row['day']), **settings)
                pooled, paths, _ = summarize(d, graph, int(row['day']))
                np.testing.assert_array_equal(x[k], np.r_[pooled, paths].astype('<f4'))
            for group, width in [('pooled', len(POOLED_NAMES)), ('paths', len(PATH_NAMES))]:
                self.assertEqual(len(first['features'][group]), width)
                for a, b in zip(first['features'][group], second['features'][group]):
                    self.assertEqual(first['outputs'][a], second['outputs'][b])
                    self.assertEqual(Path(a).stat().st_size, 3 * 4)
            with self.assertRaises(FileExistsError):
                export_features(d, Path(tmp) / 'serial', settings=settings)


if __name__ == '__main__':
    unittest.main()
