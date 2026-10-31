import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_dependency_audit import audit, baseline, graph, graph_fingerprint, neighborhood, prepare
from production_rbm_audit import digest


class DependencyAuditTests(unittest.TestCase):
    def dataset(self):
        return dict(users=2, items=3, sessions=4, mean=3.5, query_count=2, fingerprint=123)

    def baseline_bytes(self):
        d = self.dataset()
        return struct.pack('<8sIIQI4xd', b'NFPTIM1\0', 2, 3, 4, 40, 3.5) + bytes((2 * 3 + 3 * 39 + 4 * 2) * 4)

    def graph_bytes(self):
        ids = np.array([[2, 1], [2, 0], [1, 0]], '<u2')
        scores = np.tile(np.array([.5, .25], '<f4'), (3, 1))
        return struct.pack('<QQII', 0x4E46475241504831, 123, 3, 2) + ids.tobytes() + scores.tobytes()

    def test_baseline_layout_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'baseline'
            original = self.baseline_bytes()
            path.write_bytes(original)
            baseline(path, self.dataset(), 40)
            for damage in ('sessions', 'iterations', 'mean', 'nan', 'truncated', 'trailing'):
                with self.subTest(damage=damage):
                    body = bytearray(original)
                    if damage == 'sessions':
                        struct.pack_into('<Q', body, 16, 999)
                    elif damage == 'iterations':
                        struct.pack_into('<I', body, 24, 39)
                    elif damage == 'mean':
                        struct.pack_into('<d', body, 32, 4.0)
                    elif damage == 'nan':
                        struct.pack_into('<f', body, 40, float('nan'))
                    elif damage == 'truncated':
                        body = body[:-1]
                    else:
                        body += b'x'
                    path.write_bytes(body)
                    with self.assertRaises(ValueError):
                        baseline(path, self.dataset(), 40)

    def test_graph_identity_and_neighbor_validity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'graph'
            body = self.graph_bytes()
            path.write_bytes(body)
            self.assertEqual(graph(path, self.dataset(), 2), body)
            wrong = dict(self.dataset(), fingerprint=456)
            with self.assertRaises(ValueError):
                graph(path, wrong, 2)
            damaged = bytearray(body)
            struct.pack_into('<H', damaged, 24, 0)
            path.write_bytes(damaged)
            with self.assertRaises(ValueError):
                graph(path, self.dataset(), 2)

    def test_neighborhood_binds_graph_dataset_epoch_and_decay(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'neighborhood'
            g = self.graph_bytes()
            original = struct.pack('<QQQIIIf', 0x4E464E42484F4F32, 123, graph_fingerprint(g), 4, 3, 2, 0) + bytes(3 * 2 * 2 * 4)
            path.write_bytes(original)
            neighborhood(path, self.dataset(), g, 4)
            for offset, fmt, value in [(8, '<Q', 999), (16, '<Q', 999), (24, '<I', 3),
                                       (36, '<f', .1), (40, '<f', float('inf'))]:
                damaged = bytearray(original)
                struct.pack_into(fmt, damaged, offset, value)
                path.write_bytes(damaged)
                with self.assertRaises(ValueError):
                    neighborhood(path, self.dataset(), g, 4)

    def test_baseline_handoff_and_mutated_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.write_bytes(b'data')
            stages = [dict(kind='baseline', output_dir=str(root / str(i)),
                           commands=[['native', 'time-baseline']], dependency_indices=[] if i == 0 else [0],
                           files=['model.bin', 'train.log', 'reload.log']) for i in range(2)]
            plan = dict(data=str(data), dataset=self.dataset(), stages=stages, frozen={str(data): digest(data)})
            path = root / 'plan.json'
            path.write_text(json.dumps(plan))
            with self.assertRaises(FileNotFoundError):
                prepare(path, 1)
            prepare(path, 0)
            output = Path(stages[0]['output_dir'])
            (output / 'model.bin').write_bytes(self.baseline_bytes())
            (output / 'train.log').write_text(''.join(f'iteration={i} train_rmse=1 probe_rmse=3\n' for i in range(1, 41)))
            (output / 'reload.log').write_text('loaded')
            self.assertIsNone(audit(path, 0)['metric'])
            prepare(path, 1)
            (output / 'model.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                audit(path, 1)
            self.assertFalse((Path(stages[1]['output_dir']) / 'audit.json').exists())


if __name__ == '__main__':
    unittest.main()
