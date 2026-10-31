import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_joint_audit import audit, checkpoint, graph_fingerprint, prepare, sidecar
from production_rbm_audit import digest


class JointAuditTests(unittest.TestCase):
    def fixture(self, root, integrated=False, day_upgrade=False):
        data, binary = root / 'data', root / 'binary'
        data.write_bytes(b'data')
        binary.write_bytes(b'binary')
        dataset = dict(users=2, items=3, sessions=4, query_count=2, fingerprint=123)
        ids = np.array([[2, 1], [2, 0], [1, 0]], '<u2')
        scores = np.tile(np.array([.5, .25], '<f4'), (3, 1))
        graph_body = struct.pack('<QQII', 0x4E46475241504831, 123, 3, 2) + ids.tobytes() + scores.tobytes()
        graph_path = root / 'graph.bin'
        graph_path.write_bytes(graph_body)
        external_path = root / 'external.json'
        external_path.write_text(json.dumps(dict(data=str(data), dataset=dataset)))
        graph_audit = root / 'graph-audit.json'
        graph_audit.write_text(json.dumps(dict(state='verified_complete', stage=2, kind='graph64',
            metric=None, plan_sha256=digest(external_path), outputs={str(graph_path): digest(graph_path)})))
        stages = []
        for i in range(2):
            stages.append(dict(name=f'stage-{i}', family='integrated' if integrated else 'joint',
                output_dir=str(root / str(i)), command=[str(binary), 'train', str(i)], rank=2,
                model_format=2 if integrated else 3, day_rank=2 if day_upgrade and i else 0,
                seed=2026, dependency_index=None if i == 0 else 0,
                start_epoch=1 if i == 0 or day_upgrade else 3,
                fixed_final_epoch=1 if i == 1 and day_upgrade else 2 if i == 0 else 4))
        plan = dict(data=str(data), dataset=dataset, stages=stages,
                    graph_dependency=dict(plan=str(external_path), plan_sha256=digest(external_path),
                                          stage=2, audit=str(graph_audit), graph=str(graph_path), neighbors=2),
                    frozen={str(data): digest(data), str(binary): digest(binary), str(external_path): digest(external_path)})
        path = root / 'plan.json'
        path.write_text(json.dumps(plan))
        return path, plan, graph_body

    def outputs(self, plan, index, graph_body):
        stage, d = plan['stages'][index], plan['dataset']
        root = Path(stage['output_dir'])
        u, i, s, rank = d['users'], d['items'], d['sessions'], stage['rank']
        counts = [u, u, u, i, i * 30, i * 8, s, s, u * rank, i * rank, i * rank, u * rank]
        body = struct.pack('<QQII', 0x4E46504A4F494E30 + stage['model_format'], d['fingerprint'], rank, stage['fixed_final_epoch'])
        for count in counts:
            body += struct.pack('<Q', count) + bytes(count * 4)
        body += struct.pack('<IQ', stage['day_rank'], s * stage['day_rank']) + bytes(s * stage['day_rank'] * 4)
        if stage['model_format'] == 3:
            body += bytes(8)
        (root / 'model.bin').write_bytes(body)
        for name in ('model.f32', 'reload.f32'):
            np.array([2, 4], '<f4').tofile(root / name)
        if stage['family'] == 'integrated':
            (root / 'model.bin.nbr').write_bytes(struct.pack('<QQ', 0x4E46494E54474E31, graph_fingerprint(graph_body)) + bytes(3 * 2 * 2 * 4))
            header = 'integrated factors=2 seed=2026 k=2 init_model=\n'
        else:
            header = f"joint factors=2 seed=2026 fingerprint=123\nday_factors={stage['day_rank']} init_model=\n"
        (root / 'train.log').write_text(header + ''.join(f'epoch={n} train_rmse=1 probe_rmse=3\n'
            for n in range(stage['start_epoch'], stage['fixed_final_epoch'] + 1)))
        (root / 'reload.log').write_text('loaded')
        return root

    def test_complete_joint_integrated_and_epoch_reset_upgrade(self):
        for integrated, upgrade in [(False, False), (True, False), (False, True)]:
            with self.subTest(integrated=integrated, upgrade=upgrade), tempfile.TemporaryDirectory() as temp:
                path, plan, body = self.fixture(Path(temp), integrated, upgrade)
                for index in (0, 1):
                    prepare(path, index)
                    self.outputs(plan, index, body)
                    result = audit(path, index)
                    self.assertEqual(result['state'], 'verified_complete')
                    self.assertIsNone(result['metric'])
                if integrated:
                    self.assertIn(str(Path(plan['stages'][0]['output_dir']) / 'model.bin.nbr'), result['dependencies'])

    def test_checkpoint_corruption(self):
        for damage in ('epoch', 'dataset', 'rank', 'size', 'nonfinite', 'truncated', 'trailing', 'day_rank', 'frequency'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan, graph = self.fixture(Path(temp))
                prepare(path, 0)
                root = self.outputs(plan, 0, graph)
                body = bytearray((root / 'model.bin').read_bytes())
                if damage == 'epoch':
                    struct.pack_into('<I', body, 20, 1)
                elif damage == 'dataset':
                    struct.pack_into('<Q', body, 8, 456)
                elif damage == 'rank':
                    struct.pack_into('<I', body, 16, 3)
                elif damage == 'size':
                    struct.pack_into('<Q', body, 24, 999)
                elif damage == 'nonfinite':
                    struct.pack_into('<f', body, 32, float('nan'))
                elif damage == 'truncated':
                    body = body[:-4]
                elif damage == 'trailing':
                    body += b'x'
                elif damage == 'day_rank':
                    struct.pack_into('<I', body, len(body) - 20, 1)
                else:
                    struct.pack_into('<Q', body, len(body) - 8, 1)
                (root / 'model.bin').write_bytes(body)
                with self.assertRaises(ValueError):
                    audit(path, 0)
                self.assertFalse((root / 'audit.json').exists())

    def test_missing_unbound_or_mutated_resume_sidecar(self):
        for damage in ('missing', 'unbound', 'mutated_after_prepare'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan, graph = self.fixture(Path(temp), True)
                prepare(path, 0)
                parent = self.outputs(plan, 0, graph)
                audit(path, 0)
                if damage == 'missing':
                    (parent / 'model.bin.nbr').unlink()
                    with self.assertRaises(FileNotFoundError):
                        prepare(path, 1)
                elif damage == 'unbound':
                    receipt = json.loads((parent / 'audit.json').read_text())
                    del receipt['outputs'][str(parent / 'model.bin.nbr')]
                    (parent / 'audit.json').write_text(json.dumps(receipt))
                    with self.assertRaises(ValueError):
                        prepare(path, 1)
                else:
                    prepare(path, 1)
                    child = self.outputs(plan, 1, graph)
                    (parent / 'model.bin.nbr').write_bytes(b'changed')
                    with self.assertRaises(ValueError):
                        audit(path, 1)
                    self.assertFalse((child / 'audit.json').exists())

    def test_sidecar_shape_finiteness_and_graph_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            path, plan, graph = self.fixture(Path(temp), True)
            prepare(path, 0)
            root = self.outputs(plan, 0, graph)
            original = (root / 'model.bin.nbr').read_bytes()
            for damage in ('graph', 'nonfinite', 'truncated', 'trailing'):
                body = bytearray(original)
                if damage == 'graph':
                    struct.pack_into('<Q', body, 8, 999)
                elif damage == 'nonfinite':
                    struct.pack_into('<f', body, 16, float('inf'))
                elif damage == 'truncated':
                    body = body[:-4]
                else:
                    body += b'x'
                (root / 'model.bin.nbr').write_bytes(body)
                with self.assertRaises(ValueError):
                    sidecar(root / 'model.bin.nbr', graph)

    def test_incomplete_epochs_and_invalid_predictions(self):
        for damage in ('epochs', 'reload', 'count', 'nonfinite', 'header'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan, graph = self.fixture(Path(temp))
                prepare(path, 0)
                root = self.outputs(plan, 0, graph)
                if damage == 'epochs':
                    log = root / 'train.log'
                    log.write_text(log.read_text().replace('epoch=2', 'missing=2'))
                elif damage == 'header':
                    log = root / 'train.log'
                    log.write_text(log.read_text().replace('seed=2026', 'seed=7'))
                else:
                    values = {'reload': [2, 3], 'count': [2], 'nonfinite': [2, float('nan')]}[damage]
                    np.array(values, '<f4').tofile(root / 'model.f32')
                with self.assertRaises(ValueError):
                    audit(path, 0)
                self.assertFalse((root / 'audit.json').exists())


if __name__ == '__main__':
    unittest.main()
