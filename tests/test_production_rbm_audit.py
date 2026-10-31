"""Reject incomplete or mismatched native production handoffs before attestation."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_rbm_audit import audit, checkpoint, digest, prepare


class ProductionRbmAuditTests(unittest.TestCase):
    def fixture(self, root, conditional=False):
        data, binary = root / 'data.bin', root / 'binary'
        data.write_bytes(b'dataset')
        binary.write_bytes(b'binary')
        dataset = dict(users=2, items=3, sessions=4, query_count=2, fingerprint=123)
        stages = []
        for index in range(2):
            stages.append(dict(name='stage-' + str(index), output_dir=str(root / str(index)),
                               hidden=4, seed=2026, threads=2, conditional=conditional,
                               start_epoch=1 if index == 0 else 4,
                               fixed_final_epoch=3 if index == 0 else 5,
                               dependency_index=None if index == 0 else 0,
                               command=[str(binary), 'rbm', '--fixed-epoch', '--stage', str(index)]))
        plan = dict(data=str(data), dataset=dataset, stages=stages,
                    frozen={str(data): digest(data), str(binary): digest(binary)})
        path = root / 'plan.json'
        path.write_text(json.dumps(plan))
        return path, plan

    def outputs(self, plan, index):
        stage, data = plan['stages'][index], plan['dataset']
        root = Path(stage['output_dir'])
        hidden, items = stage['hidden'], data['items']
        magic = 0x4E4650524253324D if stage['conditional'] else 0x4E4650524253314D
        body = struct.pack('<QQII', magic, data['fingerprint'], hidden, stage['fixed_final_epoch'])
        counts = [items * 5 * hidden, items * 5, hidden, items * hidden]
        if stage['conditional']:
            counts += [data['users'] * 5, data['sessions'] * 5, items * 8 * 5]
        for i, n in enumerate(counts):
            if i >= 4:
                body += struct.pack('<Q', n)
            body += bytes(n * 4)
        (root / 'model.bin').write_bytes(body)
        for name in ('model.f32', 'reload.f32'):
            np.array([2.0, 4.0], '<f4').tofile(root / name)
        header = f"rbm hidden=4 threads=2 seed=2026 conditional_biases={int(stage['conditional'])} fingerprint=123\n"
        lines = [f'epoch={epoch} cd=1 train_rmse=1 probe_rmse=3 best=3 seconds=1\n'
                 for epoch in range(stage['start_epoch'], stage['fixed_final_epoch'] + 1)]
        (root / 'train.log').write_text(header + ''.join(lines))
        (root / 'reload.log').write_text('probe_rmse=3\n')
        return root

    def test_complete_both_formats_and_audited_resume(self):
        for conditional in (False, True):
            with self.subTest(conditional=conditional), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp), conditional)
                prepare(path, 0)
                self.outputs(plan, 0)
                result = audit(path, 0)
                self.assertEqual(result['state'], 'verified_complete')
                self.assertIsNone(result['metric'])
                child = prepare(path, 1)
                self.assertEqual(len(child['dependencies']), 2)
                self.outputs(plan, 1)
                self.assertEqual(audit(path, 1)['final_epoch'], 5)
                with self.assertRaises(FileExistsError):
                    audit(path, 1)

    def test_checkpoint_damage_never_attested(self):
        for damage in ('epoch', 'fingerprint', 'magic', 'nan', 'truncated', 'trailing', 'conditional_size'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp), True)
                prepare(path, 0)
                root = self.outputs(plan, 0)
                raw = bytearray((root / 'model.bin').read_bytes())
                if damage == 'epoch':
                    struct.pack_into('<I', raw, 20, 2)
                elif damage == 'fingerprint':
                    struct.pack_into('<Q', raw, 8, 456)
                elif damage == 'magic':
                    struct.pack_into('<Q', raw, 0, 0x4E4650524253314D)
                elif damage == 'nan':
                    struct.pack_into('<f', raw, 24, float('nan'))
                elif damage == 'truncated':
                    raw = raw[:-4]
                elif damage == 'trailing':
                    raw += b'x'
                else:
                    first_size = 24 + (3 * 5 * 4 + 3 * 5 + 4 + 3 * 4) * 4
                    struct.pack_into('<Q', raw, first_size, 11)
                (root / 'model.bin').write_bytes(raw)
                with self.assertRaises(ValueError):
                    audit(path, 0)
                self.assertFalse((root / 'audit.json').exists())

    def test_log_recipe_prediction_and_start_mismatches_rejected(self):
        for damage in ('missing_epoch', 'repeated_epoch', 'wrong_threads', 'early_stop',
                       'count', 'nonfinite', 'bounds', 'reload', 'started'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp))
                prepare(path, 0)
                root = self.outputs(plan, 0)
                log = root / 'train.log'
                if damage == 'missing_epoch':
                    log.write_text(log.read_text().replace('epoch=2', 'absent=2'))
                elif damage == 'repeated_epoch':
                    log.write_text(log.read_text() + 'epoch=3 duplicate\n')
                elif damage == 'wrong_threads':
                    log.write_text(log.read_text().replace('threads=2', 'threads=8'))
                elif damage == 'early_stop':
                    log.write_text(log.read_text() + 'early_stop patience=1\n')
                elif damage in ('count', 'nonfinite', 'bounds', 'reload'):
                    values = {'count': [2], 'nonfinite': [2, float('nan')], 'bounds': [2, 6], 'reload': [2, 3]}[damage]
                    np.array(values, '<f4').tofile(root / 'model.f32')
                else:
                    p = root / 'started.json'
                    value = json.loads(p.read_text())
                    value['command'].append('--changed')
                    p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    audit(path, 0)
                self.assertFalse((root / 'audit.json').exists())

    def test_parent_required_and_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, plan = self.fixture(Path(temp))
            with self.assertRaises(FileNotFoundError):
                prepare(path, 1)
            self.assertFalse(Path(plan['stages'][1]['output_dir']).exists())
            prepare(path, 0)
            root = self.outputs(plan, 0)
            audit(path, 0)
            prepare(path, 1)
            child = self.outputs(plan, 1)
            (root / 'model.bin').write_bytes(b'changed parent')
            with self.assertRaises(ValueError):
                audit(path, 1)
            self.assertFalse((child / 'audit.json').exists())

    def test_changed_frozen_input_and_duplicate_prepare_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, plan = self.fixture(Path(temp))
            prepare(path, 0)
            with self.assertRaises(FileExistsError):
                prepare(path, 0)
            Path(plan['data']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                audit(path, 0)


if __name__ == '__main__':
    unittest.main()
