import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_knn_audit import audit, check_log, prepare
from production_rbm_audit import digest


class KnnAuditTests(unittest.TestCase):
    def test_effective_cli_defaults_use_graph_width(self):
        with tempfile.TemporaryDirectory() as temp:
            graph = Path(temp) / 'graph'
            graph.write_bytes(struct.pack('<QQII', 0x4E46475241504831, 123, 300, 64))
            command = ['native', 'knn', '--graph', str(graph)]
            check_log('knn base=time neighbors=64 power=2 shrink=0.05 time_decay=0 probe_rmse=3\n', command)
            with self.assertRaises(ValueError):
                check_log('knn base=time neighbors=256 power=2 shrink=0.05 time_decay=0 probe_rmse=3\n', command)

    def fixture(self, root, joint=False):
        data, binary = root / 'data', root / 'binary'
        data.write_bytes(b'data')
        binary.write_bytes(b'binary')
        dataset = dict(query_count=2, fingerprint=123)
        external_stages = [dict(kind='baseline', output_dir=str(root / 'base'), fixed_final_epoch=2),
                           dict(kind='graph64', output_dir=str(root / 'graph'))]
        external = root / 'external.json'
        external.write_text(json.dumps(dict(data=str(data), dataset=dataset, stages=external_stages)))
        targets = [root / 'base/model.bin', root / 'graph/derived/neighbors64.bin']
        for i, target in enumerate(targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'new-phase-output-' + bytes([i]))
            receipt = dict(state='verified_complete', stage=i, kind='baseline' if i == 0 else 'graph64',
                           plan_sha256=digest(external), outputs={str(target): digest(target)}, metric=None,
                           final_epoch=2)
            (Path(external_stages[i]['output_dir']) / 'audit.json').write_text(json.dumps(receipt))
        phase = dict(plan=str(external), plan_sha256=digest(external))
        output = root / 'knn'
        command = [str(binary), 'knn', '--data', str(data), '--graph', str(targets[1]),
                   '--joint-model' if joint else '--time-model', str(targets[0]),
                   '--neighbors', '64', '--power', '1', '--shrink', '.5', '--time-decay', '.1',
                   '--predictions', str(output / 'model.f32'), '--support', str(output / 'support.f32')]
        stage = dict(output_dir=str(output), command=command, retained_columns=['alias-one', 'alias-two'],
                     base_dependency=dict(phase='joint' if joint else 'dependencies', stage=0, model=str(targets[0])),
                     graph_dependency=dict(stage=1, graph=str(targets[1])))
        plan = dict(data=str(data), dataset=dataset, stages=[stage], source_phases=dict(joint=phase, dependencies=phase),
                    frozen={str(p): digest(p) for p in (data, binary, external)})
        path = root / 'plan.json'
        path.write_text(json.dumps(plan))
        return path, plan

    def outputs(self, plan):
        stage = plan['stages'][0]
        root = Path(stage['output_dir'])
        for name in ('model.f32', 'replay.f32'):
            np.array([-.5, 5.5], '<f4').tofile(root / name)
        for name in ('support.f32', 'replay-support.f32'):
            np.array([0, .7], '<f4').tofile(root / name)
        command = stage['command']
        base = command[command.index('--joint-model') + 1] if '--joint-model' in command else 'time'
        line = f'knn base={base} neighbors=64 power=1 shrink=0.5 time_decay=0.1000000015 probe_rmse=3\n'
        for name in ('run.log', 'replay.log'):
            (root / name).write_text(line)
        return root

    def test_complete_both_base_types_and_duplicate_attestation_rejected(self):
        for joint in (False, True):
            with self.subTest(joint=joint), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp), joint)
                prepare(path, 0)
                self.outputs(plan)
                result = audit(path, 0)
                self.assertEqual(result['retained_columns'], ['alias-one', 'alias-two'])
                self.assertIsNone(result['metric'])
                self.assertEqual(result['support_min'], 0)
                with self.assertRaises(FileExistsError):
                    audit(path, 0)

    def test_invalid_arrays_and_replay_mismatches_rejected(self):
        for damage in ('count', 'trailing_byte', 'prediction_nan', 'support_nan', 'negative_support', 'prediction_replay', 'support_replay'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp))
                prepare(path, 0)
                root = self.outputs(plan)
                if damage == 'count':
                    np.array([1], '<f4').tofile(root / 'model.f32')
                elif damage == 'trailing_byte':
                    with (root / 'support.f32').open('ab') as f:
                        f.write(b'x')
                elif damage == 'prediction_nan':
                    np.array([1, float('nan')], '<f4').tofile(root / 'model.f32')
                elif damage == 'support_nan':
                    np.array([0, float('inf')], '<f4').tofile(root / 'support.f32')
                elif damage == 'negative_support':
                    np.array([0, -1], '<f4').tofile(root / 'support.f32')
                elif damage == 'prediction_replay':
                    np.array([1, 2], '<f4').tofile(root / 'replay.f32')
                else:
                    np.array([0, 1], '<f4').tofile(root / 'replay-support.f32')
                with self.assertRaises(ValueError):
                    audit(path, 0)
                self.assertFalse((root / 'audit.json').exists())

    def test_changed_missing_or_unbound_parent_blocks_start(self):
        for damage in ('unbound', 'changed', 'wrong_stage', 'query_score'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp))
                parent = Path(temp) / 'base/audit.json'
                record = json.loads(parent.read_text())
                if damage == 'unbound':
                    record['outputs'] = {}
                elif damage == 'wrong_stage':
                    record['stage'] = 99
                elif damage == 'query_score':
                    record['metric'] = 1
                else:
                    (Path(temp) / 'base/model.bin').write_bytes(b'changed')
                parent.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    prepare(path, 0)
                self.assertFalse(Path(plan['stages'][0]['output_dir']).exists())

    def test_dependency_changes_during_inference_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, plan = self.fixture(Path(temp), True)
            prepare(path, 0)
            root = self.outputs(plan)
            (Path(temp) / 'graph/derived/neighbors64.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                audit(path, 0)
            self.assertFalse((root / 'audit.json').exists())

    def test_wrong_log_hyperparameter_or_missing_completion_rejected(self):
        for damage in ('wrong', 'missing', 'duplicate'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp))
                prepare(path, 0)
                root = self.outputs(plan)
                log = root / 'run.log'
                body = log.read_text()
                log.write_text(body.replace('time_decay=0.1000000015', 'time_decay=.2') if damage == 'wrong' else '' if damage == 'missing' else body + body)
                with self.assertRaises(ValueError):
                    audit(path, 0)


if __name__ == '__main__':
    unittest.main()
