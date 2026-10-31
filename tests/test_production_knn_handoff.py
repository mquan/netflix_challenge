import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_knn_handoff import resolve
from production_rbm_audit import digest


class KnnHandoffTests(unittest.TestCase):
    def fixture(self, root, malformed=None):
        data = root / 'data'
        data.write_bytes(b'data')
        stages = [dict(output_dir=str(root / str(i)), retained_columns=[f'column-{i}']) for i in range(28)]
        stages[0]['retained_columns'].append('duplicate-column')
        columns = [dict(retained=f'column-{i}', producer_stage=i, development_sha256=f'old-{i}') for i in range(28)]
        columns.insert(3, dict(retained='duplicate-column', producer_stage=0, development_sha256='old-0'))
        supports = []
        for i, producer in enumerate([0, 0, 0, 1, 2, 2]):
            supports.append(dict(retained_input=f'support-{i}', original_sha256=f'old-support-{producer}',
                                 producer_stage=producer, output=str(root / str(producer) / 'support.f32')))
        if malformed == 'column':
            columns[0]['producer_stage'] = 1
        elif malformed == 'support':
            supports[0]['output'] = str(root / 'wrong-support.f32')
        elif malformed == 'equivalence':
            supports[1]['producer_stage'] = 4
            supports[1]['output'] = str(root / '4/support.f32')
        destination = root / 'resolved.json'
        plan = dict(stages=stages, column_occurrences=columns, support_aliases=supports,
                    resolved_manifest=str(destination), dataset=dict(query_count=1), master_sha256='master',
                    frozen={str(data): digest(data)})
        path = root / 'plan.json'
        path.write_text(json.dumps(plan))
        for i, stage in enumerate(stages):
            directory = Path(stage['output_dir'])
            directory.mkdir()
            np.array([float(i)], '<f4').tofile(directory / 'model.f32')
            np.array([i + .5], '<f4').tofile(directory / 'support.f32')
            receipt = dict(state='verified_complete', stage=i, plan_sha256=digest(path), metric=None,
                           qualifying_count=1, retained_columns=stage['retained_columns'],
                           outputs={str(directory / n): digest(directory / n) for n in ('model.f32', 'support.f32')})
            (directory / 'audit.json').write_text(json.dumps(receipt))
        return path, plan

    def test_order_and_duplicate_aliases_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path, plan = self.fixture(Path(temp))
            result = resolve(path)
            self.assertEqual([c['retained'] for c in result['columns']], [c['retained'] for c in plan['column_occurrences']])
            self.assertEqual(result['columns'][0]['qualifying_path'], result['columns'][3]['qualifying_path'])
            self.assertEqual(len(result['support_aliases']), 6)
            self.assertIsNone(result['metric'])
            with self.assertRaises(FileExistsError):
                resolve(path)

    def test_partial_or_changed_outputs_never_resolved(self):
        for damage in ('missing', 'partial', 'changed'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, plan = self.fixture(root)
                if damage == 'missing':
                    (root / '27/audit.json').unlink()
                elif damage == 'partial':
                    p = root / '27/audit.json'
                    value = json.loads(p.read_text())
                    value['state'] = 'partial'
                    p.write_text(json.dumps(value))
                else:
                    (root / '27/model.f32').write_bytes(b'changed')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    resolve(path)
                self.assertFalse(Path(plan['resolved_manifest']).exists())

    def test_wrong_producers_or_support_equivalence_rejected(self):
        for malformed in ('column', 'support', 'equivalence'):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as temp:
                path, plan = self.fixture(Path(temp), malformed)
                with self.assertRaises(ValueError):
                    resolve(path)
                self.assertFalse(Path(plan['resolved_manifest']).exists())


if __name__ == '__main__':
    unittest.main()
