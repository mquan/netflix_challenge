"""Complete-bank handoff contracts with 82 synthetic, hash-bound stage receipts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from production_rbm_audit import digest
from resolve_production_bank import COUNTS, expected_receipt, resolve
from test_user_knn import write_dataset
from test_native_fixed_epochs import query_labels


class ResolveBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        data = self.root / 'data.bin'
        write_dataset(data)
        query_labels(data, 0)
        self.bank = json.loads((ROOT / 'experiments/production-blender-plan.json').read_text())
        self.bank['qualifying']['dataset'] = dict(path=str(data), sha256=digest(data))
        self.bank['qualifying']['rows'] = 80
        self.bank['sources'] = {}
        ready = self.root / 'ready.f32'
        np.zeros(80, '<f4').tofile(ready)
        for group in ('auxiliary', 'context'):
            for index, entry in enumerate(self.bank['qualifying'][group]):
                if not entry.get('pending'):
                    self.bank['qualifying'][group][index] = dict(name=entry['name'], path=str(ready), sha256=digest(ready))
        self.bank_path = self.root / 'bank.json'
        self.bank_path.write_text(json.dumps(self.bank))
        self.mapping = json.loads((ROOT / 'experiments/P0-all-family-planned-coverage-20261031.json').read_text())
        self.mapping['frozen'] = {str(self.bank_path): digest(self.bank_path)}
        self.plans, self.audits = {}, []
        for name in COUNTS:
            source = self.mapping['phases'][name]
            plan = json.loads(Path(source['plan']).read_text())
            plan['data'] = str(data)
            plan['frozen'] = {str(data): digest(data)}
            for index, stage in enumerate(plan['stages']):
                target = self.root / name / str(index)
                target.mkdir(parents=True)
                stage['output_dir'] = str(target)
            path = self.root / (name + '-plan.json')
            path.write_text(json.dumps(plan))
            self.mapping['phases'][name] = dict(plan=str(path), plan_sha256=digest(path), stages=len(plan['stages']))
            self.mapping['frozen'][str(path)] = digest(path)
            self.plans[name] = plan
        produced = {}
        for entry in self.mapping['predictions'] + self.mapping['support_aliases']:
            name, index = entry['phase'], entry['stage']
            root = Path(self.plans[name]['stages'][index]['output_dir'])
            # Preserve repeated aliases by keeping the original output filename.
            target = root / Path(entry['planned_output']).name
            if not target.exists():
                np.full(80, 3.25, '<f4').tofile(target)
            spec = self.mapping['phases'][name]
            entry.update(plan=spec['plan'], plan_sha256=spec['plan_sha256'], planned_output=str(target),
                         required_audit=str(root / 'audit.json'))
            produced.setdefault((name, index), {})[str(target)] = digest(target)
        for name, plan in self.plans.items():
            for index, stage in enumerate(plan['stages']):
                root = Path(stage['output_dir'])
                marker = root / 'native-state.bin'
                marker.write_bytes(b'fixture-state')
                outputs = dict(produced.get((name, index), {}), **{str(marker): digest(marker)})
                # Receipt protocol fields come from the native phase-audit contracts;
                # these tests exercise the final resolver, not native training.
                record = dict(expected_receipt(name, plan, index, 80),
                              plan_sha256=self.mapping['phases'][name]['plan_sha256'], outputs=outputs, dependencies={})
                audit = root / 'audit.json'
                audit.write_text(json.dumps(record))
                self.audits.append(audit)
        self.mapping_path = self.root / 'mapping.json'
        self.mapping_path.write_text(json.dumps(self.mapping))
        self.output = self.root / 'resolved-bank.json'

    def test_all_87_inputs_resolve_in_order_and_frozen_features_remain(self):
        result = resolve(self.mapping_path, self.bank_path, self.output)
        self.assertEqual(len(result['production_provenance']['stage_audits']), 82)
        self.assertEqual(result['development'], self.bank['development'])
        self.assertEqual(result['settings'], self.bank['settings'])
        self.assertEqual(result['schema'], self.bank['schema'])
        for group in ('predictions', 'auxiliary', 'context'):
            for before, after in zip(self.bank['qualifying'][group], result['qualifying'][group]):
                self.assertEqual(before['name'], after['name'])
                self.assertFalse(after.get('pending', False))
                if not before.get('pending'):
                    self.assertEqual(before, after)
        with self.assertRaises(FileExistsError):
            resolve(self.mapping_path, self.bank_path, self.output)

    def test_missing_or_wrong_audit_never_writes_partial_bank(self):
        path = self.audits[-1]
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            resolve(self.mapping_path, self.bank_path, self.output)
        for key, value in [('state', 'partial'), ('metric', 0), ('stage', 9), ('plan_sha256', 'wrong'),
                           ('qualifying_count', 79), ('name', 'wrong')]:
            record = json.loads(original)
            record[key] = value
            path.write_text(json.dumps(record))
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve(self.mapping_path, self.bank_path, self.output)
            self.assertFalse(self.output.exists())

    def test_mutated_output_and_unbound_prediction_are_rejected(self):
        entry = self.mapping['predictions'][0]
        target = Path(entry['planned_output'])
        original = target.read_bytes()
        target.write_bytes(original[:-4])
        with self.assertRaises(ValueError):
            resolve(self.mapping_path, self.bank_path, self.output)
        target.write_bytes(original)
        audit = Path(entry['required_audit'])
        record = json.loads(audit.read_text())
        record['outputs'].pop(str(target))
        audit.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'not bound'):
            resolve(self.mapping_path, self.bank_path, self.output)
        self.assertFalse(self.output.exists())

    def test_reordered_inputs_and_changed_development_binding_are_rejected(self):
        original = copy.deepcopy(self.mapping)
        for change in ('order', 'development_sha', 'producer'):
            altered = copy.deepcopy(original)
            if change == 'order':
                altered['predictions'][0], altered['predictions'][1] = altered['predictions'][1], altered['predictions'][0]
            elif change == 'development_sha':
                altered['predictions'][0]['development_sha256'] = 'wrong'
            else:
                altered['predictions'][0]['required_audit'] = str(self.audits[-1])
            self.mapping_path.write_text(json.dumps(altered))
            with self.subTest(change=change), self.assertRaises(ValueError):
                resolve(self.mapping_path, self.bank_path, self.output)
            self.assertFalse(self.output.exists())

    def test_changed_ready_latent_or_metadata_file_is_rejected(self):
        ready = Path(self.bank['qualifying']['context'][0]['path'])
        ready.write_bytes(b'changed frozen feature')
        with self.assertRaises(ValueError):
            resolve(self.mapping_path, self.bank_path, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
