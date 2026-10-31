import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from e3_candidate_gate import SCRIPTS, audit_report, run
from train_holdout import digest
import test_p2_candidate_gate as fixtures


class RecipeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        frozen = {}
        self.expected = dict(output=str(root/'model'), epochs=2, max_batches=0)
        for key in ('data', 'warm_report', 'warm_checkpoint'):
            path = root/key
            path.write_bytes(key.encode())
            self.expected[key] = str(path)
            frozen[str(path)] = digest(path)
        self.sources = {'selection_objective.py': digest(SCRIPTS/'selection_objective.py')}
        frozen.update({str(SCRIPTS/k): v for k, v in self.sources.items()})
        self.plan = dict(frozen=frozen, libraries=dict(torch='fixture', numpy='fixture'),
                         score_epoch=2, data=self.expected['data'])
        self.config = dict(head='distribution', loss='mse')
        self.report = dict(arguments=self.expected, config=self.config, sources=self.sources,
                           torch_version='fixture', numpy_version='fixture', records=[])
        for key in ('data', 'warm_report', 'warm_checkpoint'):
            self.report[key+'_sha256'] = frozen[self.expected[key]]
        for epoch in (1, 2):
            pred = Path(self.expected['output']+f'.e{epoch}.f32')
            np.array([2.5, 3.5], '<f4').tofile(pred)
            ckpt = pred.with_suffix('.pt')
            ckpt.write_bytes(b'opaque fixture checkpoint')
            self.report['records'].append(dict(epoch=epoch, prediction_sha256=digest(pred),
                                              checkpoint_sha256=digest(ckpt)))
        self.path = root/'report.json'

    def audit(self, report=None):
        self.path.write_text(json.dumps(self.report if report is None else report))
        return audit_report(self.plan, self.path, self.expected, self.config, self.sources)

    def test_complete_recipe_binds_final_prediction_and_all_saved_epochs(self):
        receipt, bound = self.audit()
        self.assertEqual(receipt['state'], 'complete')
        self.assertEqual(receipt['prediction'], self.expected['output']+'.e2.f32')
        self.assertEqual(len(bound), 5)

    def test_partial_budget_changed_architecture_or_missing_sources_are_rejected(self):
        for key, value in [('records', self.report['records'][:1]),
                           ('config', dict(head='scalar')), ('sources', {})]:
            report = copy.deepcopy(self.report)
            report[key] = value
            with self.assertRaises(ValueError):
                self.audit(report)
        report = copy.deepcopy(self.report)
        report['arguments']['max_batches'] = 1
        with self.assertRaisesRegex(ValueError, 'arguments'):
            self.audit(report)

    def test_changed_payload_or_warm_provenance_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['warm_checkpoint_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'warm'):
            self.audit(report)
        Path(self.expected['output']+'.e2.f32').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'saved training output'):
            self.audit()

    def test_complete_fixture_runs_both_gates_and_incomplete_run_never_scores(self):
        fixture = fixtures.CandidateGateTests('test_matches_existing_comparator_and_rejects_overwrite')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = fixture.root
        frozen = dict(fixture.plan['frozen'])
        warm = {}
        for key in ('warm_report', 'warm_checkpoint'):
            path = root/key
            path.write_bytes(b'fixture warm input')
            warm[key] = str(path)
            frozen[str(path)] = digest(path)
        settings = dict(epochs=2, max_batches=0, seed=20261019)
        parent_path = root/'parent.json'
        parent = dict(settings=settings, data=str(fixture.data), **warm,
                      arms=dict(control=dict(output=str(root/'control'))))
        parent_path.write_text(json.dumps(parent))
        frozen[str(parent_path)] = digest(parent_path)
        sources = {'selection_objective.py': digest(SCRIPTS/'selection_objective.py')}
        frozen[str(SCRIPTS/'selection_objective.py')] = sources['selection_objective.py']
        libraries = dict(torch='fixture', numpy='fixture')
        expected_control = dict(settings, data=str(fixture.data), **warm,
                                output=str(root/'control'), weight=0.0)
        trainer = SCRIPTS/'e3_distribution_loss.py'
        expected_treatment = dict(expected_control, output=str(root/'treatment'),
                                  ce_weight=.1, e3_script_sha256=digest(trainer))
        reports = {}
        for arm, expected, values in [('control', expected_control, fixture.control),
                                      ('treatment', expected_treatment, fixture.treatment)]:
            report = dict(arguments=expected, config=self.config, sources=sources,
                          torch_version='fixture', numpy_version='fixture', records=[])
            for key in ('data', 'warm_report', 'warm_checkpoint'):
                report[key+'_sha256'] = frozen[expected[key]]
            for epoch in (1, 2):
                pred = Path(expected['output']+f'.e{epoch}.f32')
                values.astype('<f4').tofile(pred)
                ckpt = pred.with_suffix('.pt')
                ckpt.write_bytes(b'fixture checkpoint')
                report['records'].append(dict(epoch=epoch, prediction_sha256=digest(pred),
                                              checkpoint_sha256=digest(ckpt)))
            path = root/(arm+'.json')
            path.write_text(json.dumps(report))
            reports[arm] = report
            if arm == 'control':
                frozen[str(path)] = digest(path)
        plan = dict(parent_SO1_plan=str(parent_path), frozen=frozen, libraries=libraries,
            score_epoch=2, data=str(fixture.data), bank=fixture.paths,
            control=dict(report=str(root/'control.json')),
            treatment=dict(output=str(root/'treatment'), expected_arguments=expected_treatment,
                           command=['python', str(trainer)]),
            primary_gate=dict(minimum_gain=.0003), additional_gate=dict(minimum_gain=.0003))
        path = root/'E3-plan.json'
        path.write_text(json.dumps(plan))
        result = run(path, root/'complete-gate')
        self.assertTrue(result['passes_both_P2_gates'])
        self.assertTrue(result['all_folds']['all_folds_improve'])
        for saved in result['saved_all_fold_predictions']:
            self.assertEqual(np.load(saved).shape, fixture.y.shape)
        incomplete = copy.deepcopy(reports['treatment'])
        incomplete['records'] = incomplete['records'][:1]
        (root/'treatment.json').write_text(json.dumps(incomplete))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            run(path, root/'incomplete-gate')
        self.assertFalse((root/'incomplete-gate').exists())


if __name__ == '__main__':
    unittest.main()
