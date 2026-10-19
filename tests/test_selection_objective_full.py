"""Conditional full-data execution and matched experiment decision contracts."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import selection_objective_full as full
from gbdt_blend import blend_fold


class FullComparisonTests(unittest.TestCase):
    def test_gate_requires_real_finite_pass_bound_to_plan(self):
        gate = dict(primary={'delta_rmse': -.0006}, plan_sha256='expected', passes_full_refit_gate=True)
        full.require_gate(gate, 'expected', .0005)
        for changed in (dict(plan_sha256='other'), dict(passes_full_refit_gate=False),
                        dict(passes_full_refit_gate=1), dict(primary={'delta_rmse': -.00049}),
                        dict(primary={'delta_rmse': float('nan')}), dict(primary={'delta_rmse': -float('inf')})):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                full.require_gate(dict(gate, **changed), 'expected', .0005)

    def test_missing_or_failed_gate_launches_nothing(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(full.subprocess, 'run') as launch:
            path = Path(temp) / 'gate.json'
            plan = dict(p2_gate=str(path), p2_plan_sha256='expected', p2_minimum_gain=.0005)
            with self.assertRaises(FileNotFoundError):
                full.verify_gate(plan)
            path.write_text(json.dumps(dict(primary={'delta_rmse': -.0001},
                plan_sha256='expected', passes_full_refit_gate=False)))
            with self.assertRaises(ValueError):
                full.verify_gate(plan)
            plan.update(run_id='fixture', output=str(Path(temp)/'output'), frozen={})
            plan_path = Path(temp)/'full-plan.json'
            plan_path.write_text(json.dumps(plan))
            (Path(temp)/'experiments').mkdir()
            previous = Path.cwd()
            try:
                with patch.object(full, 'ROOT', Path(temp)), self.assertRaises(ValueError):
                    full.run(plan_path)
            finally:
                os.chdir(previous)
            self.assertFalse(Path(plan['output']).exists())
            self.assertFalse((Path(temp)/'experiments/fixture-chain.status.json').exists())
            launch.assert_not_called()

    def test_gate_recomputes_and_binds_both_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gate = dict(primary={'delta_rmse': -.0006}, plan_sha256='expected', passes_full_refit_gate=True,
                arms={'control': {'report_sha256': 'a'}, 'treatment': {'report_sha256': 'b'}},
                selected_penalties={'control': 10, 'treatment': 100})
            path = root / 'gate.json'
            path.write_text(json.dumps(gate))
            arms = {arm: {'output': str(root / arm)} for arm in ('control', 'treatment')}
            for arm in arms.values():
                for suffix in ('.json', '.e2.f32'):
                    Path(arm['output']+suffix).write_bytes(b'evidence')
            p2 = root / 'p2.json'
            p2.write_text(json.dumps(dict(gate_command=['python', 'gate.py', '--output', str(path)], arms=arms, score_epoch=2)))
            plan = dict(p2_gate=str(path), p2_plan=str(p2), p2_plan_sha256='expected', p2_minimum_gain=.0005)
            def recompute(command, **kwargs):
                Path(command[-1]).write_text(json.dumps(gate))
            with patch.object(full.subprocess, 'run', side_effect=recompute):
                self.assertEqual(len(full.verify_gate(plan)), 5)
            gate['selected_penalties']['control'] = 1000
            with patch.object(full.subprocess, 'run', side_effect=recompute), self.assertRaisesRegex(ValueError, 'evidence differs'):
                full.verify_gate(plan)

    def test_six_stages_preserve_duplicate_column_and_fixed_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('old.f32', 'support.f32'):
                (root / name).touch()
            p2 = dict(settings=dict(epochs=2, lr=.0003, seed=19, warm_stage='selection'),
                arms={'control': {'weight': 0.0}, 'treatment': {'weight': .1}}, score_epoch=2)
            base = dict(config=dict(bank_root='bank', extra=['old.f32'], extra_support=['support.f32'],
                rounds=8000, seeds=[2026, 7, 13], processes=1, no_union=False))
            neural = dict(config=dict(hidden=128, seeds=[2026, 7, 13]))
            original = copy.deepcopy((p2, base, neural))
            stages = full.make_stages(p2, base, neural, root, root/'data', root/'warm.json',
                root/'warm.pt', root/'out', 'SO1-full', 3)
            self.assertEqual((p2, base, neural), original)
            self.assertEqual([s['kind'] for s in stages], ['training', 'training', 'tree', 'neural', 'tree', 'neural'])
            for s in stages[:2]:
                self.assertEqual(s['config']['epochs'], 2)
                self.assertEqual(s['config']['warm_stage'], 'refit')
                self.assertEqual(s['config']['warm_checkpoint'], str(root/'warm.pt'))
            control, treatment = stages[2]['config'], stages[4]['config']
            self.assertEqual(control['extra'][-2:], [str(root/'out/control.e2.f32')]*2)
            self.assertEqual(treatment['extra'][-2:], [str(root/'out/control.e2.f32'), str(root/'out/treatment.e2.f32')])
            self.assertEqual(len(control['extra']), len(treatment['extra']))
            self.assertEqual(stages[2]['command'].count(str(root/'out/control.e2.f32')), 2)
            self.assertEqual(control['seeds'], base['config']['seeds'])
            self.assertEqual(control['rounds'], 8000)
            self.assertEqual(stages[3]['config']['like'], control['output']+'.json')
            self.assertEqual(stages[5]['config']['like'], treatment['output']+'.json')

    def test_execution_rechecks_and_stops_at_failed_stage(self):
        events = []
        def check(): events.append('check')
        def wait(s): events.append('wait'+s)
        def run(s): events.append('run'+s)
        def validate(s):
            events.append('validate'+s)
            if s == 'b': raise ValueError('bad output')
        with self.assertRaisesRegex(ValueError, 'bad output'):
            full.execute(['a', 'b', 'c'], check, wait, run, validate)
        self.assertEqual(events, ['check', 'waita', 'check', 'waita', 'runa', 'validatea',
            'check', 'waitb', 'check', 'waitb', 'runb', 'validateb'])

    def test_investment_requires_both_comparisons_and_every_fold(self):
        users = np.arange(100, dtype=np.uint32)
        y = np.full(100, 3.0)
        reference, control, treatment = y+.9, y+.89, y+.88
        result = full.compare_full(y, users, reference, control, treatment, .0003)
        self.assertTrue(result['substantial_auxiliary_investment'])
        self.assertFalse(result['target_reached_on_development'])
        # Beating a weak control is insufficient if the retained ensemble wins.
        result = full.compare_full(y, users, y+.87, control, treatment, .0003)
        self.assertFalse(result['substantial_auxiliary_investment'])
        # An aggregate win cannot hide a regression on one user fold.
        treatment[blend_fold(users) == 0] = 3.895
        result = full.compare_full(y, users, reference, control, treatment, .0003)
        self.assertLess(result['primary']['delta_rmse'], -.0003)
        self.assertFalse(result['substantial_auxiliary_investment'])
        result = full.compare_full(y, users, reference, control, y+.8899, .0003)
        self.assertFalse(result['substantial_auxiliary_investment'])

    def test_blend_verification_checks_bank_library_and_saved_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'experiments').mkdir()
            (root/'experiments/UNION-round-manifest.json').write_text(json.dumps({'reports': {'union': {'inputs': {'old': 'hash'}}}}))
            (root/'experiments/BOOST-support-inputs.json').write_text('{}')
            reference = root/'reference.json'
            reference.write_text(json.dumps({'lightgbm_version': 'fixed'}))
            cfg = dict(bank_root=str(root), output=str(root/'tree'), extra=[], extra_support=[])
            report = dict(config=cfg, inputs={'old': 'hash'}, lightgbm_version='fixed', refit_rmse=.5, ridge_rmse=.5)
            report_path = root/'tree.json'
            report_path.write_text(json.dumps(report))
            for name in ('refit', 'ridge'):
                np.full(10, 3.5, dtype='<f4').tofile(root/f'tree.{name}.f32')
            plan = dict(data='fixture', rows=10, base_report=str(reference))
            stage = dict(config=cfg, kind='tree')
            with patch.object(full, 'load', return_value={'probe': {'rating': np.full(10, 3.0)}}), \
                 patch.object(full.importlib.metadata, 'version', return_value='fixed'):
                self.assertEqual(len(full.verify_blend(stage, plan)), 3)
                for changes, message in (({'refit_rmse': .1}, 'score does not reproduce'),
                                         ({'inputs': {'wrong': 'hash'}}, 'input bank differs'),
                                         ({'lightgbm_version': 'changed'}, 'library differs')):
                    report_path.write_text(json.dumps(dict(report, **changes)))
                    with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                        full.verify_blend(stage, plan)


if __name__ == '__main__':
    unittest.main()
