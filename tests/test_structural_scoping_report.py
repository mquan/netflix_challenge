import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import structural_scoping_report as report
from structural_scoping import arm_indices
from train_holdout import digest


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(columns=[{}]*4, families=dict(a=[0, 1], b=[2, 3]),
            diversity=dict(counts=[1, 4], permutation_seeds=[42, 43, 44]))
        self.arms = arm_indices(self.plan, [.8, .9, .7, .7])
        self.results = {a['name']: dict(comparison=dict(treatment_rmse=.85,
            delta_rmse=.01, fold_deltas=[.01]*5)) for a in self.arms}

    def test_all_permutations_and_shared_reference_are_reported(self):
        for seed, score in zip([42, 43, 44], [.9, 1., 1.1]):
            self.results[f'diversity-s{seed}-n1']['comparison']['treatment_rmse'] = score
        result = report.summarize(self.plan, self.arms, self.results)
        curve = result['diversity']
        self.assertEqual(curve[0]['mean_rmse'], 1.)
        self.assertEqual([v['rmse'] for v in curve[0]['permutations']], [.9, 1., 1.1])
        self.assertFalse(curve[0]['shared_reference'])
        self.assertEqual({v['arm'] for v in curve[1]['permutations']}, {'all79'})
        self.assertTrue(curve[1]['shared_reference'])
        self.assertEqual(curve[1]['permutation_sd'], 0.)

    def test_quality_ratio_keeps_sign_and_zero_denominator_is_undefined(self):
        self.results['quality-a']['comparison']['delta_rmse'] = -.01
        quality = report.summarize(self.plan, self.arms, self.results)['quality_substitutions']
        self.assertAlmostEqual(quality[0]['ratio'], -.1)
        self.assertEqual(quality[0]['fold_deltas'], [.01]*5)
        self.assertIsNone(quality[1]['ratio'])

    def test_missing_or_extra_results_and_duplicate_arms_are_rejected(self):
        for bad in [dict(self.results, extra={}), {k:v for k,v in self.results.items() if k!='quality-a'}]:
            with self.assertRaisesRegex(ValueError, 'every registered arm'):
                report.summarize(self.plan, self.arms, bad)
        with self.assertRaisesRegex(ValueError, 'every registered arm'):
            report.summarize(self.plan, self.arms+[self.arms[0]], self.results)

    def test_partial_experiment_rejected_before_data_load_and_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def save(name, value):
                path = root/name
                path.write_text(json.dumps(value))
                return str(path)
            plan_path = save('protocol.json', self.plan)
            inventory = save('inventory.json', dict(plan_sha256=digest(plan_path),
                columns=[dict(rmse=v) for v in [.8, .9, .7, .7]]))
            arms = [dict(a, stages=[dict(kind=k, config=dict(output=str(root/(a['name']+k))))
                                   for k in ('tree', 'neural')]) for a in self.arms]
            execution = save('execution.json', dict(frozen={}, plan=plan_path,
                plan_sha256=digest(plan_path), standalone_inventory=inventory, arms=arms))
            parity = save('parity.json', dict(state='verified_full_reference_parity'))
            sources = [str(Path(report.__file__).with_name(n)) for n in report.SOURCES]
            binding = save('binding.json', dict(frozen={str(Path(p).resolve()):digest(p)
                for p in [execution, parity, *sources]}))
            output = root/'result.json'
            with patch.object(report, 'load', side_effect=AssertionError('must not load data')):
                with self.assertRaisesRegex(ValueError, 'incomplete fixed experiment'):
                    report.run(execution, parity, binding, output)
            self.assertFalse(output.exists())

    def test_saved_average_fold_coverage_and_bank_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stages = [dict(kind=k, config=dict(output=str(root/k))) for k in ('tree', 'neural')]
            arm = dict(stages=stages)
            tree, neural = np.array([1., 2.]), np.array([3., 4.])
            tree.astype('<f4').tofile(root/'tree.refit.f32')
            neural.astype('<f4').tofile(root/'neural.nn.f32')
            average = (tree+neural)/2
            average.astype('<f4').tofile(root/'neural.avg.f32')
            records = dict(inputs={'column':'hash'}, folds=[dict(fold=f) for f in range(5)])
            for stage in stages:
                Path(stage['config']['output']+'.json').write_text(json.dumps(records))
            with patch.object(report, 'verify_blend', return_value={}):
                np.testing.assert_array_equal(report.audit_arm(arm, {})[0], average)
                (average+.01).astype('<f4').tofile(root/'neural.avg.f32')
                with self.assertRaisesRegex(ValueError, '50/50'):
                    report.audit_arm(arm, {})
                average.astype('<f4').tofile(root/'neural.avg.f32')
                (root/'neural.json').write_text(json.dumps(dict(records, inputs={'other':'hash'})))
                with self.assertRaisesRegex(ValueError, 'input banks differ'):
                    report.audit_arm(arm, {})
                (root/'neural.json').write_text(json.dumps(dict(records, folds=[dict(fold=f) for f in [0,1,2,3,3]])))
                with self.assertRaisesRegex(ValueError, 'five stage folds'):
                    report.audit_arm(arm, {})


if __name__ == '__main__':
    unittest.main()
