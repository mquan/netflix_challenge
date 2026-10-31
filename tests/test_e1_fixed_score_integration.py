"""Real fixture features, P2 ridge/LightGBM fit, four-arm scoring and receipts."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_fixed_score as scoring
from test_joint import dataset


class ScoringIntegrationTests(unittest.TestCase):
    def test_query_labels_change_scores_but_never_fit_or_prediction_bytes(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            try:
                os.chdir(root)
                (root/'experiments').mkdir()
                (root/'build').mkdir()
                binary = root/'build/netflix-rowweights'
                binary.write_bytes(b'fixture producer receipt only; never executed')
                dataset(root/'fit.bin')
                dataset(root/'eval.bin')
                names = ['MH3', 'MT0', 'MT1', 'MT2', 'MT3', 'K1', 'knn256td', 'knn64', 'rbm']
                original = dict(columns=[f'artifacts/rehearsal/B-{name}.f32' for name in names],
                    data='eval.bin', support=['artifacts/rehearsal/B-knn256td.netflix.support.f32'])
                original['columns'][6] = 'artifacts/rehearsal/B-knn256td.netflix.f32'
                original['columns'][7] = 'artifacts/rehearsal/B-knn64.netflix.f32'
                fit = dict(columns=[f'fit-{n}.f32' for n in range(9)], data='fit.bin', support=['fit-support.f32'])
                spec = dict(fit=fit, a=fit, b=original)
                arms = {arm: scoring.arm_spec(spec, arm) for arm in scoring.ARMS}
                rng = np.random.default_rng(31)
                for arm in [spec, *arms.values()]:
                    for bank in ('fit', 'b'):
                        for name in arm[bank]['columns']+arm[bank]['support']:
                            path = root/name
                            if not path.exists():
                                path.parent.mkdir(parents=True, exist_ok=True)
                                rng.uniform(2, 4, 200).astype('<f4').tofile(path)
                command_plan = {arm: scoring.commands(arm) for arm in scoring.ARMS}
                for commands in command_plan.values():
                    for name, command in commands.items():
                        (root/'experiments'/f'{name}.json').write_text(json.dumps(dict(
                            returncode=0, command=' '.join(command), binary_sha256=scoring.digest(binary))))
                settings = dict(rounds=3, learning_rate=.02, leaves=3, min_leaf=5,
                                threads=1, processes=1, seeds=[2026], skip_cv=True)
                fit_features = scoring.rehearsal.features(fit)
                predictor, _ = scoring.rehearsal.train(fit_features['x'], fit_features['rx'],
                    fit_features['y'], fit_features['users'], argparse.Namespace(**settings), 2026)
                actual_train = scoring.rehearsal.train
                baseline = root/'baseline.json'
                plan_path = root/'plan.json'
                plan = dict(spec=spec, arm_specs=arms, commands=command_plan,
                    frozen={str(binary): scoring.digest(binary)}, settings=settings,
                    baseline_report=str(baseline), baseline_tolerance=1e-6)
                saved = {}
                scores = []
                with patch.object(scoring, 'ROOT', root), \
                        patch.object(scoring.rehearsal, 'train', wraps=actual_train) as train:
                    for revision in (0, 1):
                        if revision:
                            dataset(root/'eval.bin', change_fold=0)
                        features = scoring.rehearsal.features(original)
                        expected = predictor(features['x'], features['rx'])
                        baseline.write_text(json.dumps({'b_rmse': scoring.rehearsal.rmse(expected, features['y'])}))
                        plan_path.write_text(json.dumps(plan))
                        output = root/f'output-{revision}'
                        with contextlib.redirect_stdout(io.StringIO()):
                            scoring.score(plan_path, output)
                        saved[revision] = {arm: (output/f'{arm}.b.f32').read_bytes()
                                           for arm in ('original', *scoring.ARMS)}
                        report = json.loads((output/'comparison.json').read_text())
                        scores.append(report['baseline_rmse'])
                        self.assertEqual(len(report['generated']), 27)  # nine receipts and 18 distinct inputs
                        self.assertTrue(all(len(values) == 800 for values in saved[revision].values()))
                    receipt = root/'experiments/E1-B-K1-w1.json'
                    failed = json.loads(receipt.read_text())
                    failed['returncode'] = 1
                    receipt.write_text(json.dumps(failed))
                    with self.assertRaises(ValueError):
                        scoring.score(plan_path, root/'rejected')
                    self.assertFalse((root/'rejected').exists())
                self.assertEqual(train.call_count, 2)  # one P2 fit per complete four-arm evaluation
                for call in train.call_args_list:
                    np.testing.assert_array_equal(call.args[2], fit_features['y'])
                self.assertEqual(saved[0], saved[1])
                self.assertNotEqual(scores[0], scores[1])
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
