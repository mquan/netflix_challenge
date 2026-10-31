import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import selection_objective_n1_followup as n1


class FollowupTests(unittest.TestCase):
    def fixture(self, root, candidate):
        (root/'experiments').mkdir()
        gate = root/'gate.json'
        gate.write_text(json.dumps({'candidate_beyond_A2h': {'delta_rmse': -0.0012}}))
        ref = root/'reference.f32'
        np.full(100, 3.5, '<f4').tofile(ref)
        for suffix in ('avg.f32', 'nn.f32'):
            np.full(100, candidate, '<f4').tofile(root/('neural.'+suffix))
        np.full(100, candidate, '<f4').tofile(root/'tree.refit.f32')
        (root/'tree.json').write_text(json.dumps({'inputs': {}}))
        (root/'neural.json').write_text(json.dumps({'inputs': {}, 'avg_rmse': abs(float(np.float32(candidate))-3)}))
        (root/'experiments/SO1-full-plan.json').write_text(json.dumps(dict(
            reference_prediction=str(ref), p2_plan='p2', p2_gate='gate',
            stages=[dict(config=dict(warm_report='warm', warm_checkpoint='checkpoint'))])))
        (root/'experiments/LC6-plan.json').write_text(json.dumps({'frozen': {}}))
        np.full(100, candidate, '<f4').tofile(root/'refit.f32')
        data = {'probe': np.rec.fromarrays([np.full(100, 3.), np.arange(100)],
                                          names=['rating', 'user'])}
        plan = dict(frozen={}, gate=str(gate), stages=[dict(name='blend', kind='blend', command=['fixed'])],
                    tree_report=str(root/'tree.json'), neural_report=str(root/'neural.json'),
                    refit=str(root/'refit'), data='data')
        return plan, data

    def test_failed_gate_launches_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _ = self.fixture(Path(temp), 3.4)
            Path(plan['gate']).write_text(json.dumps({'candidate_beyond_A2h': {'delta_rmse': -0.0009}}))
            launch = Mock()
            with self.assertRaises(ValueError):
                n1.execute(plan, launch, Mock(), Mock())
            launch.assert_not_called()

    def test_success_refreshes_reference_without_overwriting_old_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, data = self.fixture(root, 3.4)
            original = (root/'experiments/SO1-full-plan.json').read_bytes()
            def prepare(args):
                Path(args.plan).write_text('{}')
            with patch.object(n1, 'ROOT', root), patch.object(n1.full, 'load', return_value=data), \
                    patch.object(n1.full, 'prepare', side_effect=prepare) as make:
                result = n1.execute(plan, Mock(), Mock(), Mock())
            self.assertTrue(result.exists())
            make.assert_called_once()
            self.assertEqual((root/'experiments/SO1-full-plan.json').read_bytes(), original)

    def test_regression_keeps_original_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, data = self.fixture(root, 3.6)
            with patch.object(n1, 'ROOT', root), patch.object(n1.full, 'load', return_value=data), \
                    patch.object(n1.full, 'prepare') as make:
                result = n1.execute(plan, Mock(), Mock(), Mock())
            self.assertIsNone(result)
            make.assert_not_called()


if __name__ == '__main__':
    unittest.main()
