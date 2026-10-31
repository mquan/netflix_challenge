import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_bayes_resources import ready


class BayesResourceTests(unittest.TestCase):
    def test_all_rbm_stages_and_conflicting_processes(self):
        required = [dict(stage=i, name=f'rbm-{i}', final_epoch=14, plan_sha256='rbm',
                         qualifying_count=2817131) for i in range(12)]
        patterns = ['S0-lane0-run.sh', 'S0-lane1-run.sh', 'P0-RBM-run.sh',
                    'P0-CPU-neural-run.sh', '/production/rbm/', '--column MH3',
                    'scripts/round4_boost.py', 'scripts/round8_nn_blend.py']
        plan = dict(resource_gate=dict(required_audits=required, blocked_process_patterns=patterns))
        receipts = [dict(row, state='verified_complete', metric=None) for row in required]
        self.assertTrue(ready(plan, 'primary-GPU\nP0-KNN', '1', receipts))
        self.assertTrue(ready(plan, 'AutoRec-GPU\nP0-JOINT', '1', receipts))
        self.assertFalse(ready(plan, '', '1', receipts[:-1]))
        for pattern in patterns:
            self.assertFalse(ready(plan, pattern, '1', receipts))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, receipts))
        for key, value in [('stage', 99), ('state', 'partial'), ('metric', 0), ('name', 'wrong'),
                           ('final_epoch', 13), ('plan_sha256', 'wrong'), ('qualifying_count', 1)]:
            changed = copy.deepcopy(receipts)
            changed[0][key] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
