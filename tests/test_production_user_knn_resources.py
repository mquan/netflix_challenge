import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_user_knn_resources import ready


class UserKnnResourceTests(unittest.TestCase):
    def test_all_kernel_stages_process_exit_and_normal_memory_required(self):
        required = [dict(stage=i, plan_sha256='kernel', name=f'kernel{i}',
                         family='gp' if i < 2 else 'local_residual', fixed_epochs=0 if i < 2 else 6,
                         qualifying_count=2817131) for i in range(4)]
        patterns = ['P0-KERNEL-run.sh', '/production/kernel/', 'P0-BAYES-run.sh',
                    'P0-KNN-run.sh', 'S0-lane0-run.sh', 'P0-RBM-run.sh', 'P0-CPU-neural-run.sh']
        plan = dict(resource_gate=dict(required_audits=required, blocked_process_patterns=patterns))
        receipts = [dict(row, state='verified_complete', metric=None) for row in required]
        self.assertTrue(ready(plan, 'primary-GPU', '1', receipts))
        self.assertTrue(ready(plan, 'AutoRec-GPU', '1', receipts))
        self.assertFalse(ready(plan, '', '1', receipts[:-1]))
        for pattern in patterns:
            self.assertFalse(ready(plan, pattern, '1', receipts))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, receipts))
        for key, value in [('stage', 99), ('plan_sha256', 'other'), ('name', 'other'), ('fixed_epochs', 3),
                           ('qualifying_count', 1), ('family', 'other'), ('state', 'partial'), ('metric', 0)]:
            changed = copy.deepcopy(receipts)
            changed[0][key] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
