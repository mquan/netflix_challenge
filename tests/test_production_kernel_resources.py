import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_kernel_resources import ready


class KernelResourceTests(unittest.TestCase):
    def test_both_queues_and_normal_memory_required(self):
        required = [dict(stage=i, plan_sha256='bayes', final_sweep=160, kept=144) for i in range(4)]
        required += [dict(stage=i, plan_sha256='knn', qualifying_count=2817131) for i in range(28)]
        patterns = ['P0-BAYES-run.sh', 'P0-KNN-run.sh', '/production/bayes/', '/production/knn/',
                    'S0-lane0-run.sh', 'P0-RBM-run.sh', 'P0-CPU-neural-run.sh']
        plan = dict(resource_gate=dict(required_audits=required, blocked_process_patterns=patterns))
        receipts = [dict(row, state='verified_complete', metric=None) for row in required]
        self.assertTrue(ready(plan, 'primary-GPU', '1', receipts))
        self.assertTrue(ready(plan, 'AutoRec-GPU', '1', receipts))
        self.assertFalse(ready(plan, '', '1', receipts[4:]))
        self.assertFalse(ready(plan, '', '1', receipts[:4]))
        for pattern in patterns:
            self.assertFalse(ready(plan, pattern, '1', receipts))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, receipts))
        for index, key, value in [(0, 'kept', 0), (0, 'final_sweep', 32), (4, 'qualifying_count', 1),
                                  (4, 'plan_sha256', 'other'), (4, 'stage', 99),
                                  (0, 'metric', 0), (4, 'state', 'partial')]:
            changed = copy.deepcopy(receipts)
            changed[index][key] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
