import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_knn_resources import ready


class KnnResourceTests(unittest.TestCase):
    def test_every_parent_required_and_conflicting_jobs_block(self):
        required = [dict(stage=i, name=f'joint-{i}', final_epoch=2, plan_sha256='joint') for i in range(12)]
        required += [dict(stage=i, kind=f'dependency-{i}', plan_sha256='dependencies') for i in range(4)]
        patterns = ['S0-lane0-run.sh', 'S0-lane1-run.sh', 'P0-JOINT-run.sh', 'P0-DEPS-run.sh']
        plan = dict(resource_gate=dict(required_audits=required, blocked_process_patterns=patterns))
        receipts = [dict(r, state='verified_complete', metric=None) for r in required]
        self.assertTrue(ready(plan, 'GPU\nRBM', '1', receipts))
        self.assertTrue(ready(plan, 'GPU\nCPU-neural', '1', receipts))
        self.assertFalse(ready(plan, '', '1', receipts[-1:]))
        for pattern in patterns:
            self.assertFalse(ready(plan, pattern, '1', receipts))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, receipts))
        for key, value in [('state', 'partial'), ('metric', 1), ('stage', 99), ('name', 'other'), ('plan_sha256', 'other'), ('final_epoch', 1)]:
            changed = copy.deepcopy(receipts)
            changed[0][key] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
