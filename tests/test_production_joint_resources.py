import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_joint_resources import ready


class JointResourceTests(unittest.TestCase):
    def test_requires_every_dependency_audit_and_free_slot(self):
        patterns = ['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                    'P0-DEPS-20261031-run.sh', 'scripts/round4_boost.py', 'scripts/round8_nn_blend.py']
        required = [dict(stage=i, kind=k, plan_sha256='dependency-plan') for i, k in
                    enumerate(['baseline', 'graph256', 'graph64', 'neighborhood'])]
        plan = dict(resource_gate=dict(blocked_process_patterns=patterns, required_dependency_audits=required))
        receipts = [dict(r, state='verified_complete', metric=None) for r in required]
        self.assertTrue(ready(plan, 'GPU\nP0-RBM-20261031-run.sh', '1', receipts))
        self.assertTrue(ready(plan, 'GPU\nP0-CPU-neural-20261031-run.sh', '1', receipts))
        for pattern in patterns:
            self.assertFalse(ready(plan, pattern, '1', receipts))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, receipts))
        self.assertFalse(ready(plan, '', '1', receipts[2:3]))
        for key, value in [('state', 'partial'), ('metric', 1), ('stage', 99), ('kind', 'other'), ('plan_sha256', 'other')]:
            changed = copy.deepcopy(receipts)
            changed[0][key] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
