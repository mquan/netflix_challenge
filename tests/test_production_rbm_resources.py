import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_rbm_resources import ready


class RbmResourcesTests(unittest.TestCase):
    def test_requires_all_cpu_audits_and_no_conflicting_jobs(self):
        required = [dict(stage=i, column=f'column-{i}', plan_sha256='cpu-plan') for i in range(7)]
        plan = dict(resource_gate=dict(blocked_process_patterns=['S0-lane0-run.sh',
                         'S0-lane1-run.sh', 'CPU-neural-run.sh', 'round4_boost.py', 'round8_nn_blend.py'],
                         required_cpu_neural_audits=required))
        receipts = [dict(item, state='verified_complete', metric=None) for item in required]
        self.assertTrue(ready(plan, 'production_neural.py --column A2h', '1', receipts))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, receipts))
        for pattern in plan['resource_gate']['blocked_process_patterns']:
            self.assertFalse(ready(plan, pattern, '1', receipts))
        self.assertFalse(ready(plan, '', '1', receipts[-1:]))
        for field, value in [('state', 'partial'), ('metric', 0), ('stage', 999),
                             ('column', 'other'), ('plan_sha256', 'other')]:
            changed = copy.deepcopy(receipts)
            changed[0][field] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
