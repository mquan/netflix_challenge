import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_final_resources import ready


class FinalResourceTests(unittest.TestCase):
    def test_every_stage_queue_exit_and_normal_memory_required(self):
        required = [dict(stage=i, plan_sha256='phase', state='verified_complete', metric=None) for i in range(82)]
        patterns = ['P0-GPU-run.sh', 'P0-AUTOREC-run.sh', 'P0-USERKNN-run.sh',
                    'S0-lane0-run.sh', '/production/neural/', '/production/user-knn/']
        plan = dict(resource_gate=dict(required_audits=required, blocked_process_patterns=patterns))
        self.assertTrue(ready(plan, '', '1', required))
        self.assertFalse(ready(plan, '', '1', required[:-1]))
        for pattern in patterns:
            self.assertFalse(ready(plan, pattern, '1', required))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, required))
        for key, value in [('stage', 99), ('plan_sha256', 'other'), ('state', 'partial'), ('metric', 0)]:
            changed = copy.deepcopy(required)
            changed[-1][key] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
