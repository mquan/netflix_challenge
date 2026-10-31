import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_autorec_resources import ready


class AutoRecResourceTests(unittest.TestCase):
    def test_every_primary_gpu_stage_required_and_no_gpu_overlap(self):
        expected = [dict(stage=i, column=f'column-{i}', fixed_epochs=i + 1, plan_sha256='primary', qualifying_count=100)
                    for i in range(7)]
        patterns = ['P0-GPU-run.sh', '--column primary', 'moonshot_sequence.py']
        plan = dict(resource_gate=dict(required_primary_gpu_audits=expected, blocked_process_patterns=patterns))
        receipts = [dict(e, state='verified_complete', metric=None) for e in expected]
        self.assertTrue(ready(plan, 'CPU neural\nRBM', '1', receipts))
        self.assertFalse(ready(plan, '', '1', receipts[-1:]))
        for p in patterns:
            self.assertFalse(ready(plan, p, '1', receipts))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure, receipts))
        for key, value in [('state', 'partial'), ('metric', 1), ('stage', 99), ('column', 'other'),
                           ('fixed_epochs', 99), ('plan_sha256', 'other'), ('qualifying_count', 99)]:
            changed = copy.deepcopy(receipts)
            changed[0][key] = value
            self.assertFalse(ready(plan, '', '1', changed))


if __name__ == '__main__':
    unittest.main()
