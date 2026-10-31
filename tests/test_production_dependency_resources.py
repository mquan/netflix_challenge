from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_dependency_resources import ready


class DependencyResourceTests(unittest.TestCase):
    def test_waits_for_all_s0_parents_and_children_and_normal_memory(self):
        patterns = ['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                    'scripts/round4_boost.py', 'scripts/round8_nn_blend.py']
        plan = dict(resource_gate=dict(blocked_process_patterns=patterns))
        for pattern in patterns:
            self.assertFalse(ready(plan, pattern, '1'))
        for pressure in ('2', '4', '', 'unknown'):
            self.assertFalse(ready(plan, '', pressure))
        # GPU + the previously guarded eight-thread RBM phase + this one thread = 10.
        self.assertTrue(ready(plan, 'production_neural.py --column A2h\nP0-RBM-20261031-run.sh', '1'))
        self.assertTrue(ready(plan, 'P0-CPU-neural-20261031-run.sh', '1'))


if __name__ == '__main__':
    unittest.main()
