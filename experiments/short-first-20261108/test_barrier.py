import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('barrier', Path(__file__).with_name('barrier.py'))
barrier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(barrier)


class BarrierTests(unittest.TestCase):
    def test_all_complete_and_exit_required(self):
        self.assertTrue(barrier.ready([True]*7, [True]*25, True, False, '1'))
        for cpu, arms, summary, live, pressure in [
            ([True]*6, [True]*25, True, False, '1'),
            ([True]*6+[False], [True]*25, True, False, '1'),
            ([True]*7, [True]*24, True, False, '1'),
            ([True]*7, [True]*25, False, False, '1'),
            ([True]*7, [True]*25, True, True, '1'),
            ([True]*7, [True]*25, True, False, '2'),
        ]:
            with self.subTest(cpu=len(cpu), arms=len(arms), summary=summary, live=live, pressure=pressure):
                self.assertFalse(barrier.ready(cpu, arms, summary, live, pressure))

    def test_receipt_requires_matching_plan_budget_stage_and_no_score(self):
        expected = dict(column='MT1', fixed_epochs=10)
        receipt = dict(state='verified_complete', stage=2, plan_sha256='frozen',
                       column='MT1', fixed_epochs=10, qualifying_count=2817131, metric=None)
        self.assertTrue(barrier.compatible_cpu(receipt, expected, 'frozen', 2))
        for key, value in [('state','training'), ('stage',3), ('plan_sha256','changed'),
                           ('column','other'), ('fixed_epochs',9), ('qualifying_count',12), ('metric',0.8)]:
            with self.subTest(field=key):
                self.assertFalse(barrier.compatible_cpu(dict(receipt, **{key:value}), expected, 'frozen', 2))
        del receipt['metric']
        self.assertFalse(barrier.compatible_cpu(receipt, expected, 'frozen', 2))


if __name__ == '__main__':
    unittest.main()
