import importlib.util
from pathlib import Path
import tempfile
import json
import unittest

spec = importlib.util.spec_from_file_location('pause_control', Path(__file__).with_name('control.py'))
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class PauseTests(unittest.TestCase):
    def test_tail_keeps_exact_pending_commands_and_setup(self):
        text = ('#!/bin/bash\nset -eu\nsetup() { true; }\n'
                'shasum first\ntrain0\naudit0\necho COMPLETE_AUDITED_0\n'
                'wait_resources\nshasum second\ntrain1\naudit1\necho COMPLETE_AUDITED_1\nfinished\n')
        self.assertEqual(control.tail_script(text, 0, 2), text)
        tail = control.tail_script(text, 1, 2)
        self.assertEqual(tail, '#!/bin/bash\nset -eu\nsetup() { true; }\nwait_resources\nshasum second\ntrain1\naudit1\necho COMPLETE_AUDITED_1\nfinished\n')
        self.assertIsNone(control.tail_script(text, 2, 2))
        with self.assertRaises(ValueError):
            control.tail_script(text, 1, 3)
        with self.assertRaises(ValueError):
            control.tail_script(text, 3, 2)

    def test_only_terminal_children_can_be_completed(self):
        self.assertTrue(control.terminal(''))
        self.assertTrue(control.terminal('Z'))
        for state in ['R', 'SN', 'T', 'Ss']:
            self.assertFalse(control.terminal(state))

    def test_noncontiguous_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dirs = [Path(temp)/str(i) for i in range(3)]
            for path in dirs:
                path.mkdir()
            plan = dict(stages=[dict(output_dir=str(p)) for p in dirs])
            self.assertEqual(control.receipt_prefix(plan), 0)
            (dirs[0]/'audit.json').write_text('{}')
            self.assertEqual(control.receipt_prefix(plan), 1)
            (dirs[2]/'audit.json').write_text('{}')
            with self.assertRaises(ValueError):
                control.receipt_prefix(plan)

    def test_real_scripts_keep_exact_remaining_suffix(self):
        for phase in ['GPU','RBM','DEPS','JOINT','KNN','AUTOREC','BAYES','KERNEL','USERKNN']:
            text = (control.ROOT/f'experiments/P0-{phase}-20261031-run.sh').read_text()
            plan = json.loads((control.ROOT/f'experiments/P0-{phase}-20261031-plan.json').read_text())
            lines = text.splitlines()
            ends = [i for i, line in enumerate(lines) if line.startswith('echo COMPLETE_AUDITED_')]
            for completed in range(len(plan['stages'])):
                tail = control.tail_script(text, completed, len(plan['stages']))
                if completed:
                    suffix = '\n'.join(lines[ends[completed-1]+1:])+'\n'
                    self.assertTrue(tail.endswith(suffix), (phase, completed))
                    self.assertEqual(sum(x.startswith('echo COMPLETE_AUDITED_') for x in tail.splitlines()), len(plan['stages'])-completed)


if __name__ == '__main__':
    unittest.main()
