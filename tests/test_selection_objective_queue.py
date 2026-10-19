import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from selection_objective_queue import inspect_idle, python_jobs, run_pair


class SelectionQueueTests(unittest.TestCase):
    def test_process_snapshot_uses_executable_and_handles_paths_with_spaces(self):
        rows = '''1 /bin/zsh
2 /opt/python/bin/Python
3 /venv with spaces/bin/python3
4 /venv/bin/python
5 /bin/caffeinate'''
        self.assertEqual([x['pid'] for x in python_jobs(rows, 4)], [2, 3])

    def test_stale_terminal_status_does_not_override_live_process(self):
        status = mock.Mock()
        status.read_text.return_value = '{"state":"complete"}'
        with mock.patch('selection_objective_queue.subprocess.check_output', return_value='2 /bin/python'):
            self.assertFalse(inspect_idle(status, 4)[0])
        with mock.patch('selection_objective_queue.subprocess.check_output', return_value='4 /bin/python'):
            self.assertTrue(inspect_idle(status, 4)[0])
            status.read_text.return_value = '{"state":"waiting"}'
            self.assertFalse(inspect_idle(status, 4)[0])
            status.read_text.side_effect = OSError('missing')
            self.assertFalse(inspect_idle(status, 4)[0])

    def test_runs_exact_commands_in_order_and_never_continues_after_failure(self):
        plan = {'arms': {'control': {'command': ['control', '--weight', '0']},
                         'treatment': {'command': ['treatment', '--weight', '.1']}},
                'gate_command': ['gate']}
        events = []
        run = lambda name, cmd: events.append((name, cmd))
        wait = lambda name: events.append(('wait', name))
        verify = lambda: events.append(('verify',))
        run_pair(plan, run, wait, verify)
        self.assertEqual([e for e in events if e[0] in ('control', 'treatment', 'gate')],
                         [('control', plan['arms']['control']['command']),
                          ('treatment', plan['arms']['treatment']['command']), ('gate', ['gate'])])
        self.assertEqual(events[:4], [('wait', 'control'), ('verify',), ('wait', 'control'),
                                     ('control', plan['arms']['control']['command'])])
        failing = mock.Mock(side_effect=RuntimeError('failed'))
        with self.assertRaises(RuntimeError):
            run_pair(plan, failing, wait, verify)
        self.assertEqual(failing.call_count, 1)


if __name__ == '__main__':
    unittest.main()
