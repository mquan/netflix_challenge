import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from selection_objective_resume import WORKER, classify, process_snapshot


class ResumeProcessTests(unittest.TestCase):
    def test_only_exact_worker_is_excluded(self):
        snapshot = '\n'.join([
            f'10 {WORKER} {WORKER} -m runlayer_cli.hook._transcript_stream_worker',
            f'11 {WORKER} {WORKER} -m another_module',
            f'12 /tmp/python /tmp/python -m runlayer_cli.hook._transcript_stream_worker',
            f'13 {WORKER} {WORKER} -m runlayer_cli.hook._transcript_stream_worker --extra',
            '14 /tmp/python /tmp/python scripts/selection_objective_resume.py',
            '15 /bin/zsh zsh -c python',
        ])
        blocked, ignored = classify(snapshot, 14)
        self.assertEqual([p['pid'] for p in blocked], [11, 12, 13])
        self.assertEqual([p['pid'] for p in ignored], [10])

    def test_malformed_snapshot_fails_closed(self):
        with self.assertRaises(ValueError):
            classify('10 /tmp/python', 14)

    def test_separate_ps_columns_and_disappearing_process(self):
        with patch('selection_objective_resume.subprocess.check_output', side_effect=[
                f'10 {WORKER}\n11 /tmp/python\n',
                f'10 {WORKER} -m runlayer_cli.hook._transcript_stream_worker\n']) as ps:
            blocked, ignored = classify(process_snapshot(), 14)
        self.assertEqual([p['pid'] for p in blocked], [11])
        self.assertEqual([p['pid'] for p in ignored], [10])
        self.assertEqual(ps.call_args_list[0].args[0][-1], 'pid=,comm=')
        self.assertEqual(ps.call_args_list[1].args[0][-1], 'pid=,args=')


if __name__ == '__main__':
    unittest.main()
