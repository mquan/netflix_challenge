import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_own330_queue as queue
import upstream_own330_watchdog as watchdog
import upstream_own330_mps as study


class CompleteStudyScheduling(unittest.TestCase):
    def test_complete_context_partition_and_original_training_budget(self):
        expected=list(range(63,68))+list(range(91,221))+list(range(302,392))
        self.assertEqual(study.CONTEXT,expected)
        self.assertEqual(len(study.SIGNALS),332)
        self.assertEqual(sorted(study.SIGNALS+study.CONTEXT),list(range(557)))
        self.assertEqual(study.CONFIG['nn_hidden'],128)
        self.assertEqual(study.CONFIG['context_rank'],32)
        self.assertEqual(study.CONFIG['nn_epochs'],40)
        self.assertEqual(study.CONFIG['nn_batch'],8192)
        self.assertEqual(study.CONFIG['seeds'],[2026,7,13])

    def test_only_incomplete_dependency_wait_owner_can_be_parked_and_restored(self):
        expected=dict(pid=222,ppid=1,pgid=222,state='S',started='original',command='python -u upstream_high_capacity_original_queue.py')
        stopped=dict(expected,state='T')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/'queue';report.mkdir()
            path=root/'experiments/upstream-high-capacity-original/queue-status.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(state='waiting_for_complete_own176_and_preceding_original_studies')))
            with patch.object(queue.u,'ROOT',root),patch.object(queue,'REPORT',report),patch.object(queue,'PARKED',None), \
                 patch.object(queue,'ready',return_value=False),patch.object(queue.owners,'children',return_value=[]), \
                 patch.object(queue.subprocess,'check_output',return_value='222 python -u upstream_high_capacity_original_queue.py\n'), \
                 patch.object(queue.owners,'process',side_effect=[expected,stopped,stopped]),patch.object(queue.os,'kill') as kill:
                queue.park_only_idle_native_queue();self.assertEqual(queue.PARKED,expected)
                queue.release_parking()
                self.assertEqual(kill.call_args_list[0].args,(222,signal.SIGSTOP))
                self.assertEqual(kill.call_args_list[1].args,(222,signal.SIGCONT))

    def test_native_model_or_reused_pid_never_receives_priority_signal(self):
        with patch.object(queue,'ready',return_value=True),patch.object(queue.subprocess,'check_output') as scan,patch.object(queue.os,'kill') as kill:
            queue.park_only_idle_native_queue();scan.assert_not_called();kill.assert_not_called()
        original=dict(pid=222,pgid=222,started='original',command='original queue')
        replacement=dict(original,started='new process',state='T')
        with patch.object(watchdog.owners,'process',return_value=replacement),patch.object(watchdog.os,'kill') as kill:
            watchdog.release_idle_queue(dict(parked_highcap=original));kill.assert_not_called()


if __name__=='__main__':unittest.main()
