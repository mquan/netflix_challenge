import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_primary_handoff_recover as recovery


class RecoveryChecks(unittest.TestCase):
    def test_present_pid_is_rejected_without_signalling(self):
        with self.assertRaisesRegex(ValueError,'PID still present'):
            recovery.require_absent(os.getpid(),{},recovery.transfer.process_rows())

    def test_real_live_wrapper_group_blocks_absent_predecessor(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        try:
            with self.assertRaisesRegex(ValueError,'wrapper group'):
                recovery.require_absent(-1,dict(child_pid=child.pid),recovery.transfer.process_rows())
            self.assertIsNone(child.poll())
        finally:
            child.terminate();child.wait(timeout=5)
        # Isolate the absent-scheduler branch; the real research scheduler can
        # be running while this test verifies its owned wrapper has exited.
        recovery.require_absent(-1,dict(child_pid=child.pid),[])

    def test_other_live_scheduler_blocks_recovery(self):
        rows=[dict(pid=123,ppid=1,pgid=123,state='S',command='python scripts/upstream_primary_guarded_aex.py')]
        with self.assertRaisesRegex(ValueError,'Another primary'):
            recovery.require_absent(-1,{},rows)


if __name__=='__main__':unittest.main()
