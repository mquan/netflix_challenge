"""Additional delegation and real scheduler-role checks for the AEX route."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_primary_guarded_aex as scheduler
import upstream_import_aex_guarded as routing


class AexDelegationTest(unittest.TestCase):
    def test_only_already_running_aex_is_added_to_delegated_jobs(self):
        self.assertEqual(scheduler.DELEGATED-scheduler.original.DELEGATED,{'aex-128'})
        self.assertNotIn('aex-16',scheduler.DELEGATED)
        self.assertNotIn('aex-64',scheduler.DELEGATED)
        jobs={name:{} for name in ['aex-128','aex-16','cfnade-96','tsvdx4p-16000lm']}
        with patch.object(scheduler.original,'DELEGATED',scheduler.DELEGATED):
            storage,delegated=scheduler.original.holds(jobs)
        self.assertEqual(delegated,{'aex-128','cfnade-96'})
        self.assertEqual(storage,{'tsvdx4p-16000lm'})

    def test_router_only_accepts_actual_new_scheduler_command(self):
        transfer=routing.inherited.transfer;children=[]
        try:
            for marker in ['scripts/upstream_primary_guarded.py','scripts/upstream_primary_guarded_aex.py']:
                children.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',marker],start_new_session=True))
            saved={k:getattr(transfer,k) for k in ['REPORT','auxiliary','followon','PRIMARY_FRAGMENT']}
            with patch.multiple(transfer,**saved):
                routing.configure()
                with self.assertRaisesRegex(ValueError,'changed'):
                    with transfer.paused_primary(children[0].pid):self.fail('Old scheduler command must be rejected')
                with transfer.paused_primary(children[1].pid):
                    self.assertIn('T',next(r for r in transfer.process_rows() if r['pid']==children[1].pid)['state'])
                self.assertNotIn('T',next(r for r in transfer.process_rows() if r['pid']==children[1].pid)['state'])
        finally:
            for child in children:
                if child.poll() is None:
                    os.kill(child.pid,signal.SIGCONT);child.terminate();child.wait(timeout=5)


if __name__=='__main__':unittest.main()
