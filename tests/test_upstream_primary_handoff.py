"""Admission coverage and actual idle/busy scheduler signal behavior."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_primary_handoff as m


class PrimaryHandoffTest(unittest.TestCase):
    def test_holds_cover_every_parallel_heap_rank_and_external_cf(self):
        jobs={name:{} for name in ['tsvdx4p-1000lm','tsvdx4p-8000lm',
            'tsvdx4p-16000lm','tsvdx4p-6000lm__knnf','cfnade-48',
            'cfnade-96','cfnade-96r','cfnade-12','attn-32','aex-128']}
        storage,delegated=m.successor.holds(jobs)
        self.assertEqual(storage,{k for k in jobs if k.startswith('tsvdx4p-')})
        self.assertEqual(delegated,{'cfnade-48','cfnade-96','cfnade-96r'})
        self.assertNotIn('cfnade-12',storage|delegated)
        self.assertNotIn('attn-32',storage|delegated)

    def worker(self,code):
        marker='owned-primary-handoff-test'
        child=subprocess.Popen([sys.executable,'-c',code,marker],start_new_session=True)
        time.sleep(.15)
        self.addCleanup(self.cleanup,child)
        return marker,child

    @staticmethod
    def cleanup(child):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGCONT)
            os.killpg(child.pid,signal.SIGTERM)
            child.wait(timeout=5)

    def test_idle_scheduler_exits_before_handoff_returns(self):
        marker,child=self.worker('import time;time.sleep(60)')
        with patch.object(m.transfer,'PRIMARY_FRAGMENT',marker):
            self.assertTrue(m.terminate_idle(child.pid,{}))
        self.assertIsNotNone(child.poll())

    def test_live_child_prevents_termination_and_parent_resumes(self):
        code='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);time.sleep(60)'
        marker,child=self.worker(code)
        with patch.object(m.transfer,'PRIMARY_FRAGMENT',marker):
            self.assertFalse(m.terminate_idle(child.pid,{}))
        row=next(r for r in m.transfer.process_rows() if r['pid']==child.pid)
        self.assertNotIn('T',row['state']);self.assertIsNone(child.poll())

    def test_changed_pid_identity_is_never_terminated(self):
        with self.assertRaisesRegex(ValueError,'changed'):
            m.terminate_idle(os.getpid(),{})


if __name__=='__main__':unittest.main()
