"""Private process and guard-frame tests; no production process is signalled."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import upstream_rbm400_observer_child_reaper as reaper

e=reaper.execution


class ChildReaperTests(unittest.TestCase):
    def test_wrong_child_identity_command_group_or_executable_never_signals(self):
        captured=dict(pid=123,ppid=456,pgid=456,started='original',command=reaper.PS_COMMAND,state='T')
        for change,path in ((dict(started='reused'),'/bin/ps'),(dict(ppid=789),'/bin/ps'),(dict(pgid=789),'/bin/ps'),(dict(command='other'),'/bin/ps'),(dict(state='S'),'/bin/ps'),({},'/other/ps')):
            with patch.object(e.identity,'process',return_value=dict(captured,**change)),patch.object(reaper,'executable',return_value=path),patch.object(reaper.os,'kill') as kill:
                with self.assertRaises(ValueError):reaper.terminate_child(captured)
                kill.assert_not_called()

    def test_only_child_sigkill_no_parent_or_group_signal(self):
        captured=dict(pid=123,ppid=456,pgid=456,started='original',command=reaper.PS_COMMAND,state='T')
        with patch.object(e.identity,'process',side_effect=[captured,None]),patch.object(reaper,'executable',return_value='/bin/ps'),patch.object(reaper.os,'kill') as kill,patch.object(reaper.os,'killpg') as killpg:
            reaper.terminate_child(captured)
            kill.assert_called_once_with(123,signal.SIGKILL);killpg.assert_not_called()

    def test_child_must_be_terminal_within_bound(self):
        captured=dict(pid=123,ppid=456,pgid=456,started='original',command=reaper.PS_COMMAND,state='T')
        with patch.object(e.identity,'process',return_value=captured),patch.object(reaper,'executable',return_value='/bin/ps'),patch.object(reaper.os,'kill'),patch.object(reaper.time,'sleep'):
            with self.assertRaises(RuntimeError):reaper.terminate_child(captured)

    def test_failed_ps_returns_to_same_guard_frame_and_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);baseline=dict(swap_used_bytes=123,compressor_bytes=0)
            path=folder/'first-scientific-baseline.json';path.write_text(json.dumps(baseline))
            guard=e.WorkerGuard.__new__(e.WorkerGuard)
            guard.phase='data';guard.limit=e.cap('data');guard.last=0.;guard.counter=0
            guard.plan={};guard.baseline=baseline;guard.baseline_sha=e.inputs.sha(path)
            guard.predecessors={};guard.captured=dict(pid=777)
            workers={'777':dict(bytes=1024,rss_bytes=1024)}
            with patch.object(e,'REPORT',folder),patch.object(e,'check_sources'),patch.object(e.os,'getpid',return_value=777),patch.object(e.resources,'footprints',side_effect=[RuntimeError('Process-list observation failed'),workers]),patch.object(e.memory,'memory',return_value=baseline),patch.object(e,'stop_child',return_value=True) as stop,patch.object(e.os,'kill') as kill,patch.object(e.os,'killpg') as killpg:
                guard()
                stop.assert_called_once_with(guard.captured)
                kill.assert_not_called();killpg.assert_not_called()
            self.assertEqual(guard.counter,1)
            self.assertEqual(guard.baseline_sha,e.inputs.sha(path))
            self.assertEqual(guard.baseline,baseline)
            failed=json.loads((folder/'data-worker-resource-request.json').read_text())
            self.assertIn('Process-list observation failed',failed['sample']['accounting_error'])
            self.assertTrue(failed['no_automatic_retry'])

    def test_actual_private_stopped_ps_with_full_pipe_is_retired(self):
        args=['ps','-axo','pid=,ppid=,stat=,rss=,lstart=,command=']
        full=subprocess.check_output(args,executable='/bin/ps')
        self.assertGreater(len(full),65536,'Actual fixture needs more than a64KiB pipe of listing output')
        parent=os.getpid()
        child=subprocess.Popen(args,executable='/bin/ps',stdout=subprocess.PIPE,start_new_session=True)
        try:
            time.sleep(.2)
            self.assertIsNone(child.poll(),'Private ps must still be blocked on its unread pipe')
            os.kill(child.pid,signal.SIGSTOP)
            captured=None
            for _ in range(100):
                captured=e.identity.process(child.pid)
                if captured and 'T' in captured['state']:break
                time.sleep(.01)
            self.assertIsNotNone(captured)
            self.assertEqual(captured['ppid'],parent)
            self.assertEqual(captured['pgid'],child.pid)
            self.assertEqual(captured['command'],reaper.PS_COMMAND)
            reaper.terminate_child(captured)
            self.assertEqual(child.wait(timeout=3),-signal.SIGKILL)
            partial=child.stdout.read()
            self.assertGreater(len(partial),0)
            self.assertLess(len(partial),len(full))
            self.assertEqual(os.getpid(),parent)
            print(json.dumps(dict(private_pipe_fixture=True,full_listing_bytes=len(full),partial_pipe_bytes=len(partial),child_only_SIGKILL=True,parent_signalled=False)))
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=3)
            child.stdout.close()


if __name__=='__main__':unittest.main()
