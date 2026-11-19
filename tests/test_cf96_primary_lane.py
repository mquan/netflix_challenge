import subprocess
import sys
import unittest
import upstream_cf96_primary_lane as lane


class IdleBoundaryTest(unittest.TestCase):
    def test_busy_owner_is_resumed_and_child_keeps_running(self):
        token='cf96-lane-test-busy-owner'
        code='import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); print(p.pid,flush=True); p.wait()'
        owner=subprocess.Popen([sys.executable,'-u','-c',code,token],stdout=subprocess.PIPE,text=True)
        child_pid=int(owner.stdout.readline())
        try:
            self.assertIsNone(lane.stop_only_if_idle(owner.pid,token))
            self.assertNotIn('T',lane.process(owner.pid)['state'])
            self.assertIsNotNone(lane.process(child_pid))
            self.assertNotIn('T',lane.process(child_pid)['state'])
        finally:
            subprocess.run(['kill','-TERM',str(child_pid)],check=False)
            owner.wait(timeout=5);owner.stdout.close()

    def test_idle_owner_exits_and_wrong_identity_is_untouched(self):
        token='cf96-lane-test-idle-owner'
        owner=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',token])
        try:
            with self.assertRaises(ValueError):lane.stop_only_if_idle(owner.pid,'wrong-owner')
            self.assertNotIn('T',lane.process(owner.pid)['state'])
            self.assertIsNone(lane.stop_only_if_idle(owner.pid,token,lambda:False))
            self.assertNotIn('T',lane.process(owner.pid)['state'])
            record=lane.stop_only_if_idle(owner.pid,token)
            self.assertEqual(record['pid'],owner.pid)
            owner.wait(timeout=5)
        finally:
            if owner.poll() is None:owner.terminate();owner.wait(timeout=5)


if __name__=='__main__':unittest.main()
