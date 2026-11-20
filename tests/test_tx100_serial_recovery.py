import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_tx100_serial_memory_recovery import safe

class ActualOversubscriptionRegression(unittest.TestCase):
    def test_monitor_loss_suspends_only_matching_original_pid(self):
        from unittest.mock import patch
        import signal
        import upstream_tx100_serial_memory_failsafe as watchdog
        first=dict(pid=100,started='original',command='original native')
        second=dict(pid=101,started='original',command='original owner')
        replacement=dict(second,started='different process')
        with patch.dict(watchdog.CAPTURED,dict(tx=first,tx_owner=second),clear=True),patch.object(watchdog.owners,'process',side_effect=[first,replacement]),patch.object(watchdog.os,'kill') as kill,patch.object(watchdog,'REPORT',Path('/nonexistent/netflix-failsafe-test')):
            self.assertEqual(watchdog.suspend_originals('monitor lost'),[100])
            kill.assert_called_once_with(100,signal.SIGSTOP)

    def test_measured_failed_overlap_rejected(self):
        workers={str(i):dict(bytes=g*2**30,rss_bytes=g*2**30) for i,g in enumerate([17,8,4,4,1])}
        baseline=dict(swap_used_bytes=0)
        self.assertFalse(safe(workers,baseline,baseline,22*2**30))
    def test_existing_serial_native_and_paused_state_fit(self):
        workers={str(i):dict(bytes=g*2**30,rss_bytes=g*2**30) for i,g in enumerate([17,4,1])}
        baseline=dict(swap_used_bytes=5*2**30)
        self.assertTrue(safe(workers,baseline,baseline,21*2**30))
        self.assertFalse(safe(workers,baseline,baseline,18*2**30))
        self.assertFalse(safe(workers,dict(swap_used_bytes=baseline['swap_used_bytes']+129*2**20),baseline,21*2**30))

if __name__=='__main__':unittest.main()
