import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upstream_high_capacity_original_queue as queue
import upstream_high_capacity_watchdog as watchdog


class HighCapacityExecutionRegression(unittest.TestCase):
    def test_22gib_reservation_rejects_preserved_native_overlap(self):
        current = dict(swap_used_bytes=6 * 2**30)
        workers = {'native': dict(bytes=17 * 2**30, rss_bytes=17 * 2**30)}
        self.assertFalse(queue.safe(workers, current, current, 23 * 2**30, 22 * 2**30))
        idle = {'voting': dict(bytes=1 * 2**30, rss_bytes=1 * 2**30)}
        self.assertTrue(queue.safe(idle, current, current, 23 * 2**30, 22 * 2**30))
        self.assertFalse(queue.safe(idle, dict(swap_used_bytes=current['swap_used_bytes']+129*2**20), current, 23*2**30))

    def test_no_build_or_training_before_complete_dependencies(self):
        with tempfile.TemporaryDirectory(dir=queue.u.ROOT) as directory:
            report = Path(directory)
            # Hash binding formation is incidental; test the actual queue
            # path that must wait before lane acquisition or any Popen.
            with patch.object(queue, 'REPORT', report), patch.object(queue.u, 'sha', return_value='frozen'), \
                 patch.object(queue, 'ready', return_value=False), \
                 patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt('end test')), \
                 patch.object(queue.signal, 'signal'), \
                 patch.object(queue.subprocess, 'Popen') as popen, \
                 patch.object(queue.resources, 'footprints') as footprints:
                with self.assertRaises(KeyboardInterrupt):queue.run()
                popen.assert_not_called();footprints.assert_not_called()

    def test_watchdog_never_signals_reused_native_pid(self):
        original = dict(pid=101, pgid=101, started='original', command='original native')
        replacement = dict(original, started='replacement')
        with patch.object(watchdog, 'CAPTURED', original), \
             patch.object(watchdog.identities, 'process', return_value=replacement), \
             patch.object(watchdog.os, 'killpg') as kill:
            watchdog.suspend('monitor lost')
            kill.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(watchdog, 'REPORT', Path(directory)), \
             patch.object(watchdog, 'CAPTURED', original), \
             patch.object(watchdog.identities, 'process', return_value=original), \
             patch.object(watchdog.os, 'killpg') as kill:
            watchdog.suspend('monitor lost')
            kill.assert_called_once_with(101, signal.SIGSTOP)


if __name__ == '__main__':unittest.main()
