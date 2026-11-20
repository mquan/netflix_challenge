"""GPU-lock lifetime and original full-capacity admission boundaries."""
import fcntl
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import upstream_knn3pair2_opportunistic_queue as scheduler


class Knn3SchedulerTests(unittest.TestCase):
    def test_original_sixteen_gib_reserve_unchanged(self):
        previous = dict(swap_used_bytes=0)
        current = dict(swap_used_bytes=0, compressor_bytes=0)
        self.assertTrue(scheduler.admitted(previous, current, 7 * 2**30, 13 * 2**30, {}))
        self.assertFalse(scheduler.admitted(previous, current, 8 * 2**30, 13 * 2**30, {}))
        self.assertFalse(scheduler.admitted(previous, current, 7 * 2**30, 12 * 2**30, {}))
        self.assertFalse(scheduler.admitted(previous, current, 7 * 2**30, 13 * 2**30, {'pid': 'prior study'}))
        self.assertFalse(scheduler.admitted(previous, dict(current, compressor_bytes=5 * 2**30), 7 * 2**30, 13 * 2**30, {}))
        self.assertFalse(scheduler.admitted(previous, dict(current, swap_used_bytes=32 * 2**20), 7 * 2**30, 13 * 2**30, {}))

    def test_rejected_headroom_releases_gpu_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'slot.lock'
            with p.open('a') as slot, p.open('a') as other:
                self.assertFalse(scheduler.try_slot(slot, lambda: False))
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(other, fcntl.LOCK_UN)

    def test_admitted_headroom_retains_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'slot.lock'
            with p.open('a') as slot, p.open('a') as other:
                self.assertTrue(scheduler.try_slot(slot, lambda: True))
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(slot, fcntl.LOCK_UN)
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(other, fcntl.LOCK_UN)

    def test_busy_gpu_lock_does_not_block(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'slot.lock'
            with p.open('a') as other, p.open('a') as slot:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(scheduler.try_slot(slot, lambda: True))
                fcntl.flock(other, fcntl.LOCK_UN)
                self.assertTrue(scheduler.try_slot(slot, lambda: True))
                fcntl.flock(slot, fcntl.LOCK_UN)

    def test_exception_during_recheck_releases_gpu_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'slot.lock'
            def fail():
                raise RuntimeError('headroom observation failed')
            with p.open('a') as slot, p.open('a') as other:
                with self.assertRaises(RuntimeError):
                    scheduler.try_slot(slot, fail)
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(other, fcntl.LOCK_UN)


if __name__ == '__main__':
    unittest.main()
