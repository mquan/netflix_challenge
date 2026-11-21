"""Resource candidate tests; no scientific processes are signalled or launched."""
import ctypes
import unittest
from unittest.mock import patch
import upstream_research_footprints_native as observer


def row(command='python upstream_example.py', ppid=999, rss=100, started='same'):
    return dict(command=command, ppid=ppid, rss_bytes=rss, started=started, state='S')


class NativeObserverTests(unittest.TestCase):
    def sample(self, before, after, usages):
        with patch.object(observer, 'processes', side_effect=[before, after]), \
                patch.object(observer, 'usage', side_effect=usages):
            return observer.footprints(1)

    def test_sdk_layout(self):
        self.assertEqual(ctypes.sizeof(observer.RUsageV0), 96)
        self.assertEqual(observer.RUsageV0.resident_size.offset, 64)
        self.assertEqual(observer.RUsageV0.phys_footprint.offset, 72)
        self.assertEqual(observer.RUsageV0.proc_start_abstime.offset, 80)

    def test_domains_have_independent_conservative_maxima(self):
        result = self.sample({1: row(rss=100)}, {1: row(rss=200)},
                             [dict(bytes=300, rss_bytes=150, started=7),
                              dict(bytes=250, rss_bytes=400, started=7)])
        self.assertEqual(result['1']['bytes'], 300)
        self.assertEqual(result['1']['rss_bytes'], 400)

    def test_process_reuse_fails(self):
        with self.assertRaises(RuntimeError):
            self.sample({1: row()}, {1: row()},
                        [dict(bytes=3, rss_bytes=4, started=7),
                         dict(bytes=3, rss_bytes=4, started=8)])

    def test_command_change_fails(self):
        with self.assertRaises(RuntimeError):
            self.sample({1: row()}, {1: row('changed')},
                        [dict(bytes=3, rss_bytes=4, started=7)] * 2)

    def test_peer_appearance_and_disappearance_fail(self):
        for after in ({}, {1: row(), 2: row()}):
            with self.subTest(after=after), self.assertRaises(RuntimeError):
                self.sample({1: row()}, after,
                            [dict(bytes=3, rss_bytes=4, started=7)] * 2)

    def test_sampling_error_never_becomes_zero(self):
        with self.assertRaises(PermissionError):
            self.sample({1: row()}, {1: row()}, [PermissionError('denied')])

    def test_original_scope_and_descendants(self):
        rows = {1: row('unrelated'), 2: row('python upstream_a.py'),
                3: row('python verify_upstream_a.py'),
                4: row('/a/target/release/model'),
                5: row(str(observer.ROOT)+'/runner'),
                6: row('child renamed', ppid=2),
                7: row('grandchild', ppid=6), 8: row('unrelated')}
        self.assertEqual(observer.selected(rows, 1), set(range(1, 8)))

    def test_required_worker_missing_fails(self):
        with self.assertRaises(RuntimeError):
            self.sample({}, {}, [])

    def test_64_bit_counters(self):
        value = dict(bytes=2**40, rss_bytes=2**42, started=2**60)
        result = self.sample({1: row()}, {1: row()}, [value, value])
        self.assertEqual(result['1']['rss_bytes'], 2**42)


if __name__ == '__main__':
    unittest.main()
