"""Scientific invariants for full paired trajectories and unchanged parent bits."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import upstream_complete_trajectory_ops as ops


class CompleteTrajectoryTests(unittest.TestCase):
    def test_native_inventory_keeps_every_epoch_and_component(self):
        prefix = 'artifacts/native/preds/tsvdx5-1200lm'
        components = ['bias', 'mf', 'nsvd1', 'nbr']
        stems = [prefix] + [prefix + '_ep%02d' % k for k in range(1, 24)]
        stems += [prefix + '-' + name for name in components]
        outputs = {s + '.' + q + '.npy': 'a' * 64 for s in stems for q in ops.COUNTS}
        outputs.update({prefix + '.out': 'b' * 64,
                        prefix + '.trainx.npy': 'c' * 64, prefix + '.fulltrain.npy': 'd' * 64})
        rows = ops.native_rows(outputs, prefix, 23, components, training=True)
        self.assertEqual(len(rows), 28)
        self.assertEqual([row['id'] for row in rows[:2]], ['native:tsvdx5-1200lm_ep01', 'native:tsvdx5-1200lm_ep02'])
        self.assertEqual(rows[23]['id'], 'native:tsvdx5-1200lm')
        self.assertEqual([r['id'] for r in rows[-4:]], ['native:tsvdx5-1200lm-' + c for c in components])
        del outputs[prefix + '_ep17.qual.npy']
        with self.assertRaises(ValueError):
            ops.native_rows(outputs, prefix, 23, components, training=True)

    def test_alias_requires_both_queries_and_retains_first_occurrence(self):
        original = {q: ['1' * 64] * 391 for q in ops.COUNTS}
        candidates = [dict(id=name) for name in ['old alias', 'different qualifying', 'new pair', 'new alias']]
        hashes = {'probex': ['1' * 64, '1' * 64, '2' * 64, '2' * 64],
                  'qual': ['1' * 64, '3' * 64, '4' * 64, '4' * 64]}
        kept, aliases = ops.partition(original, candidates, hashes)
        self.assertEqual([r['id'] for r in kept], ['different qualifying', 'new pair'])
        self.assertEqual([(r['id'], r['alias_of']) for r in aliases],
                         [('old alias', 'original391:000'), ('new alias', 'new pair')])

    def test_extension_preserves_all_parent_bits_and_intercept(self):
        z = np.arange(3 * 467, dtype=np.float64).reshape(3, 467)
        z[:, -1] = [1., -0., 2.]
        f = np.arange(3 * 617, dtype='<f4').reshape(3, 617)
        a = np.array([0., 3.6, 6.], dtype='<f4')
        b = np.array([6., 1., -0.], dtype='<f4')
        zz, ff = ops.extend(z, f, [a, b])
        self.assertEqual(zz.shape, (3, 469))
        self.assertEqual(ff.shape, (3, 619))
        np.testing.assert_array_equal(zz[:, :466].view('u8'), z[:, :466].view('u8'))
        np.testing.assert_array_equal(zz[:, -1].view('u8'), z[:, -1].view('u8'))
        np.testing.assert_array_equal(ff[:, :617].view('u4'), f.view('u4'))
        np.testing.assert_array_equal(zz[:, 466:468], np.column_stack([a, b]))
        np.testing.assert_array_equal(ff[:, 617].view('u4'), (a - np.float32(3.6)).view('u4'))
        empty_z, empty_f = ops.extend(z, f, [])
        np.testing.assert_array_equal(empty_z.view('u8'), z.view('u8'))
        np.testing.assert_array_equal(empty_f.view('u4'), f.view('u4'))

    def test_binding_conflicts_and_path_escape_are_rejected(self):
        with self.assertRaises(ValueError):
            ops.merge_bindings({'a': '1' * 64}, {'a': '2' * 64})
        with self.assertRaises(ValueError):
            ops.merge_bindings({}, {'../recovered.csv': '1' * 64})
        with self.assertRaises(ValueError):
            ops.merge_bindings({}, {'a': 'truncated'})

    def test_full_hash_preserves_bytes_and_file_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'fixture'
            payload = bytes(range(256)) * 32769
            path.write_bytes(payload)
            before = path.stat()
            self.assertEqual(ops.sha(path), hashlib.sha256(payload).hexdigest())
            after = path.stat()
            for key in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_nlink', 'st_flags']:
                self.assertEqual(getattr(before, key), getattr(after, key))

    def test_independent_expansion_replays_clipping_and_signed_zero(self):
        import verify_upstream_complete_trajectory_data as independent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'predictions.npy'
            values = np.zeros(ops.COUNTS['probex'], dtype='<f4')
            values[:3] = [-1., -0., 7.]; np.save(path, values)
            row = dict(paired_files={'probex': path.name})
            z = np.arange(3*467, dtype=np.float64).reshape(3, 467)
            f = np.arange(3*617, dtype='<f4').reshape(3, 617)
            with mock.patch.object(independent.u, 'ROOT', root):
                clipped = ops.scalar(root, row, 'probex', 0, 3)
                actual = independent.extend(z, f, [row], 'probex', 0, 3)
                expected = ops.extend(z, f, [clipped])
                for left, right, dtype in zip(actual, expected, ['u8', 'u4']):
                    np.testing.assert_array_equal(left.view(dtype), right.view(dtype))
                self.assertEqual(clipped.view('u4')[1], 2**31)
                values[0] = np.inf; np.save(path, values)
                with self.assertRaises(ValueError): ops.scalar(root, row, 'probex', 0, 3)
                with self.assertRaises(ValueError): independent.scalar(row, 'probex', 0, 3)

    def test_queue_original_resource_boundaries(self):
        import upstream_complete_trajectory_data_queue as queue
        base = dict(compressor_bytes=0, swap_used_bytes=1000)
        workers = {'1': dict(bytes=1, rss_bytes=1)}
        with mock.patch.object(queue.shutil, 'disk_usage', return_value=type('D', (), {'free': 20*2**30})()):
            self.assertTrue(queue.safe(workers, base, base, 3*2**30, 64*2**20))
            self.assertFalse(queue.safe(workers, dict(base, compressor_bytes=5*2**30), base))
            self.assertFalse(queue.safe(workers, dict(base, swap_used_bytes=1001+128*2**20), base))
            self.assertFalse(queue.safe({'1': dict(bytes=24*2**30, rss_bytes=1)}, base, base))
            self.assertFalse(queue.safe({'1': dict(bytes=1, rss_bytes=24*2**30)}, base, base))
        with mock.patch.object(queue.shutil, 'disk_usage', return_value=type('D', (), {'free': 18.75*2**30-1})()):
            self.assertFalse(queue.safe(workers, base, base))

    def test_internal_guard_suspends_and_rechecks_same_baseline(self):
        import upstream_complete_trajectory_data as study
        import signal
        with tempfile.TemporaryDirectory() as tmp:
            guard = object.__new__(study.Guard)
            guard.pid = 123; guard.queue = Path(tmp)
            guard.baseline = dict(compressor_bytes=0, swap_used_bytes=1000)
            own = {'123': dict(bytes=1, rss_bytes=1)}
            current = [dict(guard.baseline, compressor_bytes=5*2**30), dict(guard.baseline)]
            with mock.patch.object(study.resources, 'footprints', return_value=own), mock.patch.object(study.memory, 'memory', side_effect=current), mock.patch.object(study.shutil, 'disk_usage', return_value=type('D', (), {'free': 20*2**30})()), mock.patch.object(study.os, 'killpg') as suspend:
                guard.check()
                suspend.assert_called_once_with(123, signal.SIGSTOP)
            self.assertEqual(guard.baseline, dict(compressor_bytes=0, swap_used_bytes=1000))
            self.assertTrue((Path(tmp)/'self-guard-123.json').exists())
            with mock.patch.object(study.resources, 'footprints', side_effect=[RuntimeError('unavailable sample'), own]), mock.patch.object(study.memory, 'memory', return_value=guard.baseline), mock.patch.object(study.shutil, 'disk_usage', return_value=type('D', (), {'free': 20*2**30})()), mock.patch.object(study.os, 'killpg') as suspend:
                guard.check()
                suspend.assert_called_once_with(123, signal.SIGSTOP)


if __name__ == '__main__':
    unittest.main()
