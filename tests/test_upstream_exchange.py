"""Exercise independent scoring, artifact integrity, and outer-label isolation."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import upstream_exchange as exchange


class ExchangeChecks(unittest.TestCase):
    def test_users_stay_together_and_folds_ignore_row_order(self):
        users = np.repeat(np.arange(100), 3)
        folds = exchange.user_folds(users)
        self.assertEqual(len(set(folds)), 5)
        self.assertTrue(np.all(folds[::3] == folds[1::3]))
        permutation = np.random.default_rng(5).permutation(len(users))
        np.testing.assert_array_equal(exchange.user_folds(users[permutation]), folds[permutation])

    def test_outer_label_poisoning_cannot_change_that_folds_predictions(self):
        rng = np.random.default_rng(12)
        x = rng.normal(3, .5, (300, 8))
        y = rng.integers(1, 6, 300).astype(float)
        folds = exchange.user_folds(np.repeat(np.arange(100), 3))
        original = exchange.crossfit(x, y, folds)
        poisoned = y.copy()
        poisoned[folds == 0] += 1000
        altered = exchange.crossfit(x, poisoned, folds)
        np.testing.assert_array_equal(original[folds == 0], altered[folds == 0])
        self.assertGreater(np.max(np.abs(original[folds != 0]-altered[folds != 0])), .1)

    def test_evaluation_refuses_imported_extra_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'exactly the eight'):
                exchange.evaluate({'public-column': np.ones(10)}, np.ones(10), np.arange(10), Path(folder))

    def run_artifact_case(self, tampered):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            reference = base/'reference'
            reference.mkdir()
            labels = np.array([1, 2, 3, 4, 5]*4, dtype=np.int8)
            users = np.arange(len(labels), dtype=np.int32)
            values = np.full(len(labels), 3, dtype=np.float32)
            np.save(reference/'ratings.npy', labels)
            data_dir = reference/'data/probex'
            data_dir.mkdir(parents=True)
            np.save(data_dir/'ratings.npy', labels)
            np.save(reference/'fixture.npy', values)
            def npz(**arrays):
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                return stream.getvalue()
            sha = lambda data: hashlib.sha256(data).hexdigest()
            development = npz(labels=labels, users=users)
            prediction = npz(prediction=values)
            name = exchange.JOBS[0]
            plan = json.dumps(exchange.PLAN).encode()
            entry = dict(artifact=name+'.npz', artifact_sha256=sha(prediction),
                         source_npy_sha256='original-npy-sha', prediction_bytes_sha256=exchange.array_sha(values),
                         comparison={'fresh_rmse':exchange.metric(values, labels)})
            manifest = dict(source_commit=exchange.upstream.PIN, plan_sha256=sha(plan),
                            development_sha256=sha(development), labels_bytes_sha256=exchange.array_sha(labels),
                            users_bytes_sha256=exchange.array_sha(users), jobs={name:entry})
            receipt = dict(source_commit=exchange.upstream.PIN, job=name,
                           outputs={'preds_new/'+name+'.probex.npy':'original-npy-sha'})
            prefix = exchange.ARTIFACT_PREFIX
            blobs = {prefix+'manifest.json':json.dumps(manifest).encode(), prefix+'meta-plan.json':plan,
                     prefix+'development.npz':development, prefix+name+'.npz':prediction+(b'changed' if tampered else b''),
                     prefix+name+'.receipt.json':json.dumps(receipt).encode()}
            exchange.upstream.write(base/'reference-manifest.json', {'entries':[dict(name=name,path='fixture.npy',sha256=exchange.upstream.sha(reference/'fixture.npy'))]})
            with patch.object(exchange, 'LOCAL', base/'local'), patch.object(exchange.upstream, 'BASE', base), patch.object(exchange.upstream, 'REFERENCE', reference), patch.object(exchange, 'worker_blob', side_effect=lambda ref,path:blobs[path]):
                return exchange.ingest_worker('worker-ref')

    def test_real_compressed_artifact_is_independently_rescored(self):
        _, _, _, predictions, checks = self.run_artifact_case(False)
        self.assertEqual(list(predictions), [exchange.JOBS[0]])
        self.assertTrue(checks[0]['artifact_verified'])
        self.assertAlmostEqual(checks[0]['fresh_rmse'], np.sqrt(2))

    def test_changed_compressed_artifact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            self.run_artifact_case(True)

class LiveQueueCompatibility(unittest.TestCase):
    def test_absent_reference_keeps_live_renderer_running_without_claiming_a_score(self):
        import contextlib
        import upstream_repro_queue as queue
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            fresh = base/'fresh'
            reference = base/'reference'
            for root in [fresh, reference]:
                (root/'data/probex').mkdir(parents=True)
                np.save(root/'data/probex/ratings.npy', np.array([1, 3, 5], dtype=np.int8))
            output = fresh/'preds_new/rbmx2-40.probex.npy'
            output.parent.mkdir()
            np.save(output, np.array([2, 3, 4], dtype=np.float32))
            exchange.upstream.write(base/'fresh-receipts/rbmx2-40.json',
                {'outputs': {'preds_new/rbmx2-40.probex.npy':exchange.upstream.sha(output)},
                 'started':'2026-11-16T08:00:00+00:00','finished':'2026-11-16T08:01:00+00:00'})
            exchange.upstream.write(base/'queue-state.json', {'state':'running'})
            with patch.object(exchange.upstream, 'BASE', base), patch.object(exchange.upstream, 'FRESH', fresh), patch.object(exchange.upstream, 'REFERENCE', reference), patch.object(exchange.upstream, 'validate_fresh'), patch.object(exchange, 'export_existing_queue') as export, contextlib.redirect_stdout(io.StringIO()):
                exchange.upstream.compare_model('rbmx2-40')
            canonical = export.call_args.args[1]
            self.assertIsNone(canonical['released_rmse'])
            self.assertFalse(canonical['reference_available'])
            legacy = exchange.upstream.read(base/'comparisons/rbmx2-40.json')
            self.assertTrue(np.isnan(legacy['released_rmse']))
            # This is the same renderer already loaded by the remote Python process.
            with patch.object(queue, 'BASE', base), patch.object(queue, 'ROOT', base):
                queue.render({'state':'running','current':None})
            self.assertIn('nan', (base/'UPSTREAM_PROGRESS.md').read_text())


if __name__ == '__main__':
    unittest.main()
