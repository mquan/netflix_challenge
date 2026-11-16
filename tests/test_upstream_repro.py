"""Guard the fresh/reference boundary and dependency execution order."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/upstream_repro.py'
spec = importlib.util.spec_from_file_location('upstream_repro', SCRIPT)
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


def job(parents=(), inputs=(), outputs=()):
    return dict(inputs_from=list(parents), inputs=list(inputs), outputs=list(outputs), command='unused')


class UpstreamReproduction(unittest.TestCase):
    def test_implicit_producer_and_explicit_parent_order(self):
        jobs = {'base': job(outputs=['base.npy']), 'chain': job(['base'], outputs=['chain.npy']),
                'blend': job(inputs=['chain.npy', 'base.npy'])}
        self.assertEqual(repro.dependency_order(jobs, 'blend'), ['base', 'chain', 'blend'])

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            repro.dependency_order({'a': job(['b']), 'b': job(['a'])}, 'a')

    def test_nonzero_qualifying_outcomes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            fresh = base/'fresh'
            payload = fresh/'data/qual_ratings/qual_ratings.csv.gz'
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b'frozen sentinel payload')
            ratings = fresh/'data/qual/ratings.npy'
            ratings.parent.mkdir(parents=True)
            np.save(ratings, np.zeros(4, dtype=np.int8))
            repro.write(base/'fresh-training-plan.json', dict(synthetic_qualifying_payload_sha256=repro.sha(payload)))
            def git(args, **kwargs):
                return repro.PIN+'\n' if 'rev-parse' in args else 'data/qual_ratings/qual_ratings.csv.gz\n'
            with patch.object(repro, 'BASE', base), patch.object(repro, 'FRESH', fresh), patch.object(repro.subprocess, 'check_output', side_effect=git):
                repro.validate_fresh()
                np.save(ratings, np.array([0, 0, 5, 0], dtype=np.int8))
                with self.assertRaisesRegex(ValueError, 'contain outcomes'):
                    repro.validate_fresh()

    def test_existing_unreceipted_vectors_cannot_be_skipped_as_done(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            fresh = base/'fresh'
            output = fresh/'preds_new/model.probex.npy'
            output.parent.mkdir(parents=True)
            np.save(output, np.ones(3))
            repro.write(base/'fresh-data-receipt.json', {})
            plan = dict(all_jobs={'model': job(outputs=['preds_new/model.probex.npy'])})
            with patch.object(repro, 'BASE', base), patch.object(repro, 'FRESH', fresh), patch.object(repro, 'validate_fresh', return_value=plan), patch.object(repro.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Unreceipted'):
                    repro.model_job('model', execute=True)
                run.assert_not_called()

    def test_modified_completed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            fresh = base/'fresh'
            output = fresh/'preds_new/model.probex.npy'
            output.parent.mkdir(parents=True)
            np.save(output, np.ones(3))
            repro.write(base/'fresh-data-receipt.json', {})
            repro.write(base/'fresh-receipts/model.json', {'outputs': {'preds_new/model.probex.npy': repro.sha(output)}})
            np.save(output, np.full(3, 2.0))
            plan = dict(all_jobs={'model': job(outputs=['preds_new/model.probex.npy'])})
            with patch.object(repro, 'BASE', base), patch.object(repro, 'FRESH', fresh), patch.object(repro, 'validate_fresh', return_value=plan), patch.object(repro.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'outputs changed'):
                    repro.model_job('model', execute=True)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
