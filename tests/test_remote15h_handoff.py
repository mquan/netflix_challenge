"""Failure-boundary checks for the original-producer remote handoff."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import upstream_remote15h as remote


class Remote15hTests(unittest.TestCase):
    def test_recipes_exactly_match_original_registry(self):
        plan = remote.read(remote.PLAN)
        registry = remote.read(ROOT / 'experiments/upstream-all-pipeline-coverage/registry.json')['3']['jobs']
        self.assertEqual(sum(len(j['published_columns']) for j in plan['jobs']), 9)
        for job in plan['jobs']:
            self.assertEqual(job['original_registry_job'], registry[job['name']])
            env = ' '.join(k + '=' + v for k, v in job['environment'].items())
            command = 'cargo build --release --bin fit && ' + env + ' ' + ' '.join(job['command']) + ' '
            self.assertEqual(command, registry[job['name']]['command'])
            self.assertEqual(registry[job['name']]['inputs_from'], ['ingest', 'split_3'])

    def test_inherited_model_knobs_cannot_leak(self):
        with patch.dict(os.environ, {'EPOCHS': '1', 'QUAL_ONLY': '1', 'SEED': '999',
                                    'CF_RESUME': '1', 'CARGO_TARGET_DIR': '/unowned'}):
            env = remote.environment({'EPSAVE': '1'})
        for key in ['EPOCHS', 'QUAL_ONLY', 'SEED', 'CF_RESUME', 'CARGO_TARGET_DIR']:
            self.assertNotIn(key, env)
        self.assertEqual(env['EPSAVE'], '1')
        self.assertEqual(env['RAYON_NUM_THREADS'], '4')

    def test_both_complete_ordered_phase_traces_required(self):
        phase = ''.join(f'Epoch {i:02} | RMSE .9\n' for i in range(1, 9))
        remote.trace_check(phase * 2, 8)
        for wrong in [phase, phase + phase.rsplit('Epoch 08', 1)[0], phase * 3,
                      phase.replace('Epoch 04', 'Epoch 05') * 2]:
            with self.assertRaises(ValueError):
                remote.trace_check(wrong, 8)

    def test_prediction_validation_rejects_bad_transfers(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'prediction.npy'
            for a in [np.zeros(4, dtype=np.float64), np.zeros(3, dtype=np.float32),
                      np.zeros((4, 1), dtype=np.float32),
                      np.array([0, 0, np.nan, 0], dtype=np.float32),
                      np.array([0, 0, np.inf, 0], dtype=np.float32)]:
                np.save(p, a)
                with self.assertRaises(ValueError):
                    remote.vector(p, 4)
            np.save(p, np.arange(4, dtype=np.float32))
            np.testing.assert_array_equal(remote.vector(p, 4), np.arange(4))

    def test_changed_file_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / 'source.rs'
            p.write_bytes(b'original source')
            hashes = {'source.rs': remote.sha(p)}
            remote.check_files(root, hashes)
            p.write_bytes(b'changed source')
            with self.assertRaises(ValueError):
                remote.check_files(root, hashes)


if __name__ == '__main__':
    unittest.main()
