import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_research_journey as journey


class Own330QualifyingPublication(unittest.TestCase):
    path = journey.ROOT / 'experiments/upstream-own330-assembly/independent-canonical-verification.json'

    def latest(self):
        return json.loads(self.path.read_text())

    def test_actual_complete_qualifying_proof(self):
        latest = self.latest()
        self.assertEqual(journey.validate_own330_qualifying(self.path, latest), (True, True))
        self.assertTrue(latest['both_groups_improved'])
        self.assertFalse(latest['both_groups_below_target'])

    def test_changed_final_frozen_vector_is_rejected(self):
        target = journey.ROOT / 'artifacts/upstream-own330-assembly/own7-own330-known-calibration.f32'
        original = Path.read_bytes
        def changed(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            return value+b'corrupt' if path == target else value
        latest = self.latest()
        with patch.object(Path, 'read_bytes', changed):
            with self.assertRaises(ValueError):
                journey.validate_own330_qualifying(self.path, latest)

    def test_changed_original_refit_code_is_rejected(self):
        target = journey.ROOT / 'scripts/upstream_own330_full_refit.py'
        original = Path.read_bytes
        def changed(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            return value+b'corrupt' if path == target else value
        latest = self.latest()
        with patch.object(Path, 'read_bytes', changed):
            with self.assertRaises(ValueError):
                journey.validate_own330_qualifying(self.path, latest)

    def test_missing_original_full_model_replay_is_rejected(self):
        target = journey.ROOT / 'experiments/upstream-own330-full-refit/independent-verification.json'
        original = Path.read_text
        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == target:
                value = json.loads(text); value['records'][1]['actual_models_reopened'] = 2
                return json.dumps(value)
            return text
        latest = self.latest()
        with patch.object(Path, 'read_text', changed):
            with self.assertRaises(ValueError):
                journey.validate_own330_qualifying(self.path, latest)


if __name__ == '__main__':
    unittest.main()
