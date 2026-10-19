"""Retained snapshot epochs must override the trainer's later terminal epoch."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recover_neural_recipes import selected_epoch


class EpochTests(unittest.TestCase):
    def test_retained_snapshot_uses_its_own_epoch(self):
        self.assertEqual(selected_epoch('MT1-refit.refit.e10.f32', {'selected_epoch': 12}), 10)

    def test_terminal_alias_requires_recorded_epoch(self):
        self.assertEqual(selected_epoch('MT5-refit.f32', {'selected_epoch': 10}), 10)
        with self.assertRaises(ValueError):
            selected_epoch('unknown.f32', {})


if __name__ == '__main__':
    unittest.main()
