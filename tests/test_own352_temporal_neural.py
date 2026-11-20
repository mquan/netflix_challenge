from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upstream_own352_temporal_mps as study
import upstream_own352_temporal_queue as queue


class CompleteTemporalNeuralStudy(unittest.TestCase):
    def test_original_model_and_budget_with_complete_partition(self):
        self.assertEqual(len(study.SIGNALS), 354)
        self.assertEqual(len(study.CONTEXT), 225)
        self.assertEqual(sorted(study.SIGNALS+study.CONTEXT), list(range(579)))
        expected = dict(study.previous.CONFIG, architecture=study.CONFIG['architecture'])
        self.assertEqual(study.CONFIG, expected)
        for key, value in dict(nn_hidden=128, context_rank=32, nn_epochs=40,
                               nn_patience=4, nn_batch=8192).items():
            self.assertEqual(study.CONFIG[key], value)
        self.assertEqual(study.CONFIG['seeds'], [2026, 7, 13])

    def test_actual_model_has_all_signals_context_and_hidden_capacity(self):
        model = study.model_factory(579, study.CONFIG)
        self.assertEqual(tuple(model.project.weight.shape), (32, 354))
        self.assertEqual(tuple(model.gate[0].weight.shape), (128, 225))
        self.assertEqual(tuple(model.base[0].weight.shape), (128, 579))
        with self.assertRaises(ValueError):
            study.model_factory(557, study.CONFIG)

    def test_reservation_and_swap_growth_are_not_relaxed(self):
        current = dict(swap_used_bytes=0, compressor_bytes=0)
        class Disk:
            free = 20*2**30
        with patch.object(queue.shutil, 'disk_usage', return_value=Disk()):
            workers = {'native':dict(bytes=7*2**30, rss_bytes=7*2**30)}
            self.assertTrue(queue.safe(workers, current, current, 16*2**30))
            workers['native']['bytes'] = 8*2**30
            self.assertFalse(queue.safe(workers, current, current, 16*2**30))
            self.assertFalse(queue.safe({}, dict(current, swap_used_bytes=129*2**20), current))


if __name__ == '__main__':
    unittest.main()
