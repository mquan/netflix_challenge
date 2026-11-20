"""Full neural recipe retention and enforced compact execution resource bounds."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import upstream_cf96_prefix2030_compact_mps_context as model
import upstream_cf96_prefix2030_compact_queue as queue

REFERENCE_SCALER = model.n.scaled


class CF96PrefixExecutionTests(unittest.TestCase):
    def test_full_model_recipe_and_backend_unchanged(self):
        self.assertEqual(model.CONFIG, model.recipe.CONFIG)
        self.assertEqual(model.CONFIG['nn_hidden'], 128)
        self.assertEqual(model.CONFIG['context_rank'], 32)
        self.assertEqual(model.CONFIG['nn_epochs'], 40)
        self.assertEqual(model.CONFIG['nn_batch'], 8192)
        self.assertEqual(model.CONFIG['seeds'], [2026, 7, 13])
        model.configure()
        self.assertIs(model.n.train_nn, model.recipe.mps_backend.train_nn)
        self.assertIs(model.n.nn_predict, model.recipe.mps_backend.predict)
        self.assertIs(model.n.scaled, REFERENCE_SCALER)

    def test_ten_gib_own_reserve_under_original_twentyfour_gib_total(self):
        previous = dict(swap_used_bytes=0)
        current = dict(swap_used_bytes=0, compressor_bytes=0)
        self.assertTrue(queue.admitted(previous, current, 13 * 2**30, 13 * 2**30))
        self.assertFalse(queue.admitted(previous, current, 14 * 2**30, 13 * 2**30))
        self.assertFalse(queue.admitted(previous, current, 13 * 2**30, 12 * 2**30))
        self.assertFalse(queue.admitted(previous, dict(current, compressor_bytes=5 * 2**30), 13 * 2**30, 13 * 2**30))
        self.assertFalse(queue.admitted(previous, dict(current, swap_used_bytes=32 * 2**20), 13 * 2**30, 13 * 2**30))


if __name__ == '__main__':
    unittest.main()
