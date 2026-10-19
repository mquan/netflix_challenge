"""Check audited dimensions against the encoder and expose RNG compatibility risk."""
import dataclasses
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import masked_sequence as ms
from test_selection_objective import fixture


class PaddingAuditTests(unittest.TestCase):
    def test_actual_encoder_shape_and_cpu_inference_compatibility(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        d = fixture()
        c = ms.Config(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=2,
            context='session', time_bins=4, dropout=.1, seq_len=24, max_targets=8,
            pad_width=4, pad_rows=64, device='cpu')
        batch = ms.prepare_batch(d, np.array([0, 2, 4]), c, rng=np.random.default_rng(9))
        model = ms.SequenceModel(d['items'], c, ms.weeks(d))
        shapes = []
        hook = model.blocks[0].register_forward_pre_hook(lambda _, inputs: shapes.append(tuple(inputs[0].shape)))
        model.eval()
        with torch.no_grad():
            original = model.forward_batch(batch)
            model.config = dataclasses.replace(c, pad_rows=16)
            candidate = model.forward_batch(batch)
        hook.remove()
        self.assertEqual(shapes, [(64, batch['seq_width'], 8), (16, batch['seq_width'], 8)])
        torch.testing.assert_close(original, candidate, rtol=1e-6, atol=1e-6)
        # Fixed weights/seed do not preserve the training randomness stream when
        # padded shapes change. Future learning comparisons must be matched anew.
        model.train()
        states = []
        for pad in (64, 16):
            model.config = dataclasses.replace(c, pad_rows=pad)
            torch.manual_seed(23)
            model.forward_batch(batch)
            states.append(torch.get_rng_state())
        self.assertFalse(torch.equal(*states))


if __name__ == '__main__':
    unittest.main()
