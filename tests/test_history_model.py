import pathlib
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from history_model import HistoryNet, batch, objective, predict, split_rows
from gbdt_blend import PROBE


def fixture():
    probe = np.zeros(3, dtype=PROBE)
    probe['user'] = [1, 0, 0]
    probe['item'] = [1, 3, 2]
    probe['day'] = [11, 12, 13]
    probe['rating'] = [1, 2, 3]
    return dict(users=2, items=3, train_offsets=np.array([0, 2, 3]),
                train_items=np.array([1, 2, 3]), train_ratings=np.array([1, 5, 4]),
                train_days=np.array([1, 10, 2]), probe=probe)


class HistoryTests(unittest.TestCase):
    def test_target_exclusion(self):
        rng = np.random.default_rng(2)
        for n in (0, 1, 2, 20, 500):
            for _ in range(30):
                c, t = split_rows(n, rng, 16, 8)
                self.assertFalse(set(c) & set(t))
                self.assertEqual(len(set(c)), len(c))
                self.assertEqual(len(set(t)), len(t))
                self.assertLessEqual(len(c), 16)
                self.assertLessEqual(len(t), 8)
                if n:
                    self.assertGreater(len(t), 0)

    def test_labels_do_not_enter_context_or_predictions(self):
        d = fixture()
        for session in (False, True):
            torch.manual_seed(1)
            net = HistoryNet(3, 4, 8, session).eval()
            first = predict(net, d, 'cpu', 8, 2)
            d['probe']['rating'] = 5
            second = predict(net, d, 'cpu', 8, 1)
            np.testing.assert_allclose(first, second, atol=1e-6)
            self.assertTrue(np.all((first >= 1) & (first <= 5)))
            values = batch(d, [0], np.random.default_rng(42), 8, 8)
            context_items = set(((values['token'][values['token'] > 0] - 1) // 5).tolist())
            target_items = set(values['item'][values['mask']].tolist())
            self.assertFalse(context_items & target_items)

    def test_ordinal_loss_padding_and_backward(self):
        logits = torch.zeros(1, 2, 5, requires_grad=True)
        mask = torch.tensor([[True, False]])
        labels = torch.tensor([[2, 0]])
        first = objective(logits, labels, mask, torch.ones(1))
        changed = labels.clone()
        changed[0, 1] = 4
        self.assertEqual(float(first.detach()), float(objective(logits, changed, mask, torch.ones(1)).detach()))
        first.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
