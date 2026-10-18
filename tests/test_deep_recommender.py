"""The deep autoencoder must never use probe ratings as model inputs."""
import pathlib
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from deep_recommender import DeepRecommender, predict, training_csr
from gbdt_blend import PROBE


class DeepRecommenderTests(unittest.TestCase):
    def test_probe_label_isolation_and_sparse_mask(self):
        probe = np.zeros(3, dtype=PROBE)
        probe['user'] = [1, 0, 0]
        probe['item'] = [1, 3, 2]
        probe['rating'] = [1, 2, 3]
        data = dict(users=2, items=3, train_offsets=np.array([0, 2, 3]),
                    train_items=np.array([1, 2, 3]), train_ratings=np.array([1, 5, 4]),
                    probe=probe)
        csr = training_csr(data)
        np.testing.assert_array_equal(csr.toarray(), [[1, 5, 0], [0, 0, 4]])
        torch.manual_seed(3)
        net = DeepRecommender(3, (4, 5), .5)
        first = predict(net, csr, probe, 'cpu', 2)
        probe['rating'] = 5
        second = predict(net, csr, probe, 'cpu', 1)
        np.testing.assert_allclose(first, second, atol=1e-6)
        self.assertTrue(np.isfinite(first).all())


if __name__ == '__main__':
    unittest.main()
