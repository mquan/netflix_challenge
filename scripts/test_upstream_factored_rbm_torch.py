"""CPU Torch implementation checked against separate scalar CD equations.

An actual MPS replay remains a separate required preflight; CPU success does
not claim MPS support, throughput, training completion or prediction accuracy.
"""
import unittest
import numpy as np
from test_upstream_factored_rbm_reference import fixture, scalar
import upstream_factored_rbm_reference as reference
import upstream_factored_rbm_torch as kernel


class TorchKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tensors(self, model, batch, hu, vu, dtype):
        t = self.torch
        return ({k: t.tensor(v, dtype=dtype) for k, v in model.items()},
                {k: t.tensor(v, dtype=t.int64) for k, v in batch.items()},
                t.tensor(hu, dtype=dtype), t.tensor(vu, dtype=dtype))

    def test_all_cd3_states_gradients_and_full_update_against_scalar(self):
        t = self.torch; m, b, hu, vu = fixture()
        for dtype, tolerance in ((t.float64, 3e-13), (t.float32, 2e-6)):
            for tile in (1, 2, 16384):
                tm, tb, th, tv = self.tensors(m, b, hu, vu, dtype)
                # Compare against the actual rounded input values for float32.
                rm = {k: v.numpy().astype(np.float64) for k, v in tm.items()}
                want = scalar(rm, b, th.numpy().astype(np.float64), tv.numpy().astype(np.float64))
                got = kernel.batch_gradients(tm, tb, th, tv, tile)
                for key, value in want.items():
                    np.testing.assert_allclose(got[key].numpy(), value, atol=tolerance, rtol=tolerance)
                momentum = {k: t.full_like(v, .03) for k, v in tm.items()}
                expected_grad = reference.batch_gradients(rm, b, th.numpy().astype(np.float64), tv.numpy().astype(np.float64))
                rw, rv = reference.update(rm, {k: v.numpy().astype(np.float64) for k, v in momentum.items()}, expected_grad)
                kw, kv = kernel.update(tm, momentum, got)
                for key in tm:
                    np.testing.assert_allclose(kw[key].numpy(), rw[key], atol=tolerance, rtol=tolerance)
                    np.testing.assert_allclose(kv[key].numpy(), rv[key], atol=tolerance, rtol=tolerance)
                    np.testing.assert_array_equal(tm[key].numpy(), np.asarray(m[key], dtype=tm[key].numpy().dtype))

    def test_exact_categorical_boundaries_and_bipolar_zero(self):
        t = self.torch; m, b, hu, vu = fixture()
        m = {k: np.zeros_like(v) for k, v in m.items()}
        vu[:] = [0., .2, .4, .6, .8, .999]
        args = self.tensors(m, b, hu[:1], vu[:1], t.float64)
        got = kernel.batch_gradients(*args, tile_rows=1)
        np.testing.assert_array_equal(got['visible_state'].numpy(), [0, 0, 1, 2, 3, 4])
        for key in ('p', 'q', 'd', 'bh', 'positive_expected', 'negative_expected'):
            np.testing.assert_array_equal(got[key].numpy(), 0)

    def test_partial_batch_uses_actual_count_and_decays_untouched_items(self):
        t = self.torch; m, b, hu, vu = fixture()
        tm, tb, th, tv = self.tensors(m, b, hu, vu, t.float64)
        g = {k: t.zeros_like(v) for k, v in tm.items()}; g['batch_users'] = 1
        momentum = {k: t.full_like(v, .03) for k, v in tm.items()}
        _, mv = kernel.update(tm, momentum, g)
        np.testing.assert_allclose(mv['p'].numpy(), .027 - .02 * .001 * m['p'])
        np.testing.assert_allclose(mv['bv'].numpy(), .027)
        g['batch_users'] = 0
        with self.assertRaises(ValueError): kernel.update(tm, momentum, g)


if __name__ == '__main__': unittest.main()
