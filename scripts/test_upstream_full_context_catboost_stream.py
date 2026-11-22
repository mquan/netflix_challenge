"""Numerical handoff checks only; no production source or model admission."""
import hashlib
import unittest
import numpy as np
import upstream_full_context_catboost_stream as module


class Reader:
    def __init__(self, n, keep, latent):
        self.prediction = range(n)
        self.d, self.f = module.dimensions(keep, latent)
        self.calls = []

    def block(self, lo, hi):
        self.calls.append((lo, hi))
        row = np.arange(lo, hi, dtype=np.float64)[:, None]
        z = np.sin(row * .001 + np.arange(self.d)[None, :] * .01)
        f = np.cos(row * .003 + np.arange(self.f)[None, :] * .02).astype('<f4')
        return z, f


class StreamTests(unittest.TestCase):
    def fixture(self, keep=False, latent=False, n=19):
        reader = Reader(n, keep, latent)
        rng = np.random.default_rng(392)
        weights = rng.normal(0, .11, (5, 2, reader.d))
        features = np.empty((n, reader.f + 1), dtype='<f4')
        baseline = np.empty((5, 2, n), dtype='<f8')
        hashes = [hashlib.sha256(), hashlib.sha256()]
        expected = np.empty_like(baseline)
        raw = np.empty((n, reader.f), dtype='<f4')
        for lo in range(0, n, module.BLOCK):
            hi = min(lo + module.BLOCK, n)
            z, f = reader.block(lo, hi)
            hashes[0].update(z.tobytes()); hashes[1].update(f.tobytes())
            raw[lo:hi] = f
            for fold in range(5):
                for pop in range(2):
                    expected[fold, pop, lo:hi] = np.clip(z @ weights[fold, pop], 1.05, 4.95)
        reader.calls.clear()
        return reader, weights, features, baseline, [h.hexdigest() for h in hashes], expected, raw

    def test_every_row_and_ten_population_bits_all_four_full_widths(self):
        for keep in (False, True):
            for latent in (False, True):
                with self.subTest(keep=keep, latent=latent):
                    reader, w, x, b, h, expected, raw = self.fixture(keep, latent)
                    result = module.materialize(reader, w, h, x, b, retain_scalar=keep,
                                                latent=latent, guard=lambda: None)
                    self.assertEqual(b.tobytes(), expected.tobytes())
                    self.assertEqual(x[:, :-1].tobytes(), raw.tobytes())
                    self.assertTrue((x[:, -1] == 0).all())
                    self.assertEqual(result['models_fitted'], 0)
                    for fold in range(5):
                        for pop in range(2):
                            module.install_baseline(x, b[fold, pop], lambda: None)
                            self.assertEqual(x[:, -1].tobytes(), expected[fold, pop].astype('<f4').tobytes())
                            self.assertEqual(x[:, :-1].tobytes(), raw.tobytes())

    def test_crosses_original_block_boundary_and_rejects_operator_mutation(self):
        reader, w, x, b, h, expected, raw = self.fixture(True, True, 8193)
        count = [0]
        def guard(): count[0] += 1
        module.materialize(reader, w, h, x, b, retain_scalar=True, latent=True, guard=guard)
        self.assertEqual(reader.calls, [(0, 8192), (8192, 8193)])
        self.assertEqual(b.tobytes(), expected.tobytes())
        self.assertEqual(x[:, :-1].tobytes(), raw.tobytes())
        self.assertGreaterEqual(count[0], 23)
        h[1] = '0' * 64
        with self.assertRaises(ValueError):
            module.materialize(reader, w, h, x, b, retain_scalar=True, latent=True, guard=guard)

    def test_invalid_shape_weights_stream_or_baseline_never_admitted(self):
        reader, w, x, b, h, _, _ = self.fixture()
        for weights, hashes in [(w[:, :1], h), (w.astype('f4'), h), (w, ['z'*64, h[1]])]:
            with self.assertRaises(ValueError):
                module.materialize(reader, weights, hashes, x, b, retain_scalar=False,
                                   latent=False, guard=lambda: None)
        w[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            module.materialize(reader, w, h, x, b, retain_scalar=False, latent=False, guard=lambda: None)
        for invalid in (np.full(len(x), np.nan), np.full(len(x), 5.), np.ones(len(x), dtype='f4')):
            with self.assertRaises(ValueError): module.install_baseline(x, invalid, lambda: None)

    def test_guard_interrupt_propagates_without_returning_completion(self):
        reader, w, x, b, h, _, _ = self.fixture()
        def guard(): raise RuntimeError('same-frame guard stop')
        with self.assertRaisesRegex(RuntimeError, 'same-frame'):
            module.materialize(reader, w, h, x, b, retain_scalar=False, latent=False, guard=guard)
        self.assertEqual(reader.calls, [])


if __name__ == '__main__': unittest.main()
