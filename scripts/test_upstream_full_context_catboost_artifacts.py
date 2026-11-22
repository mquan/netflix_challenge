"""Actual saved-model and failure-preservation fixtures, not production fits."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_full_context_catboost as backend
import upstream_full_context_catboost_artifacts as module


class ArtifactTests(unittest.TestCase):
    def fit(self):
        rng = np.random.default_rng(399)
        x = rng.normal(size=(24, 713)).astype('<f4')
        y = (x[:, 0] * .1).astype(np.float64)
        return x, backend.train(x, y, 2026, 3, lambda **kw: None)

    def test_actual_FD_save_and_authenticated_reload_every_prediction_bit(self):
        x, model = self.fit()
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); path = root / 'fold' / 'model.cbm'
            receipt = module.save_model(path, model, seed=2026, iterations=3, width=713,
                root=root, guard=lambda **kw: calls.append(kw))
            self.assertEqual(receipt['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(path.stat().st_nlink, 1)
            self.assertFalse(path.with_name(path.name + '.staging').exists())
            with backend.actual_model(path, receipt, receipt['signature'], 713, lambda: None) as loaded:
                self.assertEqual(backend.predict(model, x, lambda: None).tobytes(),
                                 backend.predict(loaded, x, lambda: None).tobytes())
            self.assertIn({'extra_disk_bytes': module.MODEL_RESERVE}, calls)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                module.save_model(path, model, seed=2026, iterations=3, width=713, root=root, guard=lambda **kw: None)
            self.assertEqual(before, path.read_bytes())

    def test_native_failure_retains_staging_and_disallows_retry(self):
        _, model = self.fit()
        class Broken:
            def get_metadata(self): return model.get_metadata()
            feature_names_ = model.feature_names_
            tree_count_ = model.tree_count_
            def save_model(self, **kwargs): raise RuntimeError('native failure')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); path = root / 'model.cbm'
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                module.save_model(path, Broken(), seed=2026, iterations=3, width=713, root=root, guard=lambda **kw: None)
            self.assertFalse(path.exists())
            self.assertTrue(path.with_name(path.name + '.staging').exists())
            with self.assertRaises(ValueError):
                module.save_model(path, model, seed=2026, iterations=3, width=713, root=root, guard=lambda **kw: None)

    def test_wrong_config_or_output_alias_rejected_before_native_save(self):
        _, model = self.fit()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for seed, count, width in [(2027, 3, 713), (2026, 2, 713), (2026, 3, 864)]:
                with self.assertRaises(ValueError):
                    module.save_model(root/'bad.cbm', model, seed=seed, iterations=count, width=width,
                                      root=root, guard=lambda **kw: None)
            (root/'real').mkdir(); (root/'alias').symlink_to(root/'real', target_is_directory=True)
            with self.assertRaises(ValueError):
                module.save_model(root/'alias'/'bad.cbm', model, seed=2026, iterations=3, width=713,
                                  root=root, guard=lambda **kw: None)
            self.assertFalse((root/'bad.cbm.staging').exists())


if __name__ == '__main__': unittest.main()
